import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pypsa

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

spec = importlib.util.spec_from_file_location("workflow_validation", SCRIPTS_DIR / "validation.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
spec = importlib.util.spec_from_file_location("workflow_brownfield", SCRIPTS_DIR / "add_brownfield.py")
brownfield = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brownfield)


class ElectricityDemandAlignmentTests(unittest.TestCase):
    def test_owid_and_ember_reference_parsers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            owid = tmp / "owid.csv"
            owid.write_text(
                "year,iso_code,electricity_demand\n"
                "2025,USA,10.5\n"
                "2025,CAN,2.0\n"
                "2024,USA,9.0\n",
                encoding="utf-8",
            )
            parsed = validation._load_electricity_demand_reference_twh(
                {"year": 2025, "csv": str(owid), "schema": "owid_energy"}
            )
            self.assertAlmostEqual(parsed.loc["US"], 10.5)
            self.assertAlmostEqual(parsed.loc["CA"], 2.0)

            ember = tmp / "ember.csv"
            ember.write_text(
                "Entity,Code,Year,Other renewables,Bioenergy,Solar,Wind,Hydropower,Nuclear,Oil,Gas,Coal\n"
                "United States,USA,2025,1,2,3,4,5,6,7,8,9\n"
                "Australia,AUS,2024,1,1,1,1,1,1,1,1,1\n",
                encoding="utf-8",
            )
            fallback_owid = tmp / "fallback_owid.csv"
            fallback_owid.write_text(
                "year,iso_code,electricity_demand\n"
                "2024,SAU,12.0\n"
                "2024,AUS,99.0\n",
                encoding="utf-8",
            )
            parsed = validation._load_electricity_demand_reference_twh(
                {
                    "year": 2025,
                    "csv": str(ember),
                    "schema": "ember_generation_balance",
                    "fallback_references": [
                        {"year": 2024, "csv": str(ember), "schema": "ember_generation_balance"},
                        {
                            "year": 2024,
                            "csv": str(fallback_owid),
                            "schema": "owid_energy",
                            "target_column": "electricity_demand",
                        },
                    ],
                }
            )
            self.assertAlmostEqual(parsed.loc["US"], 45.0)
            self.assertAlmostEqual(parsed.loc["AU"], 9.0)
            self.assertAlmostEqual(parsed.loc["SA"], 12.0)

            bad = tmp / "bad.csv"
            bad.write_text("year,iso_code\n2025,USA\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validation._load_electricity_demand_reference_twh(
                    {"year": 2025, "csv": str(bad), "schema": "owid_energy"}
                )

    def _toy_demand_network(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2025-01-01", periods=2, freq="h"))
        n.add("Bus", "US 0", carrier="AC")
        n.add("Bus", "US gas", carrier="gas")
        n.add("Bus", "US heat", carrier="urban heat")
        n.add("Bus", "CA 0", carrier="AC")
        n.buses.loc[["US 0", "US gas", "US heat"], "country"] = "US"
        n.buses.loc["CA 0", "country"] = "CA"
        n.add("Load", "US time electric", bus="US 0", carrier="AC")
        n.add("Load", "US static electric", bus="US 0", carrier="industry electricity", p_set=10.0)
        n.add("Load", "US gas load", bus="US gas", carrier="services gas", p_set=100.0)
        n.add("Load", "CA unreferenced electric", bus="CA 0", carrier="AC", p_set=8.0)
        n.loads_t.p_set = pd.DataFrame(
            {"US time electric": [20.0, 20.0]},
            index=n.snapshots,
        )
        n.add("Link", "US heat pump", bus0="US 0", bus1="US heat", carrier="heat pump")
        n.links_t.p0 = pd.DataFrame({"US heat pump": [10.0, 10.0]}, index=n.snapshots)
        return n

    def test_alignment_scales_only_electric_loads_and_subtracts_links(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ref = tmp / "owid.csv"
            # TWh: total reference 0.00005, link withdrawals 0.00002, direct target 0.00003.
            # Direct modeled load is 0.00006 TWh, so electric loads scale by 0.5.
            ref.write_text("year,iso_code,electricity_demand\n2025,USA,0.00005\n", encoding="utf-8")
            n = self._toy_demand_network()
            cfg = {
                "global_specific": {
                    "electricity_demand_alignment": {
                        "enable": True,
                        "audit_dir": str(tmp),
                        "anchors": {
                            "2025": {
                                "year": 2025,
                                "csv": str(ref),
                                "schema": "owid_energy",
                                "electricity_demand_end_use_link_output_bus_carrier_substrings": ["heat"],
                            }
                        },
                    }
                }
            }
            validation.align_country_electricity_demand_to_owid(n, 2025, cfg)
            self.assertAlmostEqual(float(n.loads_t.p_set["US time electric"].sum()), 20.0)
            self.assertAlmostEqual(float(n.loads.at["US static electric", "p_set"]), 5.0)
            self.assertAlmostEqual(float(n.loads.at["US gas load", "p_set"]), 100.0)
            self.assertAlmostEqual(float(n.loads.at["CA unreferenced electric", "p_set"]), 8.0)
            factors = pd.read_csv(tmp / "electricity_demand_alignment_factors_2025.csv")
            self.assertAlmostEqual(float(factors.loc[factors.country.eq("US"), "factor"].iloc[0]), 0.5)

    def test_carry_forward_uses_factor_file_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pd.DataFrame({"country": ["US"], "factor": [0.25]}).to_csv(
                tmp / "electricity_demand_alignment_factors_2025.csv",
                index=False,
            )
            n = self._toy_demand_network()
            cfg = {
                "global_specific": {
                    "electricity_demand_alignment": {
                        "enable": True,
                        "audit_dir": str(tmp),
                        "carry_forward_anchor_year": 2025,
                        "anchors": {"2025": {"year": 2025, "csv": str(tmp / "unused.csv")}},
                    }
                }
            }
            validation.align_country_electricity_demand_to_owid(n, 2030, cfg)
            self.assertAlmostEqual(float(n.loads_t.p_set["US time electric"].sum()), 10.0)
            self.assertAlmostEqual(float(n.loads.at["US static electric", "p_set"]), 2.5)
            self.assertAlmostEqual(float(n.loads.at["US gas load", "p_set"]), 100.0)


class FixedZeroCapacityCleanupTests(unittest.TestCase):
    def test_cleanup_removes_only_fixed_zero_capacity_assets(self):
        n = pypsa.Network()
        n.add("Bus", "A", carrier="AC")
        n.add("Bus", "B", carrier="AC")
        n.add(
            "Generator",
            "fixed-zero generator",
            bus="A",
            p_nom=0.0,
            p_nom_min=0.0,
            p_nom_extendable=False,
        )
        n.add(
            "Generator",
            "zero build option",
            bus="A",
            p_nom=0.0,
            p_nom_min=0.0,
            p_nom_extendable=True,
        )
        n.add(
            "Generator",
            "fixed-positive generator",
            bus="A",
            p_nom=1.0,
            p_nom_min=1.0,
            p_nom_extendable=False,
        )
        n.add(
            "Link",
            "fixed-zero link",
            bus0="A",
            bus1="B",
            p_nom=0.0,
            p_nom_min=0.0,
            p_nom_extendable=False,
        )
        n.add(
            "Link",
            "fixed-zero battery charger",
            bus0="A",
            bus1="B",
            carrier="battery charger",
            p_nom=0.0,
            p_nom_min=0.0,
            p_nom_extendable=False,
        )
        n.add(
            "Line",
            "fixed-zero line",
            bus0="A",
            bus1="B",
            s_nom=0.0,
            s_nom_min=0.0,
            s_nom_extendable=False,
            x=0.1,
        )
        n.add(
            "Store",
            "fixed-zero store",
            bus="A",
            e_nom=0.0,
            e_nom_min=0.0,
            e_nom_extendable=False,
        )
        n.add(
            "StorageUnit",
            "fixed-zero storage unit",
            bus="A",
            p_nom=0.0,
            p_nom_min=0.0,
            p_nom_extendable=False,
        )

        removed = validation.remove_fixed_zero_capacity_components(n)

        self.assertEqual(removed, {"Generator": 1, "Link": 1, "Store": 0})
        self.assertNotIn("fixed-zero generator", n.generators.index)
        self.assertNotIn("fixed-zero link", n.links.index)
        self.assertIn("fixed-zero battery charger", n.links.index)
        self.assertIn("zero build option", n.generators.index)
        self.assertIn("fixed-positive generator", n.generators.index)
        self.assertIn("fixed-zero line", n.lines.index)
        self.assertIn("fixed-zero store", n.stores.index)
        self.assertIn("fixed-zero storage unit", n.storage_units.index)

    def test_cleanup_removes_only_complete_fixed_zero_battery_triplets(self):
        n = pypsa.Network()
        n.add("Bus", "A", carrier="AC")
        for key in ("dead", "incomplete", "extendable", "positive"):
            n.add("Bus", f"{key} battery bus", carrier="battery")

        def add_store(key, capacity=0.0, extendable=False):
            n.add(
                "Store",
                f"{key} battery",
                bus=f"{key} battery bus",
                carrier="battery",
                e_nom=capacity,
                e_nom_min=capacity,
                e_nom_extendable=extendable,
            )

        def add_link(key, kind, capacity=0.0, extendable=False):
            battery_bus = f"{key} battery bus"
            if kind == "charger":
                bus0, bus1 = "A", battery_bus
            else:
                bus0, bus1 = battery_bus, "A"
            n.add(
                "Link",
                f"{key} battery {kind}",
                bus0=bus0,
                bus1=bus1,
                carrier=f"battery {kind}",
                p_nom=capacity,
                p_nom_min=capacity,
                p_nom_extendable=extendable,
            )

        add_store("dead")
        add_link("dead", "charger")
        add_link("dead", "discharger")

        add_store("incomplete")
        add_link("incomplete", "charger")

        add_store("extendable", extendable=True)
        add_link("extendable", "charger", extendable=True)
        add_link("extendable", "discharger", extendable=True)

        add_store("positive", capacity=5.0)
        add_link("positive", "charger", capacity=2.0)
        add_link("positive", "discharger", capacity=2.0)

        removed = validation.remove_fixed_zero_capacity_components(n)

        self.assertEqual(removed, {"Generator": 0, "Link": 2, "Store": 1})
        self.assertNotIn("dead battery", n.stores.index)
        self.assertNotIn("dead battery charger", n.links.index)
        self.assertNotIn("dead battery discharger", n.links.index)
        self.assertIn("incomplete battery", n.stores.index)
        self.assertIn("incomplete battery charger", n.links.index)
        self.assertIn("extendable battery", n.stores.index)
        self.assertIn("positive battery", n.stores.index)


class BevChargerRepairTests(unittest.TestCase):
    def test_forecast_year_rechecks_fixed_bev_charger_capacity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n = pypsa.Network()
            n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
            n.add("Bus", "US 0", carrier="AC")
            n.add("Bus", "US 0 EV battery", carrier="Li ion")
            n.add(
                "Link",
                "US 0 BEV charger",
                bus0="US 0",
                bus1="US 0 EV battery",
                carrier="BEV charger",
                p_nom=1.0,
                p_nom_min=1.0,
                p_nom_extendable=False,
                p_max_pu=1.0,
                efficiency=0.9,
            )
            n.add(
                "Load",
                "US 0 land transport EV",
                bus="US 0 EV battery",
                carrier="land transport EV",
            )
            n.loads_t.p_set = pd.DataFrame(
                {"US 0 land transport EV": [10.0, 20.0]},
                index=n.snapshots,
            )
            config = {
                "global_specific": {
                    "debug_feasibility_sanitize": {"audit_dir": tmpdir}
                }
            }

            validation.apply_final_historical_capacity_validation_fixes(
                n,
                2030,
                config,
                context="forecast-test",
            )

            expected = (20.0 / 0.9) * 1.01
            self.assertAlmostEqual(n.links.at["US 0 BEV charger", "p_nom"], expected)
            self.assertAlmostEqual(n.links.at["US 0 BEV charger", "p_nom_min"], expected)
            self.assertTrue(
                (Path(tmpdir) / "bev_charger_capacity_repair_2030.csv").is_file()
            )


class HistoricalStockFreezeTests(unittest.TestCase):
    def test_2025_freeze_only_fixes_positive_installed_stock(self):
        n = pypsa.Network()
        n.add("Bus", "A", carrier="AC")
        n.add("Bus", "B", carrier="battery")
        n.add(
            "Generator",
            "installed solar",
            bus="A",
            carrier="solar",
            p_nom=5.0,
            p_nom_extendable=True,
        )
        n.add(
            "Generator",
            "future solar option",
            bus="A",
            carrier="solar",
            p_nom=0.0,
            p_nom_extendable=True,
        )
        for name, carrier, capacity in (
            ("installed CCGT", "CCGT", 4.0),
            ("future CCGT option", "CCGT", 0.0),
            ("installed charger", "battery charger", 2.0),
            ("future charger option", "battery charger", 0.0),
        ):
            n.add(
                "Link",
                name,
                bus0="A",
                bus1="B",
                carrier=carrier,
                p_nom=capacity,
                p_nom_extendable=True,
            )
        n.add(
            "Store",
            "installed battery",
            bus="B",
            carrier="battery",
            e_nom=8.0,
            e_nom_extendable=True,
        )
        n.add(
            "Store",
            "future battery option",
            bus="B",
            carrier="battery",
            e_nom=0.0,
            e_nom_extendable=True,
        )
        n.add(
            "Line",
            "installed AC line",
            bus0="A",
            bus1="B",
            carrier="AC",
            s_nom=10.0,
            s_nom_extendable=True,
            x=0.1,
        )
        n.add(
            "Line",
            "future AC option",
            bus0="A",
            bus1="B",
            carrier="AC",
            s_nom=0.0,
            s_nom_extendable=True,
            x=0.1,
        )

        brownfield.freeze_year2025_historical_electric_assets(n)

        for table, installed, future, nominal in (
            (n.generators, "installed solar", "future solar option", "p_nom"),
            (n.links, "installed CCGT", "future CCGT option", "p_nom"),
            (n.links, "installed charger", "future charger option", "p_nom"),
            (n.stores, "installed battery", "future battery option", "e_nom"),
            (n.lines, "installed AC line", "future AC option", "s_nom"),
        ):
            self.assertFalse(bool(table.at[installed, f"{nominal}_extendable"]))
            self.assertEqual(table.at[installed, f"{nominal}_min"], table.at[installed, nominal])
            self.assertTrue(bool(table.at[future, f"{nominal}_extendable"]))
            self.assertEqual(table.at[future, nominal], 0.0)


class TransmissionLimitTests(unittest.TestCase):
    def test_gtd_caps_allocate_by_pair_and_apply_zero_and_missing_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            existing = tmp / "existing.csv"
            existing.write_text(
                "pathway,from_region,to_region,from_country,to_country,max_flow,max_counter_flow,voltage,distance\n"
                "TRNUSACAN,USAXX,CANXX,USA,CAN,70,70,-,-\n"
                "TRNUSAMEX,USAXX,MEXXX,USA,MEX,0,0,-,-\n",
                encoding="utf-8",
            )
            planned = tmp / "planned.csv"
            planned.write_text(
                "pathway,from_region,to_region,from_country,to_country,max_flow,max_counter_flow,voltage,distance,year_planned\n",
                encoding="utf-8",
            )

            n = pypsa.Network()
            for bus, country in [("US 0", "US"), ("CA 0", "CA"), ("MX 0", "MX"), ("GT 0", "GT")]:
                n.add("Bus", bus, carrier="AC", v_nom=380.0)
                n.buses.loc[bus, "country"] = country
            n.add("Line", "US-CA 1", bus0="US 0", bus1="CA 0", s_nom=100.0, s_nom_extendable=True)
            n.add("Line", "US-CA 2", bus0="US 0", bus1="CA 0", s_nom=300.0, s_nom_extendable=True)
            n.add("Line", "US-MX", bus0="US 0", bus1="MX 0", s_nom=200.0, s_nom_extendable=True)
            n.add("Line", "US-GT", bus0="US 0", bus1="GT 0", s_nom=100.0, s_nom_extendable=True)
            n.lines["s_max_pu"] = 0.7
            n.lines["carrier"] = "AC"
            n.lines["s_nom_min"] = n.lines["s_nom"]
            n.lines["s_nom_max"] = float("inf")

            cfg = {
                "global_specific": {
                    "transmission_capacity_limits": {
                        "enable": True,
                        "audit_dir": str(tmp),
                        "existing_csv": str(existing),
                        "planned_csv": str(planned),
                        "planned_policy": "by_year",
                        "missing_pair_policy": "derived_low_cap",
                        "fallback_ratio": 0.115,
                        "explicit_zero_policy": "zero_capacity",
                    }
                }
            }
            validation.apply_gtd_transmission_capacity_limits(n, 2025, cfg, s_max_pu=0.7)
            self.assertAlmostEqual(float(n.lines.at["US-CA 1", "s_nom_max"]), 25.0)
            self.assertAlmostEqual(float(n.lines.at["US-CA 2", "s_nom_max"]), 75.0)
            self.assertAlmostEqual(float(n.lines.at["US-MX", "s_nom_max"]), 0.0)
            self.assertFalse(bool(n.lines.at["US-MX", "s_nom_extendable"]))
            self.assertAlmostEqual(float(n.lines.at["US-GT", "s_nom_max"]), 0.115 * 100.0 / 0.7)
            audit = pd.read_csv(tmp / "gtd_transmission_capacity_limits_2025.csv")
            self.assertIn("missing_derived_low_cap", set(audit.policy_source))
            self.assertIn("gtd_explicit_zero", set(audit.policy_source))

    def test_gtd_capacity_year_multiplier_expands_future_caps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            existing = tmp / "existing.csv"
            existing.write_text(
                "pathway,from_region,to_region,from_country,to_country,max_flow,max_counter_flow,voltage,distance\n"
                "TRNUSACAN,USAXX,CANXX,USA,CAN,70,70,-,-\n",
                encoding="utf-8",
            )
            planned = tmp / "planned.csv"
            planned.write_text(
                "pathway,from_region,to_region,from_country,to_country,max_flow,max_counter_flow,voltage,distance,year_planned\n",
                encoding="utf-8",
            )

            n = pypsa.Network()
            n.add("Bus", "US 0", carrier="AC", v_nom=380.0)
            n.add("Bus", "CA 0", carrier="AC", v_nom=380.0)
            n.buses.loc[["US 0", "CA 0"], "country"] = ["US", "CA"]
            n.add("Line", "US-CA", bus0="US 0", bus1="CA 0", s_nom=100.0, s_nom_extendable=True)
            n.lines["s_max_pu"] = 0.7
            n.lines["s_nom_min"] = n.lines["s_nom"]
            n.lines["carrier"] = "AC"
            n.lines["s_nom_max"] = float("inf")

            cfg = {
                "global_specific": {
                    "transmission_capacity_limits": {
                        "enable": True,
                        "audit_dir": str(tmp),
                        "existing_csv": str(existing),
                        "planned_csv": str(planned),
                        "planned_policy": "by_year",
                        "capacity_multiplier_by_year": {2025: 1.0, 2030: 1.5},
                    }
                }
            }
            validation.apply_gtd_transmission_capacity_limits(n, 2030, cfg, s_max_pu=0.7)
            self.assertAlmostEqual(float(n.lines.at["US-CA", "s_nom_max"]), 150.0)
            audit = pd.read_csv(tmp / "gtd_transmission_capacity_limits_2030.csv")
            self.assertAlmostEqual(float(audit["capacity_year_multiplier"].iloc[0]), 1.5)
            self.assertAlmostEqual(float(audit["target_effective_cap_before_multiplier_mw"].iloc[0]), 70.0)
            self.assertAlmostEqual(float(audit["target_effective_cap_mw"].iloc[0]), 105.0)


if __name__ == "__main__":
    unittest.main()
