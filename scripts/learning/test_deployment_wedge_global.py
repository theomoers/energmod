import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import linopy

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
LEARNING_DIR = SCRIPTS_DIR / "learning"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

from learning.deployment_constraints import (  # noqa: E402
    GLOBAL_DEPLOYMENT_WEDGE_REGION,
    build_deployment_wedge_table_from_history,
    get_deployment_wedge_cost_granularity,
    load_deployment_wedge_table,
    summarize_realized_deployment_wedge_rows,
)
from learning.apply_learning_costs import load_config_learning, load_learning_manifest  # noqa: E402
from learning.export_postsolve_learning_costs import (  # noqa: E402
    _attach_applied_learning_costs_from_log,
)
from solve_network import add_battery_pipeline_anchor, add_learning_deployment_wedge  # noqa: E402


class GlobalDeploymentWedgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT_DIR / "config.learning.yaml"
        cls.runtime_learning_cfg = load_config_learning(cls.config_path)
        load_learning_manifest(cls.runtime_learning_cfg, cls.config_path)

    def make_learning_cfg(self, tmpdir, level="global", technologies=None):
        irena_path = Path(tmpdir) / "irena.csv"
        battery_path = Path(tmpdir) / "battery.csv"
        pd.DataFrame(
            {
                "Technology": ["PV", "PV", "Onshore"],
                "Country": ["US", "BR", "US"],
                "2020": [100.0, 20.0, 10.0],
                "2021": [150.0, 35.0, 15.0],
                "2022": [215.0, 55.0, 25.0],
            }
        ).to_csv(irena_path, index=False)
        pd.DataFrame(
            {
                "country": ["US", "BR"],
                "capa_2020": [1000.0, 500.0],
                "capa_2025": [2000.0, 1000.0],
            }
        ).to_csv(battery_path, index=False)
        return {
            "deployment_constraints": {
                "enabled": True,
                "formulation": "three_segment_wedge",
                "technologies": technologies or ["solar_power", "onwind_power", "battery_energy"],
                "wedge": {
                    "level": level,
                    "anchor_smoothing_years": 1,
                    "reference_statistic": "mean",
                    "battery_basis": "local_energy",
                    "b1_multiplier": 1.0,
                    "b2_multiplier": 2.0,
                    "phi2_pct_capex": 0.5,
                    "phi3_pct_capex": 1.0,
                    "irena_history_csv": str(irena_path),
                    "battery_history_csv": str(battery_path),
                },
            }
        }

    def make_stochastic_runtime_wedge_cfg(
        self,
        level="global",
        cost_granularity="applied_learning_cost",
        threshold_source="stochastic_allowed_block",
    ):
        learning_cfg = deepcopy(self.runtime_learning_cfg)
        deployment_cfg = learning_cfg.setdefault("deployment_constraints", {})
        deployment_cfg["enabled"] = True
        deployment_cfg["formulation"] = "three_segment_wedge"
        deployment_cfg["technologies"] = ["onwind_power"]
        deployment_cfg.setdefault("uncertainty", {})["enabled"] = True
        deployment_cfg["uncertainty"]["mode"] = "stochastic_draw"
        wedge_cfg = deployment_cfg.setdefault("wedge", {})
        wedge_cfg["level"] = level
        wedge_cfg["cost_granularity"] = cost_granularity
        wedge_cfg.setdefault("uncertainty", {})["enabled"] = True
        wedge_cfg["uncertainty"]["threshold_source"] = threshold_source
        return learning_cfg

    def test_global_wedge_table_aggregates_country_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_cfg = self.make_learning_cfg(tmpdir, level="global")
            table = build_deployment_wedge_table_from_history(
                learning_cfg,
                current_year=2022,
                technologies=["solar_power", "onwind_power"],
            )

        self.assertEqual(set(table["region"]), {GLOBAL_DEPLOYMENT_WEDGE_REGION})
        self.assertEqual(len(table), 2)
        solar = table.loc[table["technology"].eq("solar_power")].iloc[0]
        self.assertAlmostEqual(float(solar["reference_annual_addition"]), 0.085, places=9)
        self.assertAlmostEqual(float(solar["b1"]), 0.425, places=9)
        self.assertAlmostEqual(float(solar["b2"]), 0.85, places=9)
        self.assertEqual(solar["reference_method"], "flat_recent")

    def test_global_growth_projected_wedge_uses_long_irena_global_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_cfg = self.make_learning_cfg(
                tmpdir,
                level="global",
                technologies=["solar_power"],
            )
            irena_path = Path(tmpdir) / "irena_long.csv"
            records = []
            for code, multiplier in [("GLO", 1.0), ("RAF", 100.0), ("USA", 50.0)]:
                for year, value in {
                    2021: 0.0,
                    2022: 100000.0,
                    2023: 300000.0,
                    2024: 700000.0,
                    2025: 1500000.0,
                }.items():
                    records.append(
                        {
                            "Region/area (ISO 3)": code,
                            "Region/area": code,
                            "Year": year,
                            "Data Type": "Electrical Capacity",
                            "Product Name": "Solar photovoltaic",
                            "Grid Type": "OnGrid",
                            "Unit": "Megawatt",
                            " Value ": f"{value * multiplier:,.0f}",
                        }
                    )
            pd.DataFrame(records).to_csv(irena_path, index=False)
            learning_cfg["deployment_constraints"]["wedge"]["irena_history_csv"] = str(irena_path)
            learning_cfg["deployment_constraints"]["wedge"]["reference_method"] = "growth_projected"
            learning_cfg["deployment_constraints"]["wedge"]["growth_smoothing_years"] = 3

            table = build_deployment_wedge_table_from_history(
                learning_cfg,
                current_year=2030,
                technologies=["solar_power"],
            )

        self.assertEqual(len(table), 1)
        solar = table.iloc[0]
        self.assertEqual(solar["region"], GLOBAL_DEPLOYMENT_WEDGE_REGION)
        self.assertEqual(solar["reference_method"], "growth_projected")
        self.assertAlmostEqual(float(solar["reference_latest_annual_addition"]), 800.0, places=9)
        self.assertAlmostEqual(float(solar["reference_growth_rate"]), 1.0, places=9)
        self.assertAlmostEqual(float(solar["reference_block_addition"]), 49600.0, places=9)
        self.assertAlmostEqual(float(solar["b1"]), 49600.0, places=9)
        self.assertAlmostEqual(float(solar["b2"]), 99200.0, places=9)

    def test_global_growth_projected_wedge_falls_back_on_nonpositive_cagr_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_cfg = self.make_learning_cfg(
                tmpdir,
                level="global",
                technologies=["solar_power"],
            )
            irena_path = Path(tmpdir) / "irena_long.csv"
            pd.DataFrame(
                [
                    {
                        "Region/area (ISO 3)": "GLO",
                        "Region/area": "World",
                        "Year": year,
                        "Data Type": "Electrical Capacity",
                        "Product Name": "Solar photovoltaic",
                        "Grid Type": "OnGrid",
                        "Unit": "Megawatt",
                        " Value ": value,
                    }
                    for year, value in [
                        (2021, "100,000"),
                        (2022, "100,000"),
                        (2023, "200,000"),
                        (2024, "300,000"),
                        (2025, "500,000"),
                    ]
                ]
            ).to_csv(irena_path, index=False)
            learning_cfg["deployment_constraints"]["wedge"]["irena_history_csv"] = str(irena_path)
            learning_cfg["deployment_constraints"]["wedge"]["reference_method"] = "growth_projected"
            learning_cfg["deployment_constraints"]["wedge"]["growth_smoothing_years"] = 3

            table = build_deployment_wedge_table_from_history(
                learning_cfg,
                current_year=2030,
                technologies=["solar_power"],
            )

        self.assertEqual(len(table), 1)
        solar = table.iloc[0]
        self.assertEqual(solar["reference_method"], "growth_projected_fallback_flat_recent")
        self.assertAlmostEqual(float(solar["reference_annual_addition"]), 200.0, places=9)
        self.assertAlmostEqual(float(solar["reference_block_addition"]), 1000.0, places=9)
        self.assertTrue(pd.isna(solar["reference_growth_rate"]))

    def test_global_growth_projected_wedge_skips_inactive_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_cfg = self.make_learning_cfg(
                tmpdir,
                level="global",
                technologies=["solar_power"],
            )
            irena_path = Path(tmpdir) / "irena_long.csv"
            pd.DataFrame(
                [
                    {
                        "Region/area (ISO 3)": "GLO",
                        "Region/area": "World",
                        "Year": year,
                        "Data Type": "Electrical Capacity",
                        "Product Name": "Solar photovoltaic",
                        "Grid Type": "OnGrid",
                        "Unit": "Megawatt",
                        " Value ": value,
                    }
                    for year, value in [
                        (2021, "100,000"),
                        (2022, "100,000"),
                        (2023, "100,000"),
                        (2024, "100,000"),
                        (2025, "100,000"),
                    ]
                ]
            ).to_csv(irena_path, index=False)
            learning_cfg["deployment_constraints"]["wedge"]["irena_history_csv"] = str(irena_path)
            learning_cfg["deployment_constraints"]["wedge"]["reference_method"] = "growth_projected"
            learning_cfg["deployment_constraints"]["wedge"]["growth_smoothing_years"] = 3

            table = build_deployment_wedge_table_from_history(
                learning_cfg,
                current_year=2030,
                technologies=["solar_power"],
            )

        self.assertTrue(table.empty)

    def test_country_level_wedge_table_keeps_country_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_cfg = self.make_learning_cfg(tmpdir, level="country_level")
            table = build_deployment_wedge_table_from_history(
                learning_cfg,
                current_year=2022,
                technologies=["solar_power"],
            )

        self.assertEqual(set(table["region"]), {"BR", "US"})
        self.assertEqual(len(table), 2)

    def test_wedge_cost_granularity_defaults_by_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            global_cfg = self.make_learning_cfg(tmpdir, level="global")
            country_cfg = self.make_learning_cfg(tmpdir, level="country_level")

        self.assertEqual(get_deployment_wedge_cost_granularity(global_cfg), "applied_learning_cost")
        self.assertEqual(get_deployment_wedge_cost_granularity(country_cfg), "asset_exact")

    def test_invalid_wedge_cost_granularity_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_cfg = self.make_learning_cfg(tmpdir, level="global")
            learning_cfg["deployment_constraints"]["wedge"]["cost_granularity"] = "per_bus"

        with self.assertRaises(ValueError):
            get_deployment_wedge_cost_granularity(learning_cfg)

    def test_global_stochastic_thresholds_are_seed_specific_and_reproducible(self):
        learning_cfg = self.make_stochastic_runtime_wedge_cfg()

        table_seed_a = load_deployment_wedge_table(
            learning_cfg,
            current_year=2030,
            technologies=["onwind_power"],
            config_file=self.config_path,
            learning_seed="s0000",
        )
        table_seed_a_repeat = load_deployment_wedge_table(
            learning_cfg,
            current_year=2030,
            technologies=["onwind_power"],
            config_file=self.config_path,
            learning_seed="s0000",
        )
        table_seed_b = load_deployment_wedge_table(
            learning_cfg,
            current_year=2030,
            technologies=["onwind_power"],
            config_file=self.config_path,
            learning_seed="s0001",
        )

        row_a = table_seed_a.iloc[0]
        row_a_repeat = table_seed_a_repeat.iloc[0]
        row_b = table_seed_b.iloc[0]
        self.assertTrue(bool(row_a["thresholds_stochastic"]))
        self.assertEqual(row_a["threshold_source"], "stochastic_allowed_block")
        self.assertAlmostEqual(float(row_a["b1"]), float(row_a_repeat["b1"]), places=9)
        self.assertAlmostEqual(float(row_a["b2"]), float(row_a_repeat["b2"]), places=9)
        self.assertAlmostEqual(
            float(row_a["allowed_block_addition"]),
            float(row_a["threshold_block_addition"]),
            places=9,
        )
        self.assertNotAlmostEqual(float(row_a["b1"]), float(row_b["b1"]), places=9)
        self.assertNotAlmostEqual(float(row_a["b2"]), float(row_b["b2"]), places=9)

    def test_global_stochastic_growth_projected_thresholds_preserve_old_scale(self):
        deterministic_cfg = self.make_stochastic_runtime_wedge_cfg(
            threshold_source="stochastic_growth_projected"
        )
        deterministic_cfg["deployment_constraints"]["wedge"]["uncertainty"]["enabled"] = False
        stochastic_cfg = self.make_stochastic_runtime_wedge_cfg(
            threshold_source="stochastic_growth_projected"
        )

        deterministic = load_deployment_wedge_table(
            deterministic_cfg,
            current_year=2030,
            technologies=["onwind_power"],
            config_file=self.config_path,
            learning_seed="s0000",
        ).iloc[0]
        stochastic_a = load_deployment_wedge_table(
            stochastic_cfg,
            current_year=2030,
            technologies=["onwind_power"],
            config_file=self.config_path,
            learning_seed="s0000",
        ).iloc[0]
        stochastic_b = load_deployment_wedge_table(
            stochastic_cfg,
            current_year=2030,
            technologies=["onwind_power"],
            config_file=self.config_path,
            learning_seed="s0001",
        ).iloc[0]

        self.assertEqual(stochastic_a["threshold_source"], "stochastic_growth_projected")
        self.assertTrue(bool(stochastic_a["thresholds_stochastic"]))
        self.assertNotAlmostEqual(float(stochastic_a["b1"]), float(stochastic_b["b1"]), places=9)
        self.assertLess(abs(float(stochastic_a["b1"]) - float(deterministic["b1"])), float(deterministic["b1"]))

    def test_global_stochastic_thresholds_leave_deterministic_mode_unchanged(self):
        learning_cfg = self.make_stochastic_runtime_wedge_cfg()
        learning_cfg["deployment_constraints"]["wedge"]["uncertainty"]["enabled"] = False

        table_seed_a = load_deployment_wedge_table(
            learning_cfg,
            current_year=2030,
            technologies=["onwind_power"],
            config_file=self.config_path,
            learning_seed="s0000",
        )
        table_seed_b = load_deployment_wedge_table(
            learning_cfg,
            current_year=2030,
            technologies=["onwind_power"],
            config_file=self.config_path,
            learning_seed="s0001",
        )

        row_a = table_seed_a.iloc[0]
        row_b = table_seed_b.iloc[0]
        self.assertFalse(bool(row_a["thresholds_stochastic"]))
        self.assertAlmostEqual(float(row_a["b1"]), float(row_b["b1"]), places=9)
        self.assertAlmostEqual(float(row_a["b2"]), float(row_b["b2"]), places=9)

    def test_stochastic_thresholds_reject_unsupported_wedge_modes(self):
        asset_exact_cfg = self.make_stochastic_runtime_wedge_cfg(cost_granularity="asset_exact")
        country_level_cfg = self.make_stochastic_runtime_wedge_cfg(level="country_level")

        with self.assertRaises(ValueError):
            load_deployment_wedge_table(
                asset_exact_cfg,
                current_year=2030,
                technologies=["onwind_power"],
                config_file=self.config_path,
                learning_seed="s0000",
            )
        with self.assertRaises(ValueError):
            load_deployment_wedge_table(
                country_level_cfg,
                current_year=2030,
                technologies=["onwind_power"],
                config_file=self.config_path,
                learning_seed="s0000",
            )

    def test_global_wedge_diagnostics_use_global_realized_build(self):
        buses = pd.DataFrame(
            {"country": ["US", "BR"]},
            index=["US 0", "BR 0"],
        )
        generators = pd.DataFrame(
            {
                "carrier": ["solar", "solar"],
                "bus": ["US 0", "BR 0"],
                "p_nom_extendable": [True, True],
                "build_year": [2022, 2022],
                "p_nom": [100.0, 200.0],
                "p_nom_opt": [100.0, 200.0],
                "capital_cost": [10.0, 20.0],
            },
            index=["solar-us", "solar-br"],
        )
        network = SimpleNamespace(
            buses=buses,
            generators=generators,
            stores=pd.DataFrame(),
            meta={
                "learning_costs": {
                    "solar_power": {
                        "capital_cost": 12.0,
                        "c_overnight": 100.0,
                    }
                }
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            learning_cfg = self.make_learning_cfg(
                tmpdir,
                level="global",
                technologies=["solar_power"],
            )
            rows = summarize_realized_deployment_wedge_rows(
                network,
                current_year=2022,
                learning_cfg=learning_cfg,
                technologies=["solar_power"],
            )

        self.assertEqual(len(rows), 1)
        row = rows.iloc[0]
        self.assertEqual(row["country"], GLOBAL_DEPLOYMENT_WEDGE_REGION)
        self.assertAlmostEqual(float(row["realized_block_addition_constrained_basis"]), 0.3, places=9)
        self.assertAlmostEqual(float(row["realized_seg1"]), 0.3, places=9)
        self.assertAlmostEqual(float(row["realized_seg2"]), 0.0, places=9)
        self.assertAlmostEqual(float(row["realized_seg3"]), 0.0, places=9)

    def test_global_wedge_diagnostics_require_applied_learning_cost_metadata(self):
        buses = pd.DataFrame(
            {"country": ["US", "BR"]},
            index=["US 0", "BR 0"],
        )
        generators = pd.DataFrame(
            {
                "carrier": ["solar", "solar"],
                "bus": ["US 0", "BR 0"],
                "p_nom_extendable": [True, True],
                "build_year": [2022, 2022],
                "p_nom": [500.0, 500.0],
                "p_nom_opt": [500.0, 500.0],
                "capital_cost": [10.0, 20.0],
            },
            index=["solar-us", "solar-br"],
        )
        network = SimpleNamespace(buses=buses, generators=generators, stores=pd.DataFrame(), meta={})

        with tempfile.TemporaryDirectory() as tmpdir:
            learning_cfg = self.make_learning_cfg(
                tmpdir,
                level="global",
                technologies=["solar_power"],
            )
            with self.assertRaises(ValueError):
                summarize_realized_deployment_wedge_rows(
                    network,
                    current_year=2022,
                    learning_cfg=learning_cfg,
                    technologies=["solar_power"],
                )

    def test_global_wedge_diagnostics_use_applied_learning_cost(self):
        buses = pd.DataFrame(
            {"country": ["US", "BR"]},
            index=["US 0", "BR 0"],
        )
        generators = pd.DataFrame(
            {
                "carrier": ["solar", "solar"],
                "bus": ["US 0", "BR 0"],
                "p_nom_extendable": [True, True],
                "build_year": [2022, 2022],
                "p_nom": [500.0, 500.0],
                "p_nom_opt": [500.0, 500.0],
                "capital_cost": [10.0, 20.0],
            },
            index=["solar-us", "solar-br"],
        )
        network = SimpleNamespace(
            buses=buses,
            generators=generators,
            stores=pd.DataFrame(),
            meta={
                "learning_costs": {
                    "solar_power": {
                        "capital_cost": 42.0,
                        "c_overnight": 700.0,
                    }
                }
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            learning_cfg = self.make_learning_cfg(
                tmpdir,
                level="global",
                technologies=["solar_power"],
            )
            learning_cfg["deployment_constraints"]["wedge"]["cost_granularity"] = "applied_learning_cost"
            rows = summarize_realized_deployment_wedge_rows(
                network,
                current_year=2022,
                learning_cfg=learning_cfg,
                technologies=["solar_power"],
            )
            network.meta["learning_costs"]["solar_power"]["capital_cost"] = 84.0
            rows_updated = summarize_realized_deployment_wedge_rows(
                network,
                current_year=2022,
                learning_cfg=learning_cfg,
                technologies=["solar_power"],
            )

        row = rows.iloc[0]
        self.assertEqual(row["country"], GLOBAL_DEPLOYMENT_WEDGE_REGION)
        self.assertEqual(row["cost_granularity"], "applied_learning_cost")
        self.assertEqual(row["basis_cost_source"], "network_meta_learning_costs")
        self.assertAlmostEqual(float(row["learning_capital_cost"]), 42.0, places=9)
        self.assertAlmostEqual(float(row["learning_c_overnight"]), 700.0, places=9)
        self.assertAlmostEqual(float(row["basis_cost_eur_per_unit_mean"]), 15000.0, places=9)
        self.assertAlmostEqual(float(row["basis_cost_eur_per_unit_representative"]), 42000.0, places=9)
        self.assertAlmostEqual(float(row["realized_seg1"]), 0.425, places=9)
        self.assertAlmostEqual(float(row["realized_seg2"]), 0.425, places=9)
        self.assertAlmostEqual(float(row["realized_seg3"]), 0.15, places=9)
        self.assertAlmostEqual(float(row["realized_wedge_cost_eur"]), 15225.0, places=9)
        self.assertAlmostEqual(float(rows_updated.iloc[0]["realized_wedge_cost_eur"]), 30450.0, places=9)

    def test_global_applied_learning_cost_solver_uses_raw_epigraph(self):
        buses = pd.DataFrame(
            {"country": ["US", "BR"]},
            index=["US 0", "BR 0"],
        )
        generators = pd.DataFrame(
            {
                "carrier": ["solar", "solar"],
                "bus": ["US 0", "BR 0"],
                "p_nom_extendable": [True, True],
                "build_year": [2022, 2022],
                "p_nom": [0.0, 0.0],
                "capital_cost": [10.0, 20.0],
            },
            index=["solar-us", "solar-br"],
        )
        network = SimpleNamespace(
            buses=buses,
            generators=generators,
            stores=pd.DataFrame(),
            meta={
                "learning_costs": {
                    "solar_power": {
                        "capital_cost": 42.0,
                        "c_overnight": 700.0,
                    }
                }
            },
        )
        network.model = linopy.Model()
        assets = pd.Index(["solar-us", "solar-br"], name="Generator")
        p_nom = network.model.add_variables(lower=0.0, coords=[assets], name="Generator-p_nom")
        network.model.add_objective(0 * p_nom.sum())

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "learning": self.make_learning_cfg(
                    tmpdir,
                    level="global",
                    technologies=["solar_power"],
                )
            }
            config["learning"]["deployment_constraints"]["wedge"][
                "cost_granularity"
            ] = "applied_learning_cost"
            add_learning_deployment_wedge(network, planning_year=2022, config=config)

        variable_names = set(network.model.variables)
        constraint_names = set(network.model.constraints)
        self.assertIn("learning_deployment_excess1_raw__solar_power__2022", variable_names)
        self.assertIn("learning_deployment_excess2_raw__solar_power__2022", variable_names)
        self.assertNotIn("learning_deployment_seg1__solar_power__2022", variable_names)
        self.assertNotIn(
            "learning_deployment_wedge_technology_balance__solar_power__2022__GLOBAL",
            constraint_names,
        )
        self.assertIn(
            "learning_deployment_wedge_excess1_raw__solar_power__2022__GLOBAL",
            constraint_names,
        )
        row = network.meta["learning_deployment_wedge"][0]
        self.assertEqual(row["wedge_solver_formulation"], "epigraph_raw_units")
        self.assertAlmostEqual(float(row["learning_capital_cost"]), 42.0, places=9)
        self.assertAlmostEqual(float(row["basis_cost_eur_per_unit_representative"]), 42000.0, places=9)
        self.assertAlmostEqual(float(row["b1_raw"]), 425.0, places=9)
        self.assertAlmostEqual(float(row["b2_raw"]), 850.0, places=9)

    def test_battery_pipeline_anchor_uses_global_energy_and_power_only_in_2030(self):
        stores = pd.DataFrame(
            {
                "carrier": ["battery", "battery", "battery"],
                "bus": ["US 0", "BR 0", "US 0"],
                "e_nom_extendable": [False, True, True],
                "build_year": [2025, 2030, 2035],
                "e_nom": [500000.0, 0.0, 0.0],
            },
            index=["battery-2025", "battery-2030", "battery-2035"],
        )
        links = pd.DataFrame(
            {
                "carrier": ["battery charger", "battery discharger"],
                "p_nom_extendable": [True, True],
                "build_year": [2030, 2030],
                "p_nom": [0.0, 0.0],
            },
            index=["charger-2030", "discharger-2030"],
        )
        network = SimpleNamespace(stores=stores, links=links, meta={})
        network.model = linopy.Model()
        store_assets = pd.Index(["battery-2030", "battery-2035"], name="Store")
        store_nom = network.model.add_variables(lower=0.0, coords=[store_assets], name="Store-e_nom")
        link_assets = pd.Index(["charger-2030", "discharger-2030"], name="Link")
        link_nom = network.model.add_variables(lower=0.0, coords=[link_assets], name="Link-p_nom")
        network.model.add_objective(0 * store_nom.sum() + 0 * link_nom.sum())

        config = {
            "learning": {
                "battery_pipeline_anchor": {
                    "enabled": True,
                    "anchor_year": 2030,
                    "global_energy_gwh": 1900.0,
                    "global_power_gw": 760.0,
                    "applies_to": "battery_energy",
                    "scope": "global",
                    "active_only_until_anchor_year": True,
                }
            }
        }
        add_battery_pipeline_anchor(network, planning_year=2030, config=config)

        self.assertNotIn("battery_pipeline_anchor_shortfall__2030", set(network.model.variables))
        self.assertIn(
            "battery_pipeline_anchor_global_energy_stock__2030",
            set(network.model.constraints),
        )
        self.assertIn(
            "battery_pipeline_anchor_global_charger_power__2030",
            set(network.model.constraints),
        )
        self.assertIn(
            "battery_pipeline_anchor_global_discharger_power__2030",
            set(network.model.constraints),
        )
        self.assertNotIn("battery_pipeline_anchor_shortfall__2035", set(network.model.variables))
        anchor_meta = network.meta["battery_pipeline_anchor"]
        self.assertAlmostEqual(anchor_meta["fixed_battery_store_stock_gwh"], 500.0, places=9)
        self.assertEqual(anchor_meta["variable_battery_store_asset_count"], 1)
        self.assertAlmostEqual(anchor_meta["fixed_battery_charger_power_gw"], 0.0, places=9)
        self.assertAlmostEqual(anchor_meta["fixed_battery_discharger_power_gw"], 0.0, places=9)
        self.assertEqual(anchor_meta["variable_battery_charger_asset_count"], 1)
        self.assertEqual(anchor_meta["variable_battery_discharger_asset_count"], 1)
        self.assertAlmostEqual(anchor_meta["target_charger_power_gw"], 760.0, places=9)
        self.assertAlmostEqual(anchor_meta["target_discharger_power_gw"], 760.0, places=9)
        self.assertAlmostEqual(anchor_meta["reference_duration_hours"], 2.5, places=9)

        later_network = SimpleNamespace(stores=stores, links=links, meta={})
        later_network.model = linopy.Model()
        later_store_nom = later_network.model.add_variables(
            lower=0.0,
            coords=[store_assets],
            name="Store-e_nom",
        )
        later_network.model.add_objective(0 * later_store_nom.sum())
        add_battery_pipeline_anchor(later_network, planning_year=2035, config=config)
        self.assertNotIn("battery_pipeline_anchor_shortfall__2030", set(later_network.model.variables))
        self.assertNotIn("battery_pipeline_anchor", later_network.meta)

    def test_global_wedge_skips_when_b1_exceeds_finite_asset_upper_bound(self):
        buses = pd.DataFrame(
            {"country": ["US", "BR"]},
            index=["US 0", "BR 0"],
        )
        generators = pd.DataFrame(
            {
                "carrier": ["solar", "solar"],
                "bus": ["US 0", "BR 0"],
                "p_nom_extendable": [True, True],
                "build_year": [2022, 2022],
                "p_nom": [0.0, 0.0],
                "p_nom_max": [100.0, 200.0],
                "capital_cost": [10.0, 20.0],
            },
            index=["solar-us", "solar-br"],
        )
        network = SimpleNamespace(
            buses=buses,
            generators=generators,
            stores=pd.DataFrame(),
            meta={
                "learning_costs": {
                    "solar_power": {
                        "capital_cost": 42.0,
                        "c_overnight": 700.0,
                    }
                }
            },
        )
        network.model = linopy.Model()
        assets = pd.Index(["solar-us", "solar-br"], name="Generator")
        p_nom = network.model.add_variables(lower=0.0, coords=[assets], name="Generator-p_nom")
        network.model.add_objective(0 * p_nom.sum())

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "learning": self.make_learning_cfg(
                    tmpdir,
                    level="global",
                    technologies=["solar_power"],
                )
            }
            config["learning"]["deployment_constraints"]["wedge"][
                "cost_granularity"
            ] = "applied_learning_cost"
            add_learning_deployment_wedge(network, planning_year=2022, config=config)

        variable_names = set(network.model.variables)
        constraint_names = set(network.model.constraints)
        self.assertNotIn("learning_deployment_excess1_raw__solar_power__2022", variable_names)
        self.assertNotIn("learning_deployment_excess2_raw__solar_power__2022", variable_names)
        self.assertNotIn(
            "learning_deployment_wedge_excess1_raw__solar_power__2022__GLOBAL",
            constraint_names,
        )
        self.assertNotIn("learning_deployment_wedge", network.meta)

    def test_global_asset_exact_granularity_allows_missing_learning_cost_metadata(self):
        buses = pd.DataFrame(
            {"country": ["US", "BR"]},
            index=["US 0", "BR 0"],
        )
        generators = pd.DataFrame(
            {
                "carrier": ["solar", "solar"],
                "bus": ["US 0", "BR 0"],
                "p_nom_extendable": [True, True],
                "build_year": [2022, 2022],
                "p_nom": [500.0, 500.0],
                "p_nom_opt": [500.0, 500.0],
                "capital_cost": [10.0, 20.0],
            },
            index=["solar-us", "solar-br"],
        )
        network = SimpleNamespace(buses=buses, generators=generators, stores=pd.DataFrame(), meta={})

        with tempfile.TemporaryDirectory() as tmpdir:
            learning_cfg = self.make_learning_cfg(
                tmpdir,
                level="global",
                technologies=["solar_power"],
            )
            learning_cfg["deployment_constraints"]["wedge"]["cost_granularity"] = "asset_exact"
            rows = summarize_realized_deployment_wedge_rows(
                network,
                current_year=2022,
                learning_cfg=learning_cfg,
                technologies=["solar_power"],
            )

        row = rows.iloc[0]
        self.assertEqual(row["cost_granularity"], "asset_exact")
        self.assertEqual(row["basis_cost_source"], "asset_costs")
        self.assertAlmostEqual(float(row["basis_cost_eur_per_unit_representative"]), 15000.0, places=9)

    def test_exporter_attaches_applied_learning_costs_from_base_log(self):
        network = SimpleNamespace(meta={})
        base_cost_log = pd.DataFrame(
            {
                "technology": ["solar_power", "battery_energy"],
                "capital_cost": [42.0, 12.0],
                "c_overnight": [700.0, 220.0],
                "unit": ["kW", "kWh"],
                "planning_horizon": [2025, 2025],
            }
        )

        _attach_applied_learning_costs_from_log(network, base_cost_log)

        self.assertEqual(network.meta["learning_costs"]["solar_power"]["capital_cost"], 42.0)
        self.assertEqual(network.meta["learning_costs"]["battery_energy"]["c_overnight"], 220.0)


if __name__ == "__main__":
    unittest.main()
