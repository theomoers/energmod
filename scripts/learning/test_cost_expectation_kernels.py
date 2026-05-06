import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
MODULE_PATH = ROOT / "scripts" / "learning" / "apply_learning_costs.py"
SPEC = importlib.util.spec_from_file_location("apply_learning_costs_module", MODULE_PATH)
alc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alc)
EXPORT_MODULE_PATH = ROOT / "scripts" / "learning" / "export_postsolve_learning_costs.py"
EXPORT_SPEC = importlib.util.spec_from_file_location("export_postsolve_learning_costs_module", EXPORT_MODULE_PATH)
ep = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(ep)


def _make_state(committed_year, level_by_tech):
    return {
        "last_applied_year": int(committed_year),
        "technology_states": {
            tech: {"last_log_capex": math.log(float(level) * 1000.0)}
            for tech, level in level_by_tech.items()
        },
        "annual_cost_history": {},
    }


def test_build_lagged_kernel_years_and_split():
    assert alc.build_lagged_kernel_years(2030, 3) == [2023, 2024, 2025, 2026, 2027]
    assert alc.build_lagged_kernel_years(2030, 4) == [2022, 2023, 2024, 2025, 2026]
    assert alc.build_lagged_kernel_years(2030, 2) == [2024, 2025, 2026, 2027, 2028]

    known, expected = alc.split_kernel_years([2023, 2024, 2025, 2026, 2027], 2025)
    assert known == [2023, 2024, 2025]
    assert expected == [2026, 2027]


def test_ssbr_runtime_has_no_autonomous_drift_when_experience_and_shocks_are_zero():
    artifacts = {
        "battery_energy": {
            "parameter_summary": {
                "b_slow": 0.25,
                "b_fast": 0.40,
                "sigma_slow_mean": 0.0,
                "sigma_fast_mean": 0.0,
            },
            "uncertainty_terms": {
                "b_slow_draws": [0.25],
                "b_fast_draws": [0.40],
                "sigma_slow_draws": [0.0],
                "sigma_fast_draws": [0.0],
                "p_slow_slow_draws": [1.0],
                "p_fast_fast_draws": [1.0],
            },
        }
    }
    state = {
        "technology_states": {"battery_energy": {"last_log_capex": math.log(100.0)}},
        "shared_regime_state": {"initial_regime_probs": [1.0, 0.0], "current_regime": 0},
    }
    result = alc.simulate_shared_state_runtime(
        artifacts=artifacts,
        state=state,
        block_dlog_experience_by_tech={"battery_energy": 0.0},
        elapsed_years=5,
        rng=np.random.default_rng(123),
        sample_mode="single_draw",
        n_samples=1,
    )
    assert result["final_log_costs"]["battery_energy"] == pytest.approx(math.log(100.0))


def test_historical_bootstrap_kernel_uses_lagged_years_and_weights(monkeypatch):
    loaded = []

    def fake_load_cost(tech, year, learning_cfg):
        loaded.append((tech, int(year)))
        return float(year)

    monkeypatch.setattr(alc, "load_cost_from_historical_csv", fake_load_cost)

    years = alc.build_lagged_kernel_years(2025, lag_year=2, kernel_length=5)
    value, costs = alc.load_historical_kernel_cost(
        "solar_power",
        years,
        [1, 1, 1, 1, 1],
        {},
    )

    assert years == [2019, 2020, 2021, 2022, 2023]
    assert loaded == [("solar_power", year) for year in years]
    assert costs == pytest.approx([2019.0, 2020.0, 2021.0, 2022.0, 2023.0])
    assert value == pytest.approx(2021.0)


def test_fossil_2025_bootstrap_average_uses_available_historical_years():
    params_row = {"fuel_type": "gas", "market": "test_market"}
    historical = {
        ("gas", "test_market", year): float(year)
        for year in range(2021, 2026)
    }

    value, weights_json = alc._compute_fossil_bootstrap_historical_average_price(
        params_row=params_row,
        current_year=2025,
        historical_by_key=historical,
        expectation_weights=np.ones(5),
    )

    assert value == pytest.approx(2023.0)
    assert json.loads(weights_json) == pytest.approx([0.2, 0.2, 0.2, 0.2, 0.2])

    historical.pop(("gas", "test_market", 2025))
    value, weights_json = alc._compute_fossil_bootstrap_historical_average_price(
        params_row=params_row,
        current_year=2025,
        historical_by_key=historical,
        expectation_weights=np.ones(5),
    )

    assert value == pytest.approx(2022.5)
    assert json.loads(weights_json) == pytest.approx([0.25, 0.25, 0.25, 0.25, 0.0])


def test_fossil_2025_bootstrap_payload_is_deterministic_from_history():
    params_df = pd.DataFrame(
        [
            {
                "fuel_type": "gas",
                "market": "test_market",
                "phi": 0.0,
                "kappa": 0.0,
                "sigma_epsilon": 100.0,
                "last_observed_year": 2024,
                "last_observed_price_eur_mwh": 40.0,
                "last_observed_log_price": math.log(40.0),
            }
        ]
    )
    country_map_df = pd.DataFrame(
        [{"fuel_type": "gas", "market": "test_market", "country": "AA"}]
    )
    historical_by_key = {
        ("gas", "test_market", 2021): 10.0,
        ("gas", "test_market", 2022): 20.0,
        ("gas", "test_market", 2023): 30.0,
        ("gas", "test_market", 2024): 40.0,
    }
    learning_cfg = {
        "seed": 123,
        "fossil_price_uncertainty": {
            "enabled": True,
            "fuels": ["gas"],
            "expectation_mode": "block_average_expected",
            "annual_weights": [1, 1, 1, 1, 1],
        },
        "_fossil_price_bundle": {
            "params_df": params_df,
            "country_map_df": country_map_df,
            "params_by_key": {
                ("gas", "test_market"): params_df.iloc[0].to_dict(),
            },
            "historical_by_key": historical_by_key,
            "manifest_path": "",
            "schema_version": "test",
            "manifest_sha256": "",
        },
    }

    payload = alc._build_fossil_price_payload(
        learning_cfg,
        runtime_metadata={},
        current_year=2025,
        state_payload={
            "fossil_price_states": {
                "gas": {
                    "test_market": {
                        "realized_log_price": math.log(1.0),
                        "realized_price_eur_mwh": 1.0,
                        "last_state_year": 2020,
                        "source_state_year": 2020,
                    }
                }
            }
        },
    )

    row = payload["log_df"].iloc[0]
    assert row["applied_price_eur_mwh"] == pytest.approx(25.0)
    assert row["price_terminal_point_eur_mwh"] == pytest.approx(40.0)
    assert row["last_state_year"] == 2024
    assert row["source_state_year"] == 2024
    assert json.loads(row["fossil_price_expectation_weights_json"]) == pytest.approx(
        [0.25, 0.25, 0.25, 0.25, 0.0]
    )


def test_fossil_2020_bootstrap_payload_uses_2020_historical_price():
    params_df = pd.DataFrame(
        [
            {
                "fuel_type": "oil",
                "market": "test_market",
                "phi": 0.0,
                "kappa": 0.0,
                "sigma_epsilon": 100.0,
                "last_observed_year": 2020,
                "last_observed_price_eur_mwh": 42.0,
                "last_observed_log_price": math.log(42.0),
            }
        ]
    )
    country_map_df = pd.DataFrame(
        [{"fuel_type": "oil", "market": "test_market", "country": "AA"}]
    )
    historical_by_key = {
        ("oil", "test_market", 2018): 10.0,
        ("oil", "test_market", 2019): 20.0,
        ("oil", "test_market", 2020): 42.0,
    }
    learning_cfg = {
        "seed": 123,
        "fossil_price_uncertainty": {
            "enabled": True,
            "fuels": ["oil"],
            "expectation_mode": "block_average_expected",
            "annual_weights": [1, 1, 1, 1, 1],
        },
        "_fossil_price_bundle": {
            "params_df": params_df,
            "country_map_df": country_map_df,
            "params_by_key": {
                ("oil", "test_market"): params_df.iloc[0].to_dict(),
            },
            "historical_by_key": historical_by_key,
            "manifest_path": "",
            "schema_version": "test",
            "manifest_sha256": "",
        },
    }

    payload = alc._build_fossil_price_payload(
        learning_cfg,
        runtime_metadata={},
        current_year=2020,
        state_payload={},
    )

    row = payload["log_df"].iloc[0]
    assert row["applied_price_eur_mwh"] == pytest.approx(42.0)
    assert row["price_terminal_point_eur_mwh"] == pytest.approx(42.0)
    assert row["last_state_year"] == 2020
    assert row["source_state_year"] == 2020
    assert json.loads(row["fossil_price_expectation_weights_json"]) == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 1.0]
    )


def test_get_known_annual_cost_prefers_runtime_state_and_history(monkeypatch):
    state = _make_state(
        2030,
        {"solar_power": 30.0, "onwind_power": 1.0, "battery_energy": 1.0},
    )
    state["annual_cost_history"] = {"solar_power": {"2028": 12.0, "2029": 18.0}}

    monkeypatch.setattr(alc, "load_cost_from_historical_csv", lambda tech, year, cfg: float(year))

    assert alc.get_known_annual_cost_for_year("solar_power", 2030, state, {}) == pytest.approx(30.0)
    assert alc.get_known_annual_cost_for_year("solar_power", 2028, state, {}) == pytest.approx(12.0)
    assert alc.get_known_annual_cost_for_year("solar_power", 2024, state, {}) == pytest.approx(2024.0)


def test_battery_bos_adder_uses_confirmed_grid_storage_anchor(tmp_path, monkeypatch):
    anchor = tmp_path / "battery_grid_storage_cost_anchor.csv"
    anchor.write_text(
        "year,cost_usd2025_per_kwh,cost_eur2020_per_kwh,source\n"
        "2025,125.0,88.07002359846918,test_anchor\n",
        encoding="utf-8",
    )
    learning_cfg = {
        "_manifest_root": tmp_path,
        "_manifest": {
            "battery_treatment": {
                "grid_storage_cost_anchor": {
                    "series_csv": "battery_grid_storage_cost_anchor.csv",
                },
            },
        },
    }

    monkeypatch.setattr(
        alc,
        "load_cost_from_historical_csv",
        lambda tech, year, cfg: 76.2053294936,
    )

    adder = alc.get_battery_energy_bos_adder(learning_cfg, "costs_2030.csv")

    assert adder == pytest.approx(88.07002359846918 - 76.2053294936)
    assert learning_cfg["_battery_energy_bos_cache"]["confirmed_grid_storage_anchor_adder"] == pytest.approx(adder)


def test_battery_phi_future_years_hold_last_observed_value(tmp_path):
    mapping_csv = tmp_path / "battery_phi_mapping.csv"
    mapping_csv.write_text(
        "year,rolling_median_5yr_phi\n"
        "2022,17.75\n"
        "2023,13.97318861400647\n"
        "2024,11.0\n",
        encoding="utf-8",
    )
    learning_cfg = {
        "_manifest_root": tmp_path,
        "_manifest": {
            "battery_treatment": {
                "global_liion_mapping": {
                    "series_csv": "battery_phi_mapping.csv",
                    "series_column": "rolling_median_5yr_phi",
                    "fallback_value": 16.97314803391489,
                    "block_aggregation": "arithmetic_mean",
                },
            },
        },
    }

    assert alc.get_battery_phi_for_year(learning_cfg, 2024) == pytest.approx(11.0)
    assert alc.get_battery_phi_for_year(learning_cfg, 2030) == pytest.approx(11.0)
    assert alc.get_battery_phi_for_block(learning_cfg, 2025, 2030) == pytest.approx(11.0)


def test_committed_battery_state_records_experience_increment(monkeypatch):
    monkeypatch.setattr(ep, "get_battery_phi_for_block", lambda learning_cfg, prev_year, current_year: 11.0)
    payload = {
        "technology_states": {"battery_energy": {}},
        "capacity_history": {"battery_energy": {"2025": 1000.0}},
        "modeled_capacity_history": {"battery_energy": {"2025": 50.0}},
    }

    updated = ep._update_committed_capacity_histories(
        payload,
        solved_capacity_by_tech={"battery_energy": 60.0},
        learning_cfg={},
        current_year=2030,
    )

    assert updated["modeled_capacity_history"]["battery_energy"]["2030"] == pytest.approx(60.0)
    assert updated["experience_increment_history"]["battery_energy"]["2030"] == pytest.approx(110.0)
    assert updated["capacity_history"]["battery_energy"]["2030"] == pytest.approx(1110.0)
    details = updated["experience_increment_details"]["battery_energy"]["2030"]
    assert details["modeled_block_addition"] == pytest.approx(10.0)
    assert details["phi"] == pytest.approx(11.0)
    assert details["experience_increment"] == pytest.approx(110.0)


def test_block_average_expected_costs_use_kernel_split_and_level_weights(monkeypatch):
    learning_cfg = {
        "seed": 0,
        "cost_expectations": {
            "mode": "block_average_expected",
            "kernel_mode": "technology_specific_lagged_window",
            "annual_weights": [0.1, 0.2, 0.3, 0.2, 0.2],
            "lag_years_by_tech": {
                "solar_power": 3,
                "onwind_power": 4,
                "battery_energy": 2,
            },
        },
    }
    state = _make_state(
        2025,
        {"solar_power": 30.0, "onwind_power": 40.0, "battery_energy": 25.0},
    )
    artifacts = {
        "solar_power": {},
        "onwind_power": {},
        "battery_energy": {},
    }

    monkeypatch.setattr(alc, "convert_to_capital_cost", lambda value, *args, **kwargs: float(value))
    monkeypatch.setattr(alc, "get_battery_energy_bos_adder", lambda learning_cfg, costs_file: 2.0)
    monkeypatch.setattr(alc, "load_battery_power_investment_cost", lambda costs_file: 100.0)
    expected_levels = np.log(
        np.array([[40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]], dtype=float) * 1000.0
    )
    monkeypatch.setattr(
        alc,
        "_simulate_frozen_block_expectation_annual_paths",
        lambda artifacts, state, selected_model, elapsed_years, rng: {
            tech: expected_levels[:, :elapsed_years] for tech in artifacts
        },
    )

    known_map = {
        "solar_power": {2023: 10.0, 2024: 20.0, 2025: 30.0},
        "onwind_power": {2022: 11.0, 2023: 21.0, 2024: 31.0, 2025: 41.0},
        "battery_energy": {2024: 15.0, 2025: 25.0},
    }
    monkeypatch.setattr(
        alc,
        "get_known_annual_cost_for_year",
        lambda tech, year, state, cfg: known_map[tech][int(year)],
    )

    diagnostics = alc._compute_block_average_expected_costs(
        artifacts=artifacts,
        state=state,
        current_year=2030,
        selected_model="shared_state_bayesian_regime_wright",
        learning_cfg=learning_cfg,
        costs_file="costs.csv",
    )

    assert diagnostics["solar_power"]["c_overnight"] == pytest.approx(32.0)
    assert diagnostics["solar_power"]["c_overnight_terminal_point"] == pytest.approx(80.0)
    assert json.loads(diagnostics["solar_power"]["known_kernel_years_json"]) == [2023, 2024, 2025]
    assert json.loads(diagnostics["solar_power"]["expected_kernel_years_json"]) == [2026, 2027]

    assert diagnostics["onwind_power"]["c_overnight"] == pytest.approx(30.8)
    assert diagnostics["battery_energy"]["raw_c_overnight"] == pytest.approx(40.5)
    assert diagnostics["battery_energy"]["c_overnight"] == pytest.approx(42.5)
    assert diagnostics["battery_energy"]["source_cost_basis"] == "liion_pack"
    assert diagnostics["battery_energy"]["network_cost_basis"] == "grid_storage_energy"
    assert diagnostics["battery_energy"]["cost_basis_conversion_method"] == "additive_pack_plus_fixed_energy_bos"
    assert diagnostics["battery_energy"]["cost_basis_conversion_multiplier"] == pytest.approx(1.0)
    assert diagnostics["battery_energy"]["battery_energy_bos_adder_eur_per_kwh"] == pytest.approx(2.0)
    assert diagnostics["battery_energy"]["reported_4h_bess_capex_eur_per_kwh"] == pytest.approx(42.5 + 25.0)
    assert json.loads(diagnostics["battery_energy"]["raw_kernel_costs_json"]) == pytest.approx([
        15.0,
        25.0,
        40.0,
        50.0,
        60.0,
    ])
    assert json.loads(diagnostics["battery_energy"]["applied_kernel_costs_json"]) == pytest.approx([
        17.0,
        27.0,
        42.0,
        52.0,
        62.0,
    ])


def test_stochastic_point_cost_converts_battery_basis_without_mutating_raw_log(monkeypatch):
    learning_cfg = {}
    runtime_metadata = {
        "engine": "stochastic_forecast",
        "selected_model": "shared_state_bayesian_regime_wright",
        "cost_expectation_mode": "point_cost",
    }
    state = _make_state(
        2025,
        {"solar_power": 30.0, "battery_energy": 25.0},
    )
    artifacts = {
        "solar_power": {},
        "battery_energy": {},
    }
    monkeypatch.setattr(alc, "convert_to_capital_cost", lambda value, *args, **kwargs: float(value))
    monkeypatch.setattr(alc, "get_battery_energy_bos_adder", lambda learning_cfg, costs_file: 2.5)
    monkeypatch.setattr(alc, "load_battery_power_investment_cost", lambda costs_file: 80.0)

    costs = alc._learning_costs_from_stochastic_state(
        artifacts=artifacts,
        state=state,
        current_year=2030,
        selected_model="shared_state_bayesian_regime_wright",
        cumulative_capacity_map={"solar_power": 1.0, "battery_energy": 1.0},
        learning_cfg=learning_cfg,
        costs_file="costs.csv",
        wacc_dict=None,
        runtime_metadata=runtime_metadata,
    )

    battery = costs["battery_energy"]
    assert battery["raw_c_overnight"] == pytest.approx(25.0)
    assert battery["c_overnight"] == pytest.approx(27.5)
    assert battery["capital_cost"] == pytest.approx(27.5)
    assert battery["battery_pack_learning_cost_eur_per_kwh"] == pytest.approx(25.0)
    assert battery["battery_energy_bos_adder_eur_per_kwh"] == pytest.approx(2.5)
    assert battery["reported_4h_bess_capex_eur_per_kwh"] == pytest.approx(47.5)
    assert battery["log_capex_runtime"] == pytest.approx(math.log(27.5 * 1000.0))
    assert battery["raw_log_capex_runtime"] == pytest.approx(math.log(25.0 * 1000.0))
    assert state["technology_states"]["battery_energy"]["last_log_capex"] == pytest.approx(
        math.log(25.0 * 1000.0)
    )


def test_battery_power_cannot_be_mapped_to_discharger(monkeypatch, tmp_path):
    class FakeNetwork:
        def __init__(self, path):
            self.links = pd.DataFrame(
                {
                    "carrier": ["battery charger", "battery discharger"],
                    "capital_cost": [10.0, 0.0],
                }
            )
            self.generators = pd.DataFrame()
            self.storage_units = pd.DataFrame()
            self.stores = pd.DataFrame()
            self.meta = {}

    monkeypatch.setattr(alc.pypsa, "Network", FakeNetwork)

    with pytest.raises(ValueError, match="battery_power costs must not be mapped"):
        alc.update_network_costs(
            network_path="input.nc",
            learning_costs={
                "battery_power": {
                    "unit": "kW",
                    "capital_cost": 99.0,
                    "c_overnight": 100.0,
                }
            },
            tech_mapping={"battery discharger": "battery_power"},
            output_path=tmp_path / "output.nc",
            learning_cfg={},
            costs_file="costs_2030.csv",
        )


def test_legacy_global_current_window_behavior_is_preserved(monkeypatch):
    learning_cfg = {
        "seed": 0,
        "cost_expectations": {
            "mode": "block_average_expected",
            "annual_weights": [1, 1, 1, 1, 1],
        },
    }
    state = _make_state(
        2025,
        {"solar_power": 10.0, "onwind_power": 10.0, "battery_energy": 10.0},
    )
    monkeypatch.setattr(alc, "convert_to_capital_cost", lambda value, *args, **kwargs: float(value))
    path = np.log(np.array([[20.0, 30.0, 40.0, 50.0, 60.0]], dtype=float) * 1000.0)
    monkeypatch.setattr(
        alc,
        "_simulate_frozen_block_expectation_annual_paths",
        lambda artifacts, state, selected_model, elapsed_years, rng: {
            tech: path[:, :elapsed_years] for tech in artifacts
        },
    )

    diagnostics = alc._compute_block_average_expected_costs(
        artifacts={"solar_power": {}},
        state=state,
        current_year=2030,
        selected_model="shared_state_bayesian_regime_wright",
        learning_cfg=learning_cfg,
        costs_file="costs.csv",
    )

    assert diagnostics["solar_power"]["c_overnight"] == pytest.approx(30.0)
    assert diagnostics["solar_power"]["c_overnight_terminal_point"] == pytest.approx(60.0)
    assert json.loads(diagnostics["solar_power"]["kernel_years_json"]) == [2025, 2026, 2027, 2028, 2029]


def test_exogenous_lagged_kernel_uses_priced_year_kernel(monkeypatch):
    learning_cfg = {
        "cost_expectations": {
            "mode": "block_average_expected",
            "kernel_mode": "technology_specific_lagged_window",
            "annual_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "lag_years_by_tech": {
                "solar_power": 3,
                "onwind_power": 4,
                "battery_energy": 2,
            },
        }
    }
    monkeypatch.setattr(alc, "convert_to_capital_cost", lambda value, *args, **kwargs: float(value))
    exogenous_df = pd.DataFrame(
        [
            {"technology": tech, "year": year, "c_overnight": float(year - 2000)}
            for tech in ("solar_power", "onwind_power", "battery_energy")
            for year in range(2022, 2031)
        ]
    )
    monkeypatch.setattr(
        alc,
        "load_exogenous_cost_path",
        lambda learning_cfg, selected_model: (exogenous_df, {"source": "stub", "interpolation": "linear"}),
    )

    costs, _ = alc.calculate_exogenous_learning_costs(
        learning_cfg=learning_cfg,
        selected_model="iea_weo_exogenous_path",
        current_year=2030,
        costs_file="costs.csv",
        runtime_metadata={
            "cost_expectation_mode": "block_average_expected",
            "cost_expectation_kernel_mode": "technology_specific_lagged_window",
        },
    )

    assert costs["solar_power"]["c_overnight"] == pytest.approx(25.0)
    assert json.loads(costs["solar_power"]["kernel_years_json"]) == [2023, 2024, 2025, 2026, 2027]
    assert costs["solar_power"]["c_overnight_terminal_point"] == pytest.approx(30.0)
