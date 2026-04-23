import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from learning import deployment_constraints  # noqa: E402


def _learning_cfg(reference_method="growth_projected"):
    return {
        "engine": "stochastic_forecast",
        "selected_model": "shared_state_bayesian_regime_wright",
        "deployment_constraints": {
            "enabled": True,
            "formulation": "three_segment_wedge",
            "mode": "both",
            "technologies": ["battery_energy"],
            "wedge": {
                "level": "global",
                "reference_method": reference_method,
                "reference_statistic": "mean",
                "b1_multiplier": 0.8,
                "b2_multiplier": 1.2,
                "phi2_pct_capex": 0.5,
                "phi3_pct_capex": 1.0,
                "growth_smoothing_years": 3,
            },
        }
    }


def test_battery_growth_projected_wedge_evolves_thresholds_by_year(monkeypatch):
    history = pd.DataFrame(
        [
            {"region": "GLOBAL", "technology": "battery_energy", "year": 2021, "annual_addition": 10.0, "basis_unit": "local_energy"},
            {"region": "GLOBAL", "technology": "battery_energy", "year": 2022, "annual_addition": 20.0, "basis_unit": "local_energy"},
            {"region": "GLOBAL", "technology": "battery_energy", "year": 2023, "annual_addition": 40.0, "basis_unit": "local_energy"},
            {"region": "GLOBAL", "technology": "battery_energy", "year": 2024, "annual_addition": 80.0, "basis_unit": "local_energy"},
        ]
    )

    monkeypatch.setattr(
        deployment_constraints,
        "load_deployment_country_history",
        lambda learning_cfg, current_year, technologies=None, config_file=None, runtime_state_payload=None: history.copy(),
    )

    frame_2030 = deployment_constraints.build_deployment_wedge_table_from_history(
        _learning_cfg(),
        current_year=2030,
        technologies=["battery_energy"],
    )
    frame_2035 = deployment_constraints.build_deployment_wedge_table_from_history(
        _learning_cfg(),
        current_year=2035,
        technologies=["battery_energy"],
    )

    row_2030 = frame_2030.loc[frame_2030["technology"].eq("battery_energy")].iloc[0]
    row_2035 = frame_2035.loc[frame_2035["technology"].eq("battery_energy")].iloc[0]

    assert row_2030["reference_method"] == "growth_projected"
    assert row_2035["reference_method"] == "growth_projected"
    assert row_2035["b1"] > row_2030["b1"]
    assert row_2035["b2"] > row_2030["b2"]
    assert row_2035["reference_block_addition"] > row_2030["reference_block_addition"]


def test_growth_projected_reference_fails_when_history_is_insufficient(monkeypatch):
    history = pd.DataFrame(
        [
            {"region": "GLOBAL", "technology": "battery_energy", "year": 2024, "annual_addition": 80.0, "basis_unit": "local_energy"},
        ]
    )

    monkeypatch.setattr(
        deployment_constraints,
        "load_deployment_country_history",
        lambda learning_cfg, current_year, technologies=None, config_file=None, runtime_state_payload=None: history.copy(),
    )

    with pytest.raises(ValueError, match="Could not compute growth_projected"):
        deployment_constraints.build_deployment_wedge_table_from_history(
            _learning_cfg(),
            current_year=2030,
            technologies=["battery_energy"],
        )


def test_battery_history_uses_runtime_modeled_capacity_history_before_static_csv(monkeypatch):
    runtime_state_payload = {
        "modeled_capacity_history": {
            "battery_energy": {
                "2020": 100.0,
                "2025": 350.0,
                "2030": 850.0,
            }
        }
    }

    monkeypatch.setattr(
        deployment_constraints,
        "_load_default_runtime_state_payload",
        lambda learning_cfg: runtime_state_payload,
    )
    monkeypatch.setattr(
        deployment_constraints,
        "_battery_capacity_history_to_country_annual_additions",
        lambda battery_csv, current_year: pd.DataFrame(
            [
                {
                    "region": "US",
                    "technology": "battery_energy",
                    "year": 2025,
                    "annual_addition": 9999.0,
                    "basis_unit": "MWh",
                }
            ]
        ),
    )

    history = deployment_constraints.load_deployment_country_history(
        _learning_cfg(),
        current_year=2035,
        technologies=["battery_energy"],
    )

    battery = history.loc[history["technology"].eq("battery_energy")].reset_index(drop=True)
    assert set(battery["region"]) == {"GLOBAL"}
    assert set(battery["basis_unit"]) == {"GWh"}
    assert battery.loc[battery["year"].eq(2021), "annual_addition"].iloc[0] == 50.0
    assert battery.loc[battery["year"].eq(2025), "annual_addition"].iloc[0] == 50.0
    assert battery.loc[battery["year"].eq(2026), "annual_addition"].iloc[0] == 100.0
    assert battery.loc[battery["year"].eq(2030), "annual_addition"].iloc[0] == 100.0


def test_battery_history_falls_back_to_static_csv_when_runtime_state_missing(monkeypatch):
    monkeypatch.setattr(
        deployment_constraints,
        "_load_default_runtime_state_payload",
        lambda learning_cfg: {},
    )
    monkeypatch.setattr(
        deployment_constraints,
        "_battery_capacity_history_to_country_annual_additions",
        lambda battery_csv, current_year: pd.DataFrame(
            [
                {
                    "region": "US",
                    "technology": "battery_energy",
                    "year": 2025,
                    "annual_addition": 123.0,
                    "basis_unit": "MWh",
                }
            ]
        ),
    )

    history = deployment_constraints.load_deployment_country_history(
        _learning_cfg(),
        current_year=2030,
        technologies=["battery_energy"],
    )

    battery = history.loc[history["technology"].eq("battery_energy")].reset_index(drop=True)
    assert len(battery) == 1
    assert battery.loc[0, "region"] == "US"
    assert battery.loc[0, "annual_addition"] == 123.0
