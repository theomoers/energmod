import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validation  # noqa: E402


def test_irena_country_capacity_reference_supports_new_2025_schema(tmp_path):
    csv_path = tmp_path / "irena_new_schema.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Region/area (ISO 3),Region/area,Year,Data Type,Product Name,Grid Type,Unit, Value ",
                "DEU,Germany,2025,Electrical Capacity,Solar photovoltaic,OnGrid,Megawatt,\" 1,234 \"",
                "DEU,Germany,2025,Electrical Capacity,Solar photovoltaic,OffGrid,Megawatt,999",
                "GBR,United Kingdom,2025,Electrical Capacity,Offshore wind energy,OnGrid,Megawatt,456",
                "GBR,United Kingdom,2025,Other Metric,Offshore wind energy,OnGrid,Megawatt,999",
                "FRA,France,2025,Electrical Capacity,Onshore wind energy,OnGrid,Gigawatt,999",
            ]
        )
    )

    ref, used_year = validation._irena_country_capacity_reference(
        irena_csv=str(csv_path),
        year=2025,
        carrier_technology_map={
            "solar": ["Solar photovoltaic"],
            "offshore_wind": ["Offshore wind energy"],
        },
        fallback_to_latest=False,
    )

    expected = pd.DataFrame(
        [
            ("DE", "solar", 1234.0),
            ("GB", "offshore_wind", 456.0),
        ],
        columns=["country", "carrier", "reference_mw"],
    )
    pd.testing.assert_frame_equal(
        ref.sort_values(["country", "carrier"]).reset_index(drop=True),
        expected,
    )
    assert used_year == 2025


def test_irena_country_capacity_reference_keeps_legacy_schema_support(tmp_path):
    csv_path = tmp_path / "irena_legacy.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Technology,Country,2024,2025",
                "PV,DE,100,120",
                "Onshore,GB,200,250",
            ]
        )
    )

    ref, used_year = validation._irena_country_capacity_reference(
        irena_csv=str(csv_path),
        year=2025,
        carrier_technology_map={"solar": ["PV"], "onwind": ["Onshore"]},
        fallback_to_latest=False,
    )

    expected = pd.DataFrame(
        [
            ("DE", "solar", 120.0),
            ("GB", "onwind", 250.0),
        ],
        columns=["country", "carrier", "reference_mw"],
    )
    pd.testing.assert_frame_equal(
        ref.sort_values(["country", "carrier"]).reset_index(drop=True),
        expected,
    )
    assert used_year == 2025


@pytest.mark.parametrize(
    ("country", "carrier", "expected_mw"),
    [
        ("DE", "onwind", 68135.0),
        ("DE", "solar", 106272.0),
        ("GB", "offshore_wind", 16965.0),
    ],
)
def test_irena_country_capacity_reference_reads_real_2025_values(country, carrier, expected_mw):
    ref, used_year = validation._irena_country_capacity_reference(
        irena_csv=str(REPO_ROOT / "validation/data/IRENA_2025_capacity.csv"),
        year=2025,
        carrier_technology_map={
            "solar": ["Solar photovoltaic"],
            "onwind": ["Onshore wind energy"],
            "offshore_wind": ["Offshore wind energy"],
            "geothermal": ["Geothermal energy"],
        },
        fallback_to_latest=False,
    )

    value = ref.set_index(["country", "carrier"]).loc[(country, carrier), "reference_mw"]
    assert value == pytest.approx(expected_mw)
    assert used_year == 2025
