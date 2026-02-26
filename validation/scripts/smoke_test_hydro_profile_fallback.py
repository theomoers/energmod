#!/usr/bin/env python3
"""Smoke test for hydro profile fallback on a toy prenetwork structure."""

from pathlib import Path
import shutil
import sys

import pandas as pd
import pypsa


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import validation as validation_hooks  # noqa: E402


def _build_toy_network():
    n = pypsa.Network()
    snapshots = pd.date_range("2020-01-01", periods=4, freq="h")
    n.set_snapshots(snapshots)

    n.add("Bus", "AA_h0", carrier="AC", x=0.0, y=0.0)
    n.add("Bus", "AA_h1", carrier="AC", x=1.0, y=0.0)
    n.add("Bus", "BB_h0", carrier="AC", x=10.0, y=0.0)
    n.buses.loc[["AA_h0", "AA_h1"], "country"] = "AA"
    n.buses.loc[["BB_h0"], "country"] = "BB"
    n.buses["location"] = n.buses.index

    n.add("StorageUnit", "RES_OK", bus="AA_h0", carrier="hydro", p_nom=100.0, max_hours=6.0)
    n.add("StorageUnit", "RES_MISS", bus="AA_h1", carrier="hydro", p_nom=50.0, max_hours=6.0)
    n.add("StorageUnit", "RES_ZERO", bus="BB_h0", carrier="hydro", p_nom=80.0, max_hours=6.0)

    n.add("Generator", "ROR_OK", bus="AA_h0", carrier="ror", p_nom=120.0, p_max_pu=0.5)
    n.add("Generator", "ROR_MISS", bus="AA_h1", carrier="ror", p_nom=60.0, p_max_pu=0.3)
    n.add("Generator", "ROR_ZERO", bus="BB_h0", carrier="ror", p_nom=70.0, p_max_pu=0.4)

    donor_inflow = pd.Series([10.0, 20.0, 30.0, 40.0], index=snapshots)
    donor_ror = pd.Series([0.20, 0.40, 0.10, 0.00], index=snapshots)

    n.storage_units_t.inflow = pd.DataFrame(
        {
            "RES_OK": donor_inflow,
            "RES_ZERO": pd.Series(0.0, index=snapshots),
        },
        index=snapshots,
    )
    n.generators_t.p_max_pu = pd.DataFrame(
        {
            "ROR_OK": donor_ror,
            "ROR_ZERO": pd.Series(0.0, index=snapshots),
        },
        index=snapshots,
    )

    return n, donor_inflow, donor_ror


def main():
    n, donor_inflow, donor_ror = _build_toy_network()

    diag_root = REPO_ROOT / "validation" / ".tmp" / "ws4_hydro_profile_fallback_smoke"
    shutil.rmtree(diag_root, ignore_errors=True)

    config = {
        "global_specific": {
            "baseyear_generation": {
                "hydro_profile_fallback_enabled": True,
                "hydro_profile_fallback_diagnostics": True,
                "hydro_profile_fallback_diagnostics_dir": "validation/.tmp/ws4_hydro_profile_fallback_smoke",
            }
        }
    }

    result = validation_hooks.apply_hydro_profile_fallback_and_diagnostics(
        n,
        investment_year=2020,
        config=config,
        output_network_path=str(REPO_ROOT / "tmp" / "toy_prenetwork.nc"),
    )

    exp_res_miss = donor_inflow * (50.0 / 100.0)
    exp_res_zero = donor_inflow * (80.0 / 100.0)

    pd.testing.assert_series_equal(
        n.storage_units_t.inflow["RES_MISS"],
        exp_res_miss,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        n.storage_units_t.inflow["RES_ZERO"],
        exp_res_zero,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        n.generators_t.p_max_pu["ROR_MISS"],
        donor_ror,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        n.generators_t.p_max_pu["ROR_ZERO"],
        donor_ror,
        check_names=False,
    )

    diag_dir = Path(result["diagnostics_dir"])
    assert diag_dir.exists(), f"Diagnostics dir not found: {diag_dir}"

    before = pd.read_csv(diag_dir / "hydro_profile_pathology_assets_before.csv")
    after = pd.read_csv(diag_dir / "hydro_profile_pathology_assets_after.csv")
    actions = pd.read_csv(diag_dir / "hydro_profile_fallback_actions.csv")

    before_needs = int(before["needs_fallback"].sum())
    after_needs = int(after["needs_fallback"].sum())
    patched = int((actions["status"] == "patched").sum())

    assert before_needs == 4, f"expected 4 pathologies before fallback, got {before_needs}"
    assert after_needs == 0, f"expected 0 residual pathologies after fallback, got {after_needs}"
    assert patched == 4, f"expected 4 patched assets, got {patched}"

    print("Hydro fallback smoke test passed")
    print(f"Diagnostics dir: {diag_dir}")
    print(f"Pathologies before: {before_needs}")
    print(f"Patched actions: {patched}")
    print(f"Residual pathologies after: {after_needs}")


if __name__ == "__main__":
    main()
