import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
LEARNING_DIR = ROOT_DIR / "scripts" / "learning"
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import submit_learning_phi_calibration_array as phi_array
import bootstrap_state_store
from bootstrap_state_store import restore_bootstrap_state


class PhiCalibrationGridTests(unittest.TestCase):
    def test_default_grid_has_stable_contract(self):
        tasks = phi_array.build_tasks()
        self.assertEqual(len(tasks), 84)
        self.assertEqual(tasks[0]["name"], "phi2_0p00_phi3_0p10")
        self.assertEqual(tasks[-1]["name"], "phi2_0p45_phi3_1p00")

        for task in tasks:
            self.assertGreaterEqual(task["phi3_pct_capex"], task["phi2_pct_capex"])
            self.assertEqual(task["b1_multiplier"], 0.8)
            self.assertEqual(task["b2_multiplier"], 1.5)
            self.assertEqual(task["planning_horizons"], [2020, 2025])
            self.assertTrue(
                task["sector_name"].startswith("Global_200/phi_calibration/"),
                task["sector_name"],
            )

    def test_worker_targets_single_solved_cost_log_not_full_bootstrap_rule(self):
        task = phi_array.build_tasks(phi2_count=1, phi3_count=1)[0]
        cmd = phi_array.build_worker_snakemake_cmd(
            task=task,
            job_dir=ROOT_DIR / "cluster_workdirs" / "dummy",
            overlay_path=ROOT_DIR / "overlay.yaml",
            snakemake_jobs="12",
        )
        joined = " ".join(cmd)

        self.assertNotIn("solve_sector_networks_myopic_learning_bootstrap", cmd)
        self.assertNotIn("solve_sector_networks_myopic_learning_bootstrap", joined)
        self.assertIn("cost_log_solved_elec_s_200_lcopt_1h_1h_2025_0.071_AB_0.0export_base.csv", joined)
        self.assertIn("postnetworks/elec_s_200_ec_lcopt_1h_1h_2025_0.071_AB_0.0export_base.nc", joined)
        self.assertIn("results/Global_200/phi_calibration/phi2_0p00_phi3_0p10/learning/", joined)
        self.assertIn("--forcerun", cmd)
        self.assertIn("code", cmd)
        self.assertNotIn("mtime", cmd)
        self.assertNotIn("solve_network_myopic_learning_bootstrap", cmd)
        self.assertNotIn("export_postsolve_learning_costs_learning_bootstrap", cmd)

    def test_restore_supports_nested_phi_target_without_copying_old_phi_runs(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT_DIR / "results")) as tmpdir:
            source = Path(tmpdir) / "Global_200"
            (source / "learning").mkdir(parents=True)
            (source / "learning" / "bootstrap_complete_elec_s_test.txt").write_text("ok\n", encoding="utf-8")
            (source / "prenetworks").mkdir()
            (source / "prenetworks" / "elec_s_test_export.nc").write_text("network\n", encoding="utf-8")
            (source / "configs").mkdir()
            (source / "configs" / "config.yaml").write_text("run: {}\n", encoding="utf-8")
            (source / "phi_calibration" / "old_run" / "learning").mkdir(parents=True)
            (source / "phi_calibration" / "old_run" / "learning" / "stale.csv").write_text(
                "old\n",
                encoding="utf-8",
            )

            target = source / "phi_calibration" / "new_run"
            summary = restore_bootstrap_state(
                source,
                target,
                exclude_relative_prefixes=("phi_calibration",),
                hardlink_first=False,
            )

            self.assertTrue((target / "learning" / "bootstrap_complete_elec_s_test.txt").is_file())
            self.assertTrue((target / "prenetworks" / "elec_s_test_export.nc").is_file())
            self.assertTrue((target / "configs" / "config.yaml").is_file())
            self.assertFalse((target / "phi_calibration" / "old_run").exists())
            self.assertGreater(summary["copied"] + summary["linked"], 0)

    def test_submission_restores_one_shared_phi_working_sector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            (source / "learning").mkdir(parents=True)
            (source / "learning" / "bootstrap_complete_elec_s_test.txt").write_text("ok\n", encoding="utf-8")
            (source / "prenetworks").mkdir()
            (source / "prenetworks" / "elec_s_test_export.nc").write_text("network\n", encoding="utf-8")

            old_root = phi_array.ROOT_DIR
            old_bootstrap_root = bootstrap_state_store.ROOT_DIR
            phi_array.ROOT_DIR = root
            bootstrap_state_store.ROOT_DIR = root
            try:
                tasks = phi_array.build_tasks(phi2_count=1, phi3_count=2)
                phi_array.restore_phi_calibration_state(tasks, source)
            finally:
                phi_array.ROOT_DIR = old_root
                bootstrap_state_store.ROOT_DIR = old_bootstrap_root

            shared = root / "results" / phi_array.resolve_working_sector_name()
            self.assertTrue((shared / "learning" / "bootstrap_complete_elec_s_test.txt").is_file())
            for task in tasks:
                self.assertFalse((root / "results" / task["sector_name"]).exists())

    def test_worker_symlinks_shared_phi_state_into_task_sector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            task = phi_array.build_tasks(phi2_count=1, phi3_count=1)[0]
            shared = job_dir / "results" / phi_array.resolve_working_sector_name()
            (shared / "learning").mkdir(parents=True)
            (shared / "learning" / "bootstrap_complete_elec_s_test.txt").write_text("ok\n", encoding="utf-8")
            (shared / "prenetworks").mkdir()
            (shared / "prenetworks" / "elec_s_test_export.nc").write_text("network\n", encoding="utf-8")

            linked = phi_array._stage_task_bootstrap_state(task, job_dir)
            target = job_dir / "results" / task["sector_name"]

            self.assertEqual(linked, 2)
            self.assertTrue((target / "learning" / "bootstrap_complete_elec_s_test.txt").is_symlink())
            self.assertTrue((target / "prenetworks" / "elec_s_test_export.nc").is_symlink())


if __name__ == "__main__":
    unittest.main()
