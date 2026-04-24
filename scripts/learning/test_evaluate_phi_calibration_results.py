import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
LEARNING_DIR = ROOT_DIR / "scripts" / "learning"
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import evaluate_phi_calibration_results as evaluator
import submit_learning_phi_calibration_array as phi_array


class PhiCalibrationEvaluationTests(unittest.TestCase):
    def test_evaluate_tasks_ranks_completed_runs_by_target_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_eval_root = evaluator.ROOT_DIR
            old_submit_root = phi_array.ROOT_DIR
            evaluator.ROOT_DIR = root
            phi_array.ROOT_DIR = root
            try:
                tasks = phi_array.build_tasks(phi2_count=1, phi3_count=2)
                complete_task = tasks[0]
                missing_task = tasks[1]

                for rel_path in phi_array.phi_calibration_output_paths(complete_task):
                    path = root / rel_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.name.startswith("deployment_constraints_"):
                        pd.DataFrame(
                            [
                                {
                                    "year": 2025,
                                    "technology": "battery_energy",
                                    "constraint_basis_unit": "global_liion_equivalent_GWh",
                                    "realized_block_addition_constrained_basis": 48.0,
                                    "realized_block_addition_modeled": 12.0,
                                    "realized_seg1": 10.0,
                                    "realized_seg2": 2.0,
                                    "realized_seg3": 0.0,
                                    "realized_wedge_cost_eur": 123.0,
                                    "binding_slack": -1.0,
                                    "b1": 40.0,
                                    "b2": 75.0,
                                    "allowed_block_addition": 50.0,
                                    "battery_phi_block": 4.0,
                                }
                            ]
                        ).to_csv(path, index=False)
                    elif path.name.startswith("system_costs_"):
                        pd.DataFrame(
                            [
                                {
                                    "annualized_capital_cost_eur": 10.0,
                                    "operating_cost_eur": 5.0,
                                    "total_system_cost_eur": 15.0,
                                }
                            ]
                        ).to_csv(path, index=False)
                    else:
                        path.write_text("ok\n", encoding="utf-8")

                frame = evaluator.evaluate_tasks(
                    [missing_task, complete_task],
                    technology="battery_energy",
                    planning_year=2025,
                    target_block_addition=50.0,
                )
            finally:
                evaluator.ROOT_DIR = old_eval_root
                phi_array.ROOT_DIR = old_submit_root

            self.assertEqual(frame.iloc[0]["name"], complete_task["name"])
            self.assertTrue(bool(frame.iloc[0]["complete"]))
            self.assertEqual(frame.iloc[0]["target_abs_error"], 2.0)
            self.assertEqual(frame.iloc[0]["realized_wedge_cost_eur"], 123.0)
            self.assertEqual(frame.iloc[0]["total_system_cost_eur"], 15.0)
            self.assertFalse(bool(frame.iloc[1]["complete"]))
            self.assertGreater(frame.iloc[1]["missing_output_count"], 0)

    def test_discover_tasks_skips_shared_working_sector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "_bootstrap_state").mkdir()
            (root / "phi2_0p00_phi3_0p10").mkdir()

            tasks = evaluator.discover_tasks(root)

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["name"], "phi2_0p00_phi3_0p10")


if __name__ == "__main__":
    unittest.main()
