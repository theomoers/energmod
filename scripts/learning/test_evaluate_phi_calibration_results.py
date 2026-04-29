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
                                    "technology": "solar_power",
                                    "constraint_basis_unit": "GW",
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
                    technology="solar_power",
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

    def test_lp_file_is_not_required_for_completed_phi_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_eval_root = evaluator.ROOT_DIR
            old_submit_root = phi_array.ROOT_DIR
            evaluator.ROOT_DIR = root
            phi_array.ROOT_DIR = root
            try:
                task = phi_array.build_tasks(phi2_count=1, phi3_count=1)[0]
                for rel_path in phi_array.phi_calibration_output_paths(task):
                    path = root / rel_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.name.startswith("deployment_constraints_"):
                        pd.DataFrame(
                            [
                                {
                                    "year": 2025,
                                    "technology": "solar_power",
                                    "constraint_basis_unit": "GW",
                                    "realized_block_addition_constrained_basis": 48.0,
                                }
                            ]
                        ).to_csv(path, index=False)
                    elif path.name.startswith("system_costs_"):
                        pd.DataFrame([{"total_system_cost_eur": 1.0}]).to_csv(path, index=False)
                    elif path.suffix == ".json":
                        path.write_text("{}\n", encoding="utf-8")
                    else:
                        path.write_text("ok\n", encoding="utf-8")

                frame = evaluator.evaluate_tasks(
                    [task],
                    technology=["solar_power"],
                    planning_year=2025,
                    target_block_additions={"solar_power": 50.0},
                )
            finally:
                evaluator.ROOT_DIR = old_eval_root
                phi_array.ROOT_DIR = old_submit_root

            self.assertTrue(bool(frame.iloc[0]["complete"]))
            self.assertEqual(frame.iloc[0]["missing_output_count"], 0)
            self.assertNotIn("lpfiles", str(frame.iloc[0]["first_missing_output"]))

    def test_network_fallback_reads_current_vintage_addition_not_cumulative_capacity(self):
        class Network:
            pass

        network = Network()
        network.generators = pd.DataFrame(
            [
                {
                    "carrier": "solar",
                    "build_year": 2020,
                    "p_nom_extendable": False,
                    "p_nom_opt": 100000.0,
                },
                {
                    "carrier": "solar",
                    "build_year": 2025,
                    "p_nom_extendable": True,
                    "p_nom_opt": 250000.0,
                },
                {
                    "carrier": "onwind",
                    "build_year": 2025,
                    "p_nom_extendable": True,
                    "p_nom_opt": 75000.0,
                },
            ]
        )

        metrics = evaluator._network_current_vintage_addition_metrics(
            network,
            technology="solar_power",
            planning_year=2025,
        )

        self.assertEqual(metrics["deployment_metrics_source"], "solved_network")
        self.assertEqual(metrics["constraint_basis_unit"], "GW")
        self.assertEqual(metrics["deployment_rows"], 1)
        self.assertAlmostEqual(metrics["realized_block_addition_constrained_basis"], 250.0)
        self.assertNotAlmostEqual(metrics["realized_block_addition_constrained_basis"], 350.0)

    def test_discover_tasks_skips_shared_working_sector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "_bootstrap_state").mkdir()
            (root / "phi2_0p00_phi3_0p10").mkdir()

            tasks = evaluator.discover_tasks(root)

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["name"], "phi2_0p00_phi3_0p10")

    def test_evaluate_tasks_scores_wind_and_solar_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_eval_root = evaluator.ROOT_DIR
            old_submit_root = phi_array.ROOT_DIR
            evaluator.ROOT_DIR = root
            phi_array.ROOT_DIR = root
            try:
                tasks = phi_array.build_tasks(phi2_count=1, phi3_count=1)
                task = tasks[0]
                for rel_path in phi_array.phi_calibration_output_paths(task):
                    path = root / rel_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.name.startswith("deployment_constraints_"):
                        pd.DataFrame(
                            [
                                {
                                    "year": 2025,
                                    "technology": "solar_power",
                                    "constraint_basis_unit": "GW",
                                    "realized_block_addition_constrained_basis": 90.0,
                                    "realized_wedge_cost_eur": 10.0,
                                },
                                {
                                    "year": 2025,
                                    "technology": "onwind_power",
                                    "constraint_basis_unit": "GW",
                                    "realized_block_addition_constrained_basis": 55.0,
                                    "realized_wedge_cost_eur": 20.0,
                                },
                            ]
                        ).to_csv(path, index=False)
                    elif path.name.startswith("system_costs_"):
                        pd.DataFrame([{"total_system_cost_eur": 1.0}]).to_csv(path, index=False)
                    else:
                        path.write_text("ok\n", encoding="utf-8")

                frame = evaluator.evaluate_tasks(
                    tasks,
                    technology=["solar_power", "onwind_power"],
                    planning_year=2025,
                    target_block_additions={"solar_power": 100.0, "onwind_power": 50.0},
                )
            finally:
                evaluator.ROOT_DIR = old_eval_root
                phi_array.ROOT_DIR = old_submit_root

            self.assertEqual(frame.iloc[0]["solar_power__target_abs_error"], 10.0)
            self.assertEqual(frame.iloc[0]["onwind_power__target_abs_error"], 5.0)
            self.assertEqual(frame.iloc[0]["aggregate_target_abs_error"], 15.0)
            self.assertEqual(frame.iloc[0]["scored_technology_count"], 2)
            self.assertEqual(
                frame.iloc[0]["solar_power__target_comparison_model_metric"],
                "realized_block_addition_constrained_basis",
            )
            self.assertEqual(frame.iloc[0]["solar_power__target_addition_kind"], "irena_capacity_delta")
            self.assertEqual(frame.iloc[0]["solar_power__target_basis_unit"], "GW")
            self.assertEqual(frame.iloc[0]["solar_power__target_block_start_year"], 2020)
            self.assertEqual(frame.iloc[0]["solar_power__target_block_end_year"], 2025)

    def test_load_target_block_additions_accepts_irena_long_capacity_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "irena_capacity.csv"
            records = []
            for region, multiplier in [("GLO", 1.0), ("USA", 100.0)]:
                for year, product, ongrid, offgrid in [
                    (2020, "Solar photovoltaic", 1000, 50),
                    (2025, "Solar photovoltaic", 3000, 100),
                    (2020, "Onshore wind energy", 2000, 0),
                    (2025, "Onshore wind energy", 2500, 20),
                ]:
                    records.extend(
                        [
                            {
                                "Region/area (ISO 3)": region,
                                "Region/area": region,
                                "Year": year,
                                "Data Type": "Electrical Capacity",
                                "Product Name": product,
                                "Grid Type": "OnGrid",
                                "Unit": "Megawatt",
                                " Value ": f"{ongrid * multiplier:,.0f}",
                            },
                            {
                                "Region/area (ISO 3)": region,
                                "Region/area": region,
                                "Year": year,
                                "Data Type": "Electrical Capacity",
                                "Product Name": product,
                                "Grid Type": "OffGrid",
                                "Unit": "Megawatt",
                                " Value ": f"{offgrid * multiplier:,.0f}",
                            },
                        ]
                    )
            pd.DataFrame(records).to_csv(path, index=False)

            targets = evaluator.load_target_block_additions(path, planning_year=2025)

        self.assertAlmostEqual(targets["solar_power"], 2.05)
        self.assertAlmostEqual(targets["onwind_power"], 0.52)
        self.assertNotAlmostEqual(targets["solar_power"], 3.10)
        self.assertNotAlmostEqual(targets["onwind_power"], 2.52)

    def test_load_target_block_additions_rejects_non_irena_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.csv"
            pd.DataFrame(
                [{"technology": "solar_power", "target_block_addition": 100.0}]
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "raw IRENA long-format"):
                evaluator.load_target_block_additions(path, planning_year=2025)

    def test_load_target_block_additions_requires_exact_five_year_irena_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "irena_capacity.csv"
            pd.DataFrame(
                [
                    {
                        "Region/area (ISO 3)": "GLO",
                        "Region/area": "World",
                        "Year": 2025,
                        "Data Type": "Electrical Capacity",
                        "Product Name": product,
                        "Grid Type": "OnGrid",
                        "Unit": "Megawatt",
                        " Value ": "1,000",
                    }
                    for product in ["Solar photovoltaic", "Onshore wind energy"]
                ]
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "years \\[2020\\]"):
                evaluator.load_target_block_additions(path, planning_year=2025)

    def test_evaluate_tasks_rejects_target_comparison_with_wrong_model_unit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_eval_root = evaluator.ROOT_DIR
            old_submit_root = phi_array.ROOT_DIR
            evaluator.ROOT_DIR = root
            phi_array.ROOT_DIR = root
            try:
                tasks = phi_array.build_tasks(phi2_count=1, phi3_count=1)
                task = tasks[0]
                for rel_path in phi_array.phi_calibration_output_paths(task):
                    path = root / rel_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.name.startswith("deployment_constraints_"):
                        pd.DataFrame(
                            [
                                {
                                    "year": 2025,
                                    "technology": "solar_power",
                                    "constraint_basis_unit": "MW",
                                    "realized_block_addition_constrained_basis": 90000.0,
                                }
                            ]
                        ).to_csv(path, index=False)
                    elif path.name.startswith("system_costs_"):
                        pd.DataFrame([{"total_system_cost_eur": 1.0}]).to_csv(path, index=False)
                    else:
                        path.write_text("ok\n", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "expected 'GW'"):
                    evaluator.evaluate_tasks(
                        tasks,
                        technology=["solar_power"],
                        planning_year=2025,
                        target_block_additions={"solar_power": 100.0},
                    )
            finally:
                evaluator.ROOT_DIR = old_eval_root
                phi_array.ROOT_DIR = old_submit_root


if __name__ == "__main__":
    unittest.main()
