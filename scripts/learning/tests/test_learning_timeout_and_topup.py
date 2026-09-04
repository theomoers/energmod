import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import pypsa

ROOT_DIR = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT_DIR / "scripts"
LEARNING_DIR = SCRIPTS_DIR / "learning"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import report_cluster_run_status as status_report
import solve_network
import submit_learning_cluster_array as cluster_array
import submit_learning_sensitivity_ensemble_array as sensitivity_array
import submit_learning_sensitivity_topup as sensitivity_topup
import submit_learning_topup as topup


class TimeoutBudgetTests(unittest.TestCase):
    def test_initial_time_limit_uses_remaining_task_time(self):
        budget = {
            "start_epoch": 100,
            "max_seconds": 28800,
            "reserve_seconds": 900,
            "retry_min_seconds": 3600,
        }
        with mock.patch.object(solve_network.time, "time", return_value=110):
            options = solve_network._prepare_initial_solver_options(budget, "gurobi", {"Method": 2})
        self.assertEqual(options["TimeLimit"], 27890)
        self.assertEqual(options["Method"], 2)

    def test_retry_time_limit_uses_remaining_task_time(self):
        budget = {
            "start_epoch": 100,
            "max_seconds": 28800,
            "reserve_seconds": 900,
            "retry_min_seconds": 3600,
        }
        with mock.patch.object(solve_network.time, "time", return_value=20100):
            options = solve_network._prepare_retry_solver_options(budget, "gurobi", {"NumericFocus": 3})
        self.assertEqual(options["TimeLimit"], 7900)
        self.assertEqual(options["NumericFocus"], 3)

    def test_retry_skipped_if_remaining_time_too_small(self):
        budget = {
            "start_epoch": 100,
            "max_seconds": 28800,
            "reserve_seconds": 900,
            "retry_min_seconds": 3600,
        }
        with mock.patch.object(solve_network.time, "time", return_value=26100):
            options = solve_network._prepare_retry_solver_options(budget, "gurobi", {"NumericFocus": 3})
        self.assertIsNone(options)

    def test_retry_skipped_when_reserve_consumes_remaining_time(self):
        budget = {
            "start_epoch": 100,
            "max_seconds": 1000,
            "reserve_seconds": 100,
            "retry_min_seconds": 0,
        }
        with mock.patch.object(solve_network.time, "time", return_value=1000):
            options = solve_network._prepare_retry_solver_options(
                budget,
                "gurobi",
                {"NumericFocus": 3},
            )
        self.assertIsNone(options)

    def test_time_limit_is_not_acceptable_success(self):
        self.assertFalse(solve_network._acceptable_solve("ok", "time_limit"))
        self.assertTrue(solve_network._acceptable_solve("ok", "suboptimal"))

    def test_suboptimal_acceptance_is_audited_in_network_metadata(self):
        network = mock.Mock()
        network.meta = {}
        solve_network._record_accepted_solve(
            network,
            "ok",
            "suboptimal",
            "retry:gurobi-numeric-focus",
        )
        self.assertEqual(network.meta["solver_status"], "ok")
        self.assertEqual(network.meta["solver_termination_condition"], "suboptimal")
        self.assertEqual(
            network.meta["solver_acceptance_stage"],
            "retry:gurobi-numeric-focus",
        )


class LoadSheddingCapacityTests(unittest.TestCase):
    def _network(self):
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2040-01-01", periods=1, freq="h"))
        network.add("Bus", "A", carrier="AC")
        return network

    def test_default_restores_effectively_unbounded_feasibility_slack(self):
        network = solve_network.prepare_network(
            self._network(),
            {"load_shedding": 100},
        )
        self.assertEqual(float(network.generators.at["A load", "p_nom"]), 1e12)

    def test_configured_capacity_is_used_and_invalid_values_are_rejected(self):
        network = solve_network.prepare_network(
            self._network(),
            {"load_shedding": 100, "load_shedding_capacity": 2e6},
        )
        self.assertEqual(float(network.generators.at["A load", "p_nom"]), 2e6)

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            solve_network.prepare_network(
                self._network(),
                {"load_shedding": 100, "load_shedding_capacity": 0},
            )


class WrapperTimeoutTests(unittest.TestCase):
    def test_wrapper_timeout_preserves_job_dir(self):
        runner = ROOT_DIR / "scripts" / "learning" / "run_learning_cluster_job.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            job_root = tmpdir_path / "jobs"
            job_root.mkdir()
            dummy_runner = tmpdir_path / "dummy_runner.sh"
            dummy_runner.write_text("#!/usr/bin/env bash\nsleep 5\n", encoding="utf-8")
            dummy_runner.chmod(0o755)

            env = os.environ.copy()
            env["LEARNING_CLUSTER_JOB_RUNNER"] = str(dummy_runner)
            env["LEARNING_TASK_MAX_SECONDS"] = "1"
            env["LEARNING_TIMEOUT_GRACE_SECONDS"] = "1"
            env["LEARNING_SCENARIO_NAME"] = "timeout-test"

            result = subprocess.run(
                ["bash", str(runner), "shared_state_bayesian_regime_wright", "7", str(job_root)],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(ROOT_DIR),
                check=False,
            )

            self.assertEqual(result.returncode, 124)
            self.assertIn("hard wallclock limit", result.stderr)
            staged_dir = job_root / "energymod_timeout-test_shared_state_bayesian_regime_wright_s0007"
            self.assertTrue(staged_dir.exists())


class ConfigOverlayValidationTests(unittest.TestCase):
    def test_launcher_bundle_uses_immutable_snapshot_chain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            launcher_dir, hashes = cluster_array.stage_launcher_bundle(tmpdir)
            self.assertEqual(set(hashes), set(cluster_array.LAUNCHER_BUNDLE_FILES))
            worker = (launcher_dir / "run_learning_cluster_array_task.sh").read_text(encoding="utf-8")
            wrapper = (launcher_dir / "run_learning_cluster_job.sh").read_text(encoding="utf-8")
            stochastic = (launcher_dir / "run_learning_stochastic_job.sh").read_text(encoding="utf-8")
            self.assertIn(
                str(launcher_dir / "submit_learning_cluster_array.py"),
                worker,
            )
            self.assertIn(
                str(launcher_dir / "run_learning_stochastic_job.sh"),
                wrapper,
            )
            for payload in (worker, wrapper, stochastic):
                self.assertIn(f"ROOT_DIR={cluster_array.ROOT_DIR}", payload)
                self.assertNotIn(
                    'ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"',
                    payload,
                )

    def test_grid_command_pins_repository_root(self):
        args = mock.Mock(
            grid_array_concurrency=200,
            grid_mem="192G",
            grid_ncpus="12",
            grid_submit="batch",
            job_root="/tmp/jobs",
            conda_env="/tmp/env",
            run_mode="branch",
        )
        cmd = cluster_array.build_grid_run_cmd(
            args,
            Path("/tmp/manifest.json"),
            2,
            "/tmp/launchers/run_learning_cluster_array_task.sh",
        )
        self.assertIn(f"LEARNING_REPO_ROOT={cluster_array.ROOT_DIR}", cmd)


    def test_empty_overlay_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                cluster_array.validate_config_overlay(path, {"solving": {}})

    def test_semantic_noop_overlay_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "noop.yaml"
            path.write_text(
                "solving:\n  options:\n    skip_iterations: true\n",
                encoding="utf-8",
            )
            base = {"solving": {"options": {"skip_iterations": True}}}
            with self.assertRaisesRegex(ValueError, "no effective"):
                cluster_array.validate_config_overlay(path, base)

    def test_changed_overlay_records_hash_and_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "changed.yaml"
            path.write_text(
                "solving:\n  options:\n    skip_iterations: false\n",
                encoding="utf-8",
            )
            base = {"solving": {"options": {"skip_iterations": True}}}
            info = cluster_array.validate_config_overlay(path, base)
            self.assertEqual(info["changed_paths"], ["solving.options.skip_iterations"])
            self.assertEqual(len(info["sha256"]), 64)



    def test_default_stack_includes_workflow_and_learning_configs(self):
        names = [Path(path).name for path in cluster_array.default_configfiles()]
        self.assertEqual(
            names,
            [
                "config.default.yaml",
                "bundle_config.yaml",
                "powerplantmatching_config.yaml",
                "config.sectorelectrification.yaml",
                "config.yaml",
                "config.myopic.yaml",
                "config.learning.yaml",
            ],
        )

    def test_snapshot_merges_positional_config_and_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.yaml"
            override = root / "override.yaml"
            base.write_text("run:\n  sector_name: base\nsolving:\n  threads: 4\n", encoding="utf-8")
            override.write_text("run:\n  sector_name: override\n", encoding="utf-8")

            merged = cluster_array.load_merged_config([base, override])
            path, digest = cluster_array.write_effective_config_snapshot(root, merged)

            self.assertEqual(merged["run"]["sector_name"], "override")
            self.assertEqual(
                digest,
                cluster_array.hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                cluster_array.yaml.safe_load(path.read_text(encoding="utf-8")),
                merged,
            )

    def test_worker_verifies_and_passes_effective_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            effective, digest = cluster_array.write_effective_config_snapshot(
                root,
                {"learning": {"enabled": True}},
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "scenario_name": "config-test",
                            "resolved_sector_name": "Global_200/config-test",
                            "model": "correlated_geometric_random_walk",
                            "seed": 7,
                            "effective_config": str(effective),
                            "effective_config_sha256": digest,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = mock.Mock(manifest=str(manifest), job_root=str(root / "jobs"))

            with mock.patch.dict(os.environ, {"SGE_TASK_ID": "1"}, clear=True):
                with mock.patch.object(cluster_array.subprocess, "run") as run:
                    cluster_array.run_worker(args)

            worker_env = run.call_args.kwargs["env"]
            self.assertEqual(worker_env["LEARNING_EFFECTIVE_CONFIG"], str(effective))
            self.assertEqual(worker_env["LEARNING_EFFECTIVE_CONFIG_SHA256"], digest)
            self.assertNotIn("LEARNING_CONFIG_OVERLAY", worker_env)

            effective.write_text("learning:\n  enabled: false\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"SGE_TASK_ID": "1"}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    cluster_array.run_worker(args)


class TopupTests(unittest.TestCase):
    def write_config_pair(self, root: Path, seed_mode: str = "sequential") -> list[str]:
        cfg1 = root / "config.myopic.yaml"
        cfg1.write_text("solving: {}\n", encoding="utf-8")
        cfg2 = root / "config.learning.yaml"
        cfg2.write_text(
            "learning:\n"
            "  monte_carlo:\n"
            f"    seed_mode: {seed_mode}\n"
            "    random_seed: 123\n"
            "    seed_upper_bound: 1000\n",
            encoding="utf-8",
        )
        return [str(cfg1), str(cfg2)]

    def write_manifest(self, submit_dir: Path, scenario: str, model: str, seeds: list[int]) -> None:
        submit_dir.mkdir(parents=True, exist_ok=True)
        (submit_dir / "logs").mkdir(exist_ok=True)
        tasks = [
            {"scenario_name": scenario, "model": model, "seed": seed}
            for seed in seeds
        ]
        (submit_dir / "task_manifest.json").write_text(json.dumps(tasks), encoding="utf-8")

    def mark_completed(self, results_dir: Path, scenario: str, model: str, seed: int) -> None:
        seed_dir = results_dir / "learning-compact" / scenario / model / f"seed_s{seed:04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "compact_complete_test.json").write_text("{}", encoding="utf-8")

    def test_topup_generates_missing_draws_with_new_seeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            submit_root = tmpdir_path / "cluster_submissions"
            original = submit_root / "scenario_20260101_000000"
            self.write_manifest(original, "scenario", "correlated_geometric_random_walk", [0, 1, 2, 3, 4])
            configfiles = self.write_config_pair(tmpdir_path)
            results_dir = tmpdir_path / "results" / "Global_200"
            for seed in [0, 1, 2]:
                self.mark_completed(results_dir, "scenario", "correlated_geometric_random_walk", seed)

            with mock.patch.object(status_report, "LEGACY_RESULTS_DIR", results_dir):
                scenario, tasks, summary = topup.build_topup_tasks(original, [original], configfiles)

            self.assertEqual(scenario, "scenario")
            self.assertEqual(summary["correlated_geometric_random_walk"]["missing"], 2)
            self.assertEqual([task["seed"] for task in tasks], [5, 6])

    def test_running_submission_blocks_topup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            submit_dir = Path(tmpdir) / "submission"
            logs_dir = submit_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "run_learning_cluster_array_task.sh.o9999.1").write_text("Host: research01\n", encoding="utf-8")
            with mock.patch.object(status_report, "active_job_tasks", return_value={(9999, 1)}):
                with self.assertRaises(RuntimeError):
                    topup.ensure_no_running_tasks([submit_dir], allow_running=False)


class SensitivityTimeoutAndTopupTests(unittest.TestCase):
    def sensitivity_task(self, run_id: str, seed: int) -> dict:
        return {
            "ensemble_name": "sensitivity-test",
            "run_id": run_id,
            "group": "kernel",
            "draw_id": seed,
            "learning_seed": seed,
            "model": "shared_state_bayesian_regime_wright",
            "scenario_name": f"sens_{run_id}",
            "sector_name": "Global_200",
            "archive_sector_name": "Global_200",
            "run_mode": "branch",
            "bootstrap_group": "AB",
            "budget": "ne_si_6200",
            "n_draws": 3,
            "common_random_numbers": True,
            "config_overrides": {},
            "status": "pending",
        }

    def mark_sensitivity_complete(self, root: Path, task: dict) -> None:
        compact_dir = sensitivity_array.compact_bundle_dir_for_sector(task, root, task["archive_sector_name"])
        compact_dir.mkdir(parents=True, exist_ok=True)
        (compact_dir / "compact_complete_test.json").write_text("{}", encoding="utf-8")

    def test_sensitivity_topup_reruns_exact_missing_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tasks = [self.sensitivity_task("K1_UNIFORM_KERNEL", seed) for seed in [0, 1, 2]]
            self.mark_sensitivity_complete(tmpdir_path, tasks[0])
            self.mark_sensitivity_complete(tmpdir_path, tasks[2])

            with mock.patch.object(sensitivity_array, "ROOT_DIR", tmpdir_path):
                missing, summary = sensitivity_topup.build_topup_tasks(tasks)

            self.assertEqual([task["learning_seed"] for task in missing], [1])
            self.assertEqual([task["draw_id"] for task in missing], [1])
            self.assertEqual(missing[0]["status"], "topup")
            self.assertEqual(summary["K1_UNIFORM_KERNEL"]["completed"], 2)
            self.assertEqual(summary["K1_UNIFORM_KERNEL"]["missing"], 1)

    def test_sensitivity_grid_command_includes_timeout_budget(self):
        args = mock.Mock(
            grid_array_concurrency=200,
            grid_mem="90G",
            grid_ncpus="12",
            grid_submit="batch",
            job_root="/tmp/sensitivity-jobs",
            conda_env="/tmp/env",
            task_max_seconds=123,
            task_reserve_seconds=12,
            retry_min_seconds=34,
            timeout_grace_seconds=5,
            timeout_bin="/usr/bin/timeout",
            dry_run=False,
            overwrite=False,
        )
        cmd = sensitivity_array.build_grid_run_cmd(args, Path("/tmp/manifest.json"), [{}])
        self.assertIn("SENSITIVITY_TASK_MAX_SECONDS=123", cmd)
        self.assertIn("SENSITIVITY_TASK_RESERVE_SECONDS=12", cmd)
        self.assertIn("SENSITIVITY_RETRY_MIN_SECONDS=34", cmd)
        self.assertIn("SENSITIVITY_TIMEOUT_GRACE_SECONDS=5", cmd)
        self.assertIn("SENSITIVITY_TIMEOUT_BIN=/usr/bin/timeout", cmd)


class StatusClassificationTests(unittest.TestCase):
    def test_time_limit_failures_are_classified(self):
        self.assertEqual(status_report.classify_failure("Termination condition: time_limit"), "time_limit")
        self.assertEqual(status_report.classify_failure("Task exceeded hard wallclock limit"), "time_limit")
        self.assertEqual(status_report.classify_failure("returned non-zero exit status 124"), "time_limit")


if __name__ == "__main__":
    unittest.main()
