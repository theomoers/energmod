#!/usr/bin/env python3
"""Reusable mock smoke test for energymod learning runtime."""

import argparse
import json
from pathlib import Path

from learning_runtime_smoke_utils import ALL_RUNTIME_MODELS, run_model_smoke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/tmp/energymod_mock_learning_smoke")
    parser.add_argument("--models", nargs="+", default=ALL_RUNTIME_MODELS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cost-expectation-mode",
        default="point_cost",
        choices=["point_cost", "block_average_expected"],
    )
    args = parser.parse_args()
    invalid_models = sorted(set(args.models) - set(ALL_RUNTIME_MODELS))
    if invalid_models:
        raise ValueError(f"Unsupported runtime models: {invalid_models}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_model_smoke(
            model_name,
            output_root,
            seed=args.seed,
            cost_expectation_mode=args.cost_expectation_mode,
        )
        for model_name in args.models
    ]
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"LEARNING_RUNTIME_SMOKE_OK {summary_path}")


if __name__ == "__main__":
    main()
