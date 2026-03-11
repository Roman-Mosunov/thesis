"""Run the TabArena Haar experiment for multiple dataset presets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tabarena_dataset_presets import (
    RECOMMENDED_SMALL_IMBALANCED_KEYS,
    format_dataset_presets_table,
    parse_dataset_presets,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Batch runner for run_tabarena_haar_experiment.py using dataset presets."
    )
    parser.add_argument(
        "--dataset-presets",
        type=str,
        default=",".join(RECOMMENDED_SMALL_IMBALANCED_KEYS),
        help="Comma-separated dataset preset keys.",
    )
    parser.add_argument(
        "--use-recommended-small-imbalanced",
        action="store_true",
        help="Use the curated small/imbalanced preset set and ignore --dataset-presets.",
    )
    parser.add_argument(
        "--list-dataset-presets",
        action="store_true",
        help="Print available presets and exit.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining datasets if one run fails.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    if args.list_dataset_presets:
        print(format_dataset_presets_table())
        return

    if args.use_recommended_small_imbalanced:
        presets = parse_dataset_presets(",".join(RECOMMENDED_SMALL_IMBALANCED_KEYS))
    else:
        presets = parse_dataset_presets(args.dataset_presets)

    script_path = Path(__file__).resolve().with_name("run_tabarena_haar_experiment.py")
    failures: list[tuple[str, int]] = []
    for idx, preset in enumerate(presets, start=1):
        cmd = [
            sys.executable,
            str(script_path),
            "--dataset-preset",
            preset.key,
            *passthrough,
        ]
        print(
            f"[{idx}/{len(presets)}] {preset.key} "
            f"(minority_rate={preset.minority_rate:.4f}, n={preset.n_samples})"
        )
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append((preset.key, result.returncode))
            print(f"Run failed for {preset.key} with exit code {result.returncode}.")
            if not args.continue_on_error:
                break

    if failures:
        failed = ", ".join(f"{key}:{code}" for key, code in failures)
        raise SystemExit(f"Batch completed with failures: {failed}")

    print("Batch completed successfully.")


if __name__ == "__main__":
    main()
