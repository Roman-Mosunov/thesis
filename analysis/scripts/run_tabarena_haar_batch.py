"""Run the TabArena Haar experiment for multiple dataset presets."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

from tabarena_dataset_presets import (
    DATASET_PRESETS,
    DEFAULT_TABARENA_METADATA_CSV,
    RECOMMENDED_SMALL_IMBALANCED_KEYS,
    format_dataset_presets_table,
    load_binary_presets_from_metadata,
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
        "--all-binary",
        action="store_true",
        help="Run all TabArena datasets marked as binary in the metadata CSV.",
    )
    parser.add_argument(
        "--tabarena-metadata-csv",
        type=Path,
        default=DEFAULT_TABARENA_METADATA_CSV,
        help="Path to tabarena_dataset_metadata.csv used for --all-binary mode.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining datasets if one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected dataset list and exit without running experiments.",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Optional cap on how many datasets from the selected list to run.",
    )
    parser.add_argument(
        "--start-at",
        type=str,
        default=None,
        help="Optional dataset key to resume from within the selected list.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    if args.list_dataset_presets:
        print(format_dataset_presets_table())
        return

    if args.all_binary:
        presets = load_binary_presets_from_metadata(metadata_csv_path=args.tabarena_metadata_csv)
    elif args.use_recommended_small_imbalanced:
        presets = parse_dataset_presets(",".join(RECOMMENDED_SMALL_IMBALANCED_KEYS))
    else:
        presets = parse_dataset_presets(args.dataset_presets)

    if args.start_at is not None:
        normalized_start = args.start_at.strip().lower().replace("_", "-")
        start_idx = next(
            (
                idx
                for idx, preset in enumerate(presets)
                if preset.key.strip().lower().replace("_", "-") == normalized_start
            ),
            None,
        )
        if start_idx is None:
            available = ", ".join(preset.key for preset in presets)
            raise KeyError(
                f"--start-at '{args.start_at}' is not in the selected dataset list: {available}"
            )
        presets = presets[start_idx:]

    if args.max_datasets is not None:
        if args.max_datasets < 1:
            raise ValueError("--max-datasets must be >= 1 when provided.")
        presets = presets[: args.max_datasets]

    if args.dry_run:
        print("Selected datasets:")
        for idx, preset in enumerate(presets, start=1):
            rate_text = (
                "unknown" if math.isnan(preset.minority_rate) else f"{preset.minority_rate:.4f}"
            )
            print(f"{idx:02d}. {preset.key} (minority_rate={rate_text}, n={preset.n_samples})")
        return

    script_path = Path(__file__).resolve().with_name("run_tabarena_haar_experiment.py")
    failures: list[tuple[str, int]] = []
    for idx, preset in enumerate(presets, start=1):
        cmd = [
            sys.executable,
            str(script_path),
        ]
        if preset.key in DATASET_PRESETS:
            cmd.extend(["--dataset-preset", preset.key])
        cmd.extend(
            [
                "--dataset-name",
                preset.dataset_name,
                "--dataset-id",
                str(preset.dataset_id),
                "--task-id",
                str(preset.task_id),
                *passthrough,
            ]
        )
        if preset.positive_label is not None:
            cmd.extend(["--positive-label", preset.positive_label])
        rate_text = (
            "unknown" if math.isnan(preset.minority_rate) else f"{preset.minority_rate:.4f}"
        )
        print(
            f"[{idx}/{len(presets)}] {preset.key} "
            f"(minority_rate={rate_text}, n={preset.n_samples})"
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
