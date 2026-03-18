from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

METHOD_ORDER = [
    "uncalibrated_logistic",
    "platt",
    "spline_fixed",
    "beta",
    "isotonic",
    "haar_gridsearch_best",
]

METHOD_LABELS = {
    "uncalibrated_logistic": "Uncalibrated",
    "platt": "Platt",
    "spline_fixed": "Spline",
    "beta": "Beta",
    "isotonic": "Isotonic",
    "haar_gridsearch_best": "Haar",
}

METHOD_COLORS = {
    "uncalibrated_logistic": "#4C72B0",
    "platt": "#DD8452",
    "spline_fixed": "#55A868",
    "beta": "#64B5CD",
    "isotonic": "#CCB974",
    "haar_gridsearch_best": "#8172B2",
}

FOCUS_DATASETS = {
    "blood-transfusion-service-center": "blood_transfusion",
    "credit_card_clients_default": "credit_card_default",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "-").replace("/", "-")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metadata(root: Path) -> pd.DataFrame:
    meta_path = root / "analysis/scripts/metadata/tabarena_dataset_metadata.csv"
    meta = pd.read_csv(meta_path)
    meta["dataset_slug"] = meta["dataset_name"].astype(str).map(slugify)
    return meta


def pretty_dataset_name(raw_name: str) -> str:
    return " ".join(str(raw_name).replace("_", " ").replace("-", " ").split())


def markdown_table(df: pd.DataFrame) -> str:
    text_df = df.fillna("").astype(str)
    header = "| " + " | ".join(text_df.columns.tolist()) + " |"
    separator = "| " + " | ".join("---" for _ in text_df.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_df.values.tolist()]
    return "\n".join([header, separator, *rows]) + "\n"


def fmt_score(value: float) -> str:
    return f"{value:.4f}"


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


def collect_outputs(
    outputs_root: Path,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    meta_lookup = metadata.set_index("dataset_slug")
    metric_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    latest_paths: dict[str, Path] = {}

    for dataset_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        runs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
        if not runs:
            continue
        latest = runs[-1]
        latest_paths[dataset_dir.name] = latest

        final_metrics_path = latest / "final_test_metrics.csv"
        mapping_path = latest / "graph_summary_estimator_mapping.csv"
        run_metadata_path = latest / "run_metadata.json"
        if not (
            final_metrics_path.exists()
            and mapping_path.exists()
            and run_metadata_path.exists()
        ):
            continue

        final_metrics = pd.read_csv(final_metrics_path)
        mapping = pd.read_csv(mapping_path)
        run_meta = json.loads(run_metadata_path.read_text(encoding="utf-8"))

        label_meta = run_meta["dataset"]["label_mapping"]
        ds_meta = run_meta["dataset"]["tabarena_dataset_metadata"]

        if dataset_dir.name in meta_lookup.index:
            curation = meta_lookup.loc[dataset_dir.name]
            dataset_name = curation["dataset_name"]
            domain = curation["domain"]
            year = curation["year"]
        else:
            dataset_name = ds_meta["dataset_name"]
            domain = "unknown"
            year = None

        for _, row in final_metrics.iterrows():
            metric_row = row.to_dict()
            metric_row.update(
                {
                    "dataset": dataset_dir.name,
                    "dataset_name": dataset_name,
                    "domain": domain,
                    "year": year,
                    "n_rows": ds_meta["n_rows"],
                    "n_features": ds_meta["n_features"],
                    "dataset_positive_rate": label_meta["positive_rate"],
                    "run_id": latest.name,
                }
            )
            metric_rows.append(metric_row)

        for _, row in mapping.iterrows():
            mapping_row = row.to_dict()
            mapping_row.update(
                {
                    "dataset": dataset_dir.name,
                    "dataset_name": dataset_name,
                    "run_id": latest.name,
                }
            )
            mapping_rows.append(mapping_row)

        base = final_metrics.loc[final_metrics["method"] == "uncalibrated_logistic"].iloc[0]
        best_brier = final_metrics.sort_values(["brier_score", "ece", "log_loss"]).iloc[0]
        best_ece = final_metrics.sort_values(["ece", "brier_score", "log_loss"]).iloc[0]
        best_log = final_metrics.sort_values(["log_loss", "brier_score", "ece"]).iloc[0]
        beta_mapping = mapping.loc[mapping["method"] == "beta"].iloc[0]

        dataset_rows.append(
            {
                "dataset": dataset_dir.name,
                "dataset_name": dataset_name,
                "domain": domain,
                "year": year,
                "run_id": latest.name,
                "n_rows": ds_meta["n_rows"],
                "n_features": ds_meta["n_features"],
                "positive_rate": label_meta["positive_rate"],
                "best_brier_method": best_brier["method"],
                "best_brier": best_brier["brier_score"],
                "best_ece_method": best_ece["method"],
                "best_ece": best_ece["ece"],
                "best_log_method": best_log["method"],
                "best_log_loss": best_log["log_loss"],
                "uncal_brier": base["brier_score"],
                "brier_gain": base["brier_score"] - best_brier["brier_score"],
                "beta_monotone": bool(beta_mapping["is_monotone_non_decreasing"]),
            }
        )

    all_metrics = pd.DataFrame(metric_rows)
    all_mappings = pd.DataFrame(mapping_rows)
    dataset_summary = pd.DataFrame(dataset_rows).sort_values("dataset").reset_index(drop=True)
    return all_metrics, all_mappings, dataset_summary, latest_paths


def build_method_summary(all_metrics: pd.DataFrame, dataset_summary: pd.DataFrame) -> pd.DataFrame:
    metric_means = (
        all_metrics.groupby("method")[
            [
                "brier_score",
                "brier_calibration_loss",
                "brier_refinement_loss",
                "ece",
                "log_loss",
            ]
        ]
        .mean()
        .reindex(METHOD_ORDER)
    )
    metric_means["brier_wins"] = (
        dataset_summary["best_brier_method"].value_counts().reindex(METHOD_ORDER).fillna(0).astype(int)
    )
    metric_means["ece_wins"] = (
        dataset_summary["best_ece_method"].value_counts().reindex(METHOD_ORDER).fillna(0).astype(int)
    )
    metric_means["log_wins"] = (
        dataset_summary["best_log_method"].value_counts().reindex(METHOD_ORDER).fillna(0).astype(int)
    )
    metric_means = metric_means.reset_index().rename(columns={"index": "method"})
    return metric_means


def copy_focus_figures(latest_paths: dict[str, Path], figures_dir: Path) -> None:
    for dataset_slug, short_name in FOCUS_DATASETS.items():
        run_dir = latest_paths[dataset_slug]
        plots_dir = run_dir / "plots"
        shutil.copy2(
            plots_dir / "reliability_all_estimators_comparison.png",
            figures_dir / f"{short_name}_reliability.png",
        )
        shutil.copy2(
            plots_dir / "estimator_all_calibrators_comparison.png",
            figures_dir / f"{short_name}_mapping.png",
        )


def make_primary_results_figure(method_summary: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = [METHOD_LABELS[method] for method in method_summary["method"]]
    colors = [METHOD_COLORS[method] for method in method_summary["method"]]

    axes[0].bar(labels, method_summary["brier_score"], color=colors)
    axes[0].set_title("Mean Brier score across 30 datasets")
    axes[0].set_ylabel("Brier score")
    axes[0].tick_params(axis="x", rotation=30)
    for idx, value in enumerate(method_summary["brier_score"]):
        axes[0].text(idx, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    axes[1].bar(labels, method_summary["brier_wins"], color=colors)
    axes[1].set_title("Number of per-dataset Brier wins")
    axes[1].set_ylabel("Datasets won")
    axes[1].tick_params(axis="x", rotation=30)
    for idx, value in enumerate(method_summary["brier_wins"]):
        axes[1].text(idx, value + 0.1, str(int(value)), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_brier_decomposition_figure(method_summary: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.5))

    labels = [METHOD_LABELS[method] for method in method_summary["method"]]
    calibration = method_summary["brier_calibration_loss"]
    refinement = method_summary["brier_refinement_loss"]

    ax.bar(labels, calibration, color="#C44E52", label="Calibration loss")
    ax.bar(labels, refinement, bottom=calibration, color="#9C9EDE", label="Refinement loss")
    ax.set_title("Mean Brier decomposition")
    ax.set_ylabel("Brier score = calibration loss + refinement loss")
    ax.tick_params(axis="x", rotation=30)

    totals = calibration + refinement
    for idx, total in enumerate(totals):
        ax.text(idx, total + 0.002, f"{total:.3f}", ha="center", va="bottom", fontsize=10)

    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_scope_snippet(dataset_summary: pd.DataFrame, output_path: Path) -> None:
    n_datasets = len(dataset_summary)
    top_domains = (
        dataset_summary["domain"]
        .value_counts()
        .head(5)
        .rename_axis("domain")
        .reset_index(name="count")
    )
    top_domain_text = ", ".join(
        f"{row.domain} ({int(row.count)})" for row in top_domains.itertuples(index=False)
    )

    lines = [
        f"- `{n_datasets}` binary TabArena datasets from {top_domain_text}.",
        (
            f"- Sample size range: `{fmt_int(dataset_summary['n_rows'].min())}` to "
            f"`{fmt_int(dataset_summary['n_rows'].max())}` "
            f"(median `{fmt_int(dataset_summary['n_rows'].median())}`)."
        ),
        (
            f"- Feature range: `{fmt_int(dataset_summary['n_features'].min())}` to "
            f"`{fmt_int(dataset_summary['n_features'].max())}` "
            f"(median `{fmt_int(dataset_summary['n_features'].median())}`)."
        ),
        (
            f"- Positive-class rate: `{fmt_pct(dataset_summary['positive_rate'].min())}` to "
            f"`{fmt_pct(dataset_summary['positive_rate'].max())}` "
            f"(median `{fmt_pct(dataset_summary['positive_rate'].median())}`)."
        ),
        "- All summaries use the latest local run per dataset in `analysis/outputs/tabarena/...`.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_summary_snippet(method_summary: pd.DataFrame, output_path: Path) -> None:
    table = pd.DataFrame(
        {
            "Method": method_summary["method"].map(METHOD_LABELS),
            "Mean Brier": method_summary["brier_score"].map(fmt_score),
            "Mean CL": method_summary["brier_calibration_loss"].map(fmt_score),
            "Mean RL": method_summary["brier_refinement_loss"].map(fmt_score),
            "Mean ECE": method_summary["ece"].map(fmt_score),
            "Mean Log Loss": method_summary["log_loss"].map(fmt_score),
            "Brier Wins": method_summary["brier_wins"].astype(int).astype(str),
        }
    )
    output_path.write_text(markdown_table(table), encoding="utf-8")


def write_focus_snippet(
    dataset_slug: str,
    all_metrics: pd.DataFrame,
    all_mappings: pd.DataFrame,
    output_path: Path,
) -> None:
    focus_methods = [
        "uncalibrated_logistic",
        "beta",
        "isotonic",
        "haar_gridsearch_best",
    ]
    metric_slice = (
        all_metrics.loc[
            (all_metrics["dataset"] == dataset_slug) & (all_metrics["method"].isin(focus_methods)),
            ["method", "brier_score", "ece", "log_loss"],
        ]
        .copy()
        .set_index("method")
        .reindex(focus_methods)
        .reset_index()
    )

    if dataset_slug == "credit_card_clients_default":
        mapping_slice = (
            all_mappings.loc[
                (all_mappings["dataset"] == dataset_slug)
                & (all_mappings["method"].isin(focus_methods)),
                ["method", "is_monotone_non_decreasing"],
            ]
            .copy()
            .set_index("method")
            .reindex(focus_methods)
            .reset_index()
        )
        metric_slice["monotone"] = mapping_slice["is_monotone_non_decreasing"].map(
            lambda value: "Yes" if bool(value) else "No"
        )
        table = pd.DataFrame(
            {
                "Method": metric_slice["method"].map(METHOD_LABELS),
                "Brier": metric_slice["brier_score"].map(fmt_score),
                "ECE": metric_slice["ece"].map(fmt_score),
                "Log loss": metric_slice["log_loss"].map(fmt_score),
                "Monotone": metric_slice["monotone"],
            }
        )
    else:
        table = pd.DataFrame(
            {
                "Method": metric_slice["method"].map(METHOD_LABELS),
                "Brier": metric_slice["brier_score"].map(fmt_score),
                "ECE": metric_slice["ece"].map(fmt_score),
                "Log loss": metric_slice["log_loss"].map(fmt_score),
            }
        )

    output_path.write_text(markdown_table(table), encoding="utf-8")


def write_dataset_results_snippets(dataset_summary: pd.DataFrame, snippets_dir: Path) -> None:
    ordered = dataset_summary.sort_values("positive_rate").reset_index(drop=True)
    table = pd.DataFrame(
        {
            "Dataset": ordered["dataset_name"].map(pretty_dataset_name),
            "n": ordered["n_rows"].map(fmt_int),
            "Pos. rate": ordered["positive_rate"].map(fmt_pct),
            "Best by Brier": ordered["best_brier_method"].map(METHOD_LABELS),
            "Best Brier": ordered["best_brier"].map(fmt_score),
            "Gain vs uncal.": ordered["brier_gain"].map(fmt_score),
        }
    )
    midpoint = (len(table) + 1) // 2
    part_1 = table.iloc[:midpoint].reset_index(drop=True)
    part_2 = table.iloc[midpoint:].reset_index(drop=True)
    (snippets_dir / "all_dataset_results_part1.md").write_text(
        markdown_table(part_1),
        encoding="utf-8",
    )
    (snippets_dir / "all_dataset_results_part2.md").write_text(
        markdown_table(part_2),
        encoding="utf-8",
    )


def main() -> None:
    root = repo_root()
    presentation_dir = root / "analysis/presentation"
    assets_dir = presentation_dir / "assets"
    figures_dir = assets_dir / "figures"
    data_dir = assets_dir / "data"
    snippets_dir = assets_dir / "snippets"

    for path in (assets_dir, figures_dir, data_dir, snippets_dir):
        ensure_dir(path)

    metadata = load_metadata(root)
    outputs_root = root / "analysis/outputs/tabarena"
    all_metrics, all_mappings, dataset_summary, latest_paths = collect_outputs(
        outputs_root,
        metadata,
    )
    method_summary = build_method_summary(all_metrics, dataset_summary)

    all_metrics.to_csv(data_dir / "all_metrics_latest.csv", index=False)
    all_mappings.to_csv(data_dir / "all_mappings_latest.csv", index=False)
    dataset_summary.to_csv(data_dir / "dataset_summary_latest.csv", index=False)
    method_summary.to_csv(data_dir / "method_summary_latest.csv", index=False)

    copy_focus_figures(latest_paths, figures_dir)
    make_primary_results_figure(method_summary, figures_dir / "aggregate_primary_results.png")
    make_brier_decomposition_figure(
        method_summary,
        figures_dir / "aggregate_brier_decomposition.png",
    )

    write_scope_snippet(dataset_summary, snippets_dir / "dataset_scope.md")
    write_method_summary_snippet(method_summary, snippets_dir / "method_summary.md")
    write_focus_snippet(
        "blood-transfusion-service-center",
        all_metrics,
        all_mappings,
        snippets_dir / "focus_blood_metrics.md",
    )
    write_focus_snippet(
        "credit_card_clients_default",
        all_metrics,
        all_mappings,
        snippets_dir / "focus_credit_metrics.md",
    )
    write_dataset_results_snippets(dataset_summary, snippets_dir)

    print(f"Wrote assets to {assets_dir}")


if __name__ == "__main__":
    main()
