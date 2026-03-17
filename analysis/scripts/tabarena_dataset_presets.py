"""Dataset presets for TabArena calibration experiments."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPreset:
    key: str
    dataset_name: str
    dataset_id: int
    task_id: int
    positive_label: str | None
    minority_rate: float
    n_samples: int
    notes: str


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "blood-transfusion-service-center": DatasetPreset(
        key="blood-transfusion-service-center",
        dataset_name="blood-transfusion-service-center",
        dataset_id=46913,
        task_id=363621,
        positive_label="Yes",
        minority_rate=0.2400,
        n_samples=748,
        notes="Current default baseline dataset.",
    ),
    "kddcup09-appetency": DatasetPreset(
        key="kddcup09-appetency",
        dataset_name="kddcup09_appetency",
        dataset_id=46939,
        task_id=363683,
        positive_label="1",
        minority_rate=0.0178,
        n_samples=50000,
        notes="Very imbalanced; larger sample size and sparse/high-dimensional features.",
    ),
    "diabetes130us": DatasetPreset(
        key="diabetes130us",
        dataset_name="Diabetes130US",
        dataset_id=46922,
        task_id=363630,
        positive_label="Yes",
        minority_rate=0.0880,
        n_samples=71518,
        notes="Medical readmission target; moderately imbalanced and large.",
    ),
    "bank-marketing": DatasetPreset(
        key="bank-marketing",
        dataset_name="bank-marketing",
        dataset_id=46910,
        task_id=363618,
        positive_label="yes",
        minority_rate=0.1170,
        n_samples=45211,
        notes="Classic conversion dataset; moderate imbalance.",
    ),
    "credit-card-clients-default": DatasetPreset(
        key="credit-card-clients-default",
        dataset_name="credit_card_clients_default",
        dataset_id=46919,
        task_id=363627,
        positive_label="Yes",
        minority_rate=0.2212,
        n_samples=30000,
        notes="Well-known default benchmark; class imbalance is moderate.",
    ),
    "heloc": DatasetPreset(
        key="heloc",
        dataset_name="heloc",
        dataset_id=46932,
        task_id=363676,
        positive_label="Good",
        minority_rate=0.4781,
        n_samples=10459,
        notes="Useful interpretability benchmark but nearly balanced.",
    ),
    "polish-companies-bankruptcy": DatasetPreset(
        key="polish-companies-bankruptcy",
        dataset_name="polish_companies_bankruptcy",
        dataset_id=46950,
        task_id=363694,
        positive_label="Yes",
        minority_rate=0.0694,
        n_samples=5910,
        notes="Small/medium tabular; strong imbalance for bankruptcy detection.",
    ),
    "taiwanese-bankruptcy-prediction": DatasetPreset(
        key="taiwanese-bankruptcy-prediction",
        dataset_name="taiwanese_bankruptcy_prediction",
        dataset_id=46962,
        task_id=363706,
        positive_label="Yes",
        minority_rate=0.0323,
        n_samples=6819,
        notes="Small/medium and very imbalanced financial-risk dataset.",
    ),
    "coil2000-insurance-policies": DatasetPreset(
        key="coil2000-insurance-policies",
        dataset_name="coil2000_insurance_policies",
        dataset_id=46916,
        task_id=363624,
        positive_label="Yes",
        minority_rate=0.0597,
        n_samples=9822,
        notes="Small/medium with low positive rate in customer response.",
    ),
}


RECOMMENDED_SMALL_IMBALANCED_KEYS: tuple[str, ...] = (
    "taiwanese-bankruptcy-prediction",
    "coil2000-insurance-policies",
    "polish-companies-bankruptcy",
)
DEFAULT_TABARENA_METADATA_CSV = (
    Path(__file__).resolve().parent / "metadata" / "tabarena_dataset_metadata.csv"
)


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("_", "-")


def resolve_dataset_preset(key: str) -> DatasetPreset:
    normalized = _normalize_key(key)
    if normalized not in DATASET_PRESETS:
        available = ", ".join(sorted(DATASET_PRESETS))
        raise KeyError(f"Unknown dataset preset '{key}'. Available presets: {available}")
    return DATASET_PRESETS[normalized]


def parse_dataset_presets(raw: str) -> list[DatasetPreset]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if not keys:
        raise ValueError("Expected at least one dataset preset key.")
    return [resolve_dataset_preset(key) for key in keys]


def load_binary_presets_from_metadata(
    *,
    metadata_csv_path: Path | str = DEFAULT_TABARENA_METADATA_CSV,
) -> list[DatasetPreset]:
    csv_path = Path(metadata_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"TabArena metadata CSV not found: {csv_path}")

    presets: list[DatasetPreset] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("problem_type") != "binary":
                continue

            dataset_name = row["dataset_name"]
            normalized_key = _normalize_key(dataset_name)
            existing = DATASET_PRESETS.get(normalized_key)

            task_id = int(float(row["task_id"]))
            dataset_id = int(float(row["dataset_id"]))
            n_samples = int(float(row["num_instances"])) if row.get("num_instances") else 0
            minority_rate = existing.minority_rate if existing else math.nan
            positive_label = existing.positive_label if existing else None
            notes = (
                existing.notes
                if existing
                else "Loaded from TabArena metadata; positive class inferred automatically."
            )

            presets.append(
                DatasetPreset(
                    key=normalized_key,
                    dataset_name=dataset_name,
                    dataset_id=dataset_id,
                    task_id=task_id,
                    positive_label=positive_label,
                    minority_rate=minority_rate,
                    n_samples=n_samples,
                    notes=notes,
                )
            )

    return sorted(presets, key=lambda item: item.key)


def format_dataset_presets_table() -> str:
    header = (
        "key,dataset_name,dataset_id,task_id,minority_rate,n_samples,positive_label,notes"
    )
    lines = [header]
    for preset in sorted(DATASET_PRESETS.values(), key=lambda item: item.minority_rate):
        rate_text = "" if math.isnan(preset.minority_rate) else f"{preset.minority_rate:.4f}"
        lines.append(
            f"{preset.key},{preset.dataset_name},{preset.dataset_id},"
            f"{preset.task_id},{rate_text},{preset.n_samples},"
            f"{preset.positive_label or ''},{preset.notes}"
        )
    return "\n".join(lines)
