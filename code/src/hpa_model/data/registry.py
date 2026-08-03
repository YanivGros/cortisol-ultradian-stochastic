"""Registry for the copied packaged datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_ROOT = PROJECT_ROOT / "data" / "catalog" / "datasets"


@dataclass(frozen=True)
class SignalSpec:
    name: str
    column: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label: str
    id_col: str
    time_col: str
    signals: tuple[SignalSpec, ...]
    variants: tuple[str, ...] = ("raw", "shifted")

    @property
    def signal_names(self) -> tuple[str, ...]:
        return tuple(signal.name for signal in self.signals)

    @property
    def signal_columns(self) -> tuple[str, ...]:
        return tuple(signal.column for signal in self.signals)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "habs": DatasetSpec(
        name="habs",
        label="HABS",
        id_col="ID",
        time_col="time_min",
        signals=(
            SignalSpec("ACTH", "ACTH"),
            SignalSpec("Cortisol", "Cortisol"),
        ),
    ),
    "all_digitized": DatasetSpec(
        name="all_digitized",
        label="All Digitized",
        id_col="ID",
        time_col="time_min",
        signals=(SignalSpec("Cortisol", "cortisol"),),
    ),
    "digitize_2019": DatasetSpec(
        name="digitize_2019",
        label="Russell & Lightman",
        id_col="series_id",
        time_col="time_min",
        signals=(
            SignalSpec("ACTH", "ACTH"),
            SignalSpec("Cortisol", "value"),
        ),
    ),
    "habs_microdialysis_cortisol": DatasetSpec(
        name="habs_microdialysis_cortisol",
        label="HABS Microdialysis",
        id_col="MasterID",
        time_col="time_min",
        signals=(SignalSpec("Cortisol", "Cortisol"),),
    ),
}


def list_dataset_names() -> tuple[str, ...]:
    return tuple(sorted(DATASET_SPECS))


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {name}") from exc


def get_dataset_path(name: str, variant: str) -> Path:
    get_dataset_spec(name)
    filename = "data_raw.csv" if variant == "raw" else "data_shifted.csv"
    path = DATASETS_ROOT / name / variant / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_dataset(name: str, variant: str) -> pd.DataFrame:
    return pd.read_csv(get_dataset_path(name, variant))


def get_shift_params_path(name: str, variant: str = "shifted") -> Path:
    get_dataset_spec(name)
    path = DATASETS_ROOT / name / variant / "shift_params.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_shift_params(name: str, variant: str = "shifted") -> pd.DataFrame:
    return pd.read_csv(get_shift_params_path(name, variant))
