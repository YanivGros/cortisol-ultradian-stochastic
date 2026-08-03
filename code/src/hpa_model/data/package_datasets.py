"""Regenerate packaged raw and shifted datasets from source inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .registry import DATASETS_ROOT, PROJECT_ROOT
from .two_harmonic_shift import PERIOD_MIN, ShiftResult, infer_native_timestep, minutes_to_hhmm, shift_dataframe_by_peak24


RAW_INPUT_ROOT = PROJECT_ROOT / "data" / "raw_data_input"


@dataclass(frozen=True)
class DatasetPackageSpec:
    name: str
    raw_source: Path
    id_col: str
    raw_value_cols: tuple[str, ...]
    shifted_value_cols: tuple[str, ...]
    fit_value_col: str
    description: str


PACKAGE_SPECS: tuple[DatasetPackageSpec, ...] = (
    DatasetPackageSpec(
        name="habs",
        raw_source=RAW_INPUT_ROOT / "Upton et al. (2023) blood.csv",
        id_col="ID",
        raw_value_cols=("Cortisol", "ACTH"),
        shifted_value_cols=("Cortisol", "ACTH"),
        fit_value_col="Cortisol",
        description="HABS blood cortisol and ACTH series preserved at native per-ID sampling.",
    ),
    DatasetPackageSpec(
        name="all_digitized",
        raw_source=RAW_INPUT_ROOT / "Young et al. (2004).csv",
        id_col="ID",
        raw_value_cols=("cortisol",),
        shifted_value_cols=("cortisol",),
        fit_value_col="cortisol",
        description="Combined digitized cortisol time-series preserved at native per-ID sampling.",
    ),
    DatasetPackageSpec(
        name="digitize_2019",
        raw_source=RAW_INPUT_ROOT / "Russell & Lightman.csv",
        id_col="series_id",
        raw_value_cols=("value", "ACTH"),
        shifted_value_cols=("value", "ACTH"),
        fit_value_col="value",
        description="Russell & Lightman digitized ACTH and cortisol series preserved at native per-ID sampling.",
    ),
)


def _prepare_habs_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time = pd.to_datetime(frame["Time"], format="%H:%M:%S")
    raw = frame.loc[:, ["ID", "Time", "Cortisol", "ACTH"]].copy()
    raw = raw.dropna(subset=["ACTH", "Cortisol"], how="all").copy()
    raw["time_min"] = time.loc[raw.index].dt.hour * 60 + time.loc[raw.index].dt.minute
    raw["Time"] = time.loc[raw.index].dt.strftime("%H:%M")
    return raw.sort_values(["ID", "time_min"]).reset_index(drop=True)


def _prepare_all_digitized_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    raw = frame.rename(columns={"minute": "time_min"}).loc[:, ["ID", "time_min", "cortisol"]].copy()
    raw["time_min"] = pd.to_numeric(raw["time_min"], errors="coerce")
    raw["cortisol"] = pd.to_numeric(raw["cortisol"], errors="coerce")
    raw = raw.dropna(subset=["ID", "time_min", "cortisol"]).copy()
    return raw.sort_values(["ID", "time_min"]).reset_index(drop=True)


def _prepare_digitize_2019_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if {"series_id", "time_min", "value", "ACTH"}.issubset(frame.columns):
        prepared = frame.loc[:, ["series_id", "time_min", "value", "ACTH"]].copy()
    elif {"series_id", "time_min", "value"}.issubset(frame.columns):
        prepared = frame.loc[:, ["series_id", "time_min", "value"]].copy()
        prepared["ACTH"] = np.nan
    elif {"id", "time", "cortisol", "acth"}.issubset(frame.columns):
        prepared = (
            frame.rename(columns={"id": "series_id", "time": "time_min", "cortisol": "value", "acth": "ACTH"})
            .loc[:, ["series_id", "time_min", "value", "ACTH"]]
            .copy()
        )
        prepared["series_id"] = prepared["series_id"].astype(str).map(
            lambda value: value if value.endswith(".csv") else f"{value}.csv"
        )
    else:
        raise ValueError(
            "digitize_2019 source must contain either "
            "('series_id', 'time_min', 'value', 'ACTH') or ('id', 'time', 'cortisol', 'acth') columns"
        )

    prepared["time_min"] = pd.to_numeric(prepared["time_min"], errors="coerce")
    prepared["value"] = pd.to_numeric(prepared["value"], errors="coerce")
    prepared["ACTH"] = pd.to_numeric(prepared["ACTH"], errors="coerce")
    prepared = prepared.dropna(subset=["series_id", "time_min", "value"]).copy()
    return prepared.sort_values(["series_id", "time_min"]).reset_index(drop=True)


def _complete_native_grid(
    frame: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    value_cols: tuple[str, ...],
    time_label_col: str | None = None,
) -> pd.DataFrame:
    completed_groups: list[pd.DataFrame] = []
    for series_id, group in frame.groupby(id_col, sort=False):
        work = group.copy().sort_values(time_col)
        work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col]).copy()
        work = work.groupby([id_col, time_col], as_index=False).agg({column: "mean" for column in value_cols})
        times = work[time_col].to_numpy(dtype=float)
        step = infer_native_timestep(times)
        full_day = bool(times.size > 1 and (float(times.max()) - float(times.min()) >= PERIOD_MIN - 2.0 * step))
        if full_day:
            target = np.arange(0.0, PERIOD_MIN, step, dtype=float)
        else:
            target = np.arange(float(times.min()), float(times.max()) + 0.5 * step, step, dtype=float)
        completed = pd.DataFrame({id_col: series_id, time_col: target})
        for column in value_cols:
            values = pd.to_numeric(work[column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            if not np.any(valid):
                completed[column] = np.nan
                continue
            src_times = times[valid]
            src_values = values[valid]
            if full_day:
                ext_times = np.concatenate([src_times - PERIOD_MIN, src_times, src_times + PERIOD_MIN])
                ext_values = np.concatenate([src_values, src_values, src_values])
                completed[column] = np.interp(target, ext_times, ext_values)
            else:
                completed[column] = np.interp(target, src_times, src_values)
        if time_label_col:
            completed[time_label_col] = completed[time_col].map(minutes_to_hhmm)
        ordered_cols = [id_col]
        if time_label_col:
            ordered_cols.append(time_label_col)
        ordered_cols.extend([*value_cols, time_col])
        completed_groups.append(completed[ordered_cols])
    return pd.concat(completed_groups, ignore_index=True)


def build_raw_dataset(name: str) -> pd.DataFrame:
    for spec in PACKAGE_SPECS:
        if spec.name != name:
            continue
        if name == "habs":
            raw = _prepare_habs_raw(spec.raw_source)
            return _complete_native_grid(raw, id_col=spec.id_col, time_col="time_min", value_cols=spec.raw_value_cols, time_label_col="Time")
        if name == "all_digitized":
            raw = _prepare_all_digitized_raw(spec.raw_source)
            return _complete_native_grid(raw, id_col=spec.id_col, time_col="time_min", value_cols=spec.raw_value_cols)
        if name == "digitize_2019":
            raw = _prepare_digitize_2019_raw(spec.raw_source)
            return _complete_native_grid(raw, id_col=spec.id_col, time_col="time_min", value_cols=spec.raw_value_cols)
    raise KeyError(name)


def _plot_series_grid(frame: pd.DataFrame, *, id_col: str, time_col: str, value_col: str, output_path: Path, title: str) -> None:
    groups = list(frame.groupby(id_col, sort=False))
    n = len(groups)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 2.8 * nrows), squeeze=False)
    for ax, (series_id, group) in zip(axes.ravel(), groups, strict=False):
        ax.plot(group[time_col], group[value_col], linewidth=1.2)
        ax.set_title(str(series_id))
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(value_col)
        ax.set_xlim(0, 1440)
        ax.set_xticks([0, 360, 720, 1080, 1440], ["0", "6", "12", "18", "24"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_overlay(raw: pd.DataFrame, shifted: pd.DataFrame, *, id_col: str, raw_time_col: str, shifted_time_col: str, value_col: str, output_path: Path) -> None:
    groups = list(raw.groupby(id_col, sort=False))
    n = len(groups)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 2.8 * nrows), squeeze=False)
    shifted_groups = {series_id: group for series_id, group in shifted.groupby(id_col, sort=False)}
    for ax, (series_id, raw_group) in zip(axes.ravel(), groups, strict=False):
        shifted_group = shifted_groups[series_id]
        ax.plot(raw_group[raw_time_col], raw_group[value_col], label="raw", linewidth=1.1)
        ax.plot(shifted_group[shifted_time_col], shifted_group[value_col], label="shifted", linewidth=1.1)
        ax.set_title(str(series_id))
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(value_col)
        ax.set_xlim(0, 1440)
        ax.set_xticks([0, 360, 720, 1080, 1440], ["0", "6", "12", "18", "24"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_readme(spec: DatasetPackageSpec, raw: pd.DataFrame, shift_result: ShiftResult) -> None:
    dataset_dir = DATASETS_ROOT / spec.name
    raw_path = dataset_dir / "raw" / "data_raw.csv"
    shifted_path = dataset_dir / "shifted" / "data_shifted.csv"
    sidecar_path = dataset_dir / "shifted" / "shift_params.csv"
    value_col = spec.fit_value_col
    raw_steps = sorted(
        {
            infer_native_timestep(group["time_min"].to_numpy(dtype=float))
            for _, group in raw.groupby(spec.id_col, sort=False)
        }
    )
    shifted_steps = sorted(
        {
            infer_native_timestep(group["time_min"].to_numpy(dtype=float))
            for _, group in shift_result.shifted.groupby(spec.id_col, sort=False)
        }
    )

    def schema_table(frame: pd.DataFrame) -> str:
        lines = ["| Column | Dtype |", "| --- | --- |"]
        for column, dtype in frame.dtypes.items():
            lines.append(f"| `{column}` | `{dtype}` |")
        return "\n".join(lines)

    readme = "\n".join(
        [
            f"# {spec.name if spec.name != 'digitize_2019' else 'digitize_2019'}",
            "",
            "## Description",
            spec.description,
            "",
            "## Source Provenance",
            f"- Raw source: `{spec.raw_source.relative_to(PROJECT_ROOT)}`",
            "- Shifted source: `generated in-repo from the packaged raw file`",
            "- Shift convention: independent-phase two-harmonic fit with 24-hour phase peak aligned to `10:00`",
            "- Shift model: `y(t) = a24*sin(w24*t + phi24) + a12*sin(w12*t + phi12) + c`",
            "",
            "## Files",
            "- `raw/data_raw.csv`: packaged pre-shift dataset completed onto each ID native sampling grid",
            "- `shifted/data_shifted.csv`: packaged two-harmonic phase-shifted dataset completed onto each ID native sampling grid",
            "- `shifted/shift_params.csv`: per-ID two-harmonic fit metadata including `phase12`",
            "- `plots/raw_all_series_grid.pdf`: grid plot of all raw fit-signal series",
            "- `plots/raw_all_series_grid.png`: PNG copy of the raw grid plot",
            "- `plots/shifted_all_series_grid.pdf`: grid plot of all shifted fit-signal series",
            "- `plots/shifted_all_series_grid.png`: PNG copy of the shifted grid plot",
            "- `plots/cortisol_original_vs_shifted_sinepeak.pdf`: per-series overlay showing original vs shifted fit-signal traces",
            "",
            "## Raw File",
            f"`{raw_path.relative_to(dataset_dir)}` preserves each series sampling rate and fills missing native-grid times by interpolation.",
            "",
            "## Shifted File",
            f"`{shifted_path.relative_to(dataset_dir)}` applies the per-ID two-harmonic 24-hour phase alignment and fills missing native-grid times by interpolation.",
            "",
            "## Sidecar Metadata",
            f"`{sidecar_path.relative_to(dataset_dir)}` stores per-ID fit metadata including `phase24`, `phase12`, amplitudes, peaks, cost, fallback mode, and applied shift.",
            "",
            "## Plotting",
            f"The plots are generated from the raw and shifted files using fit value column `{value_col}` and time column `time_min`. Top and right spines are removed in all plots.",
            "",
            "## Column Schema",
            "### Raw",
            schema_table(raw),
            "",
            "### Shifted",
            schema_table(shift_result.shifted),
            "",
            "### Shift Parameters",
            schema_table(shift_result.metadata),
            "",
            "## Counts Summary",
            f"- Signals included: {', '.join(spec.raw_value_cols)}",
            f"- Raw rows: {len(raw)}",
            f"- Raw series: {raw[spec.id_col].nunique()}",
            f"- Raw inferred native time step(s): {', '.join(f'{step:g}' for step in raw_steps)} minutes",
            f"- Shifted rows: {len(shift_result.shifted)}",
            f"- Shifted series: {shift_result.shifted[spec.id_col].nunique()}",
            f"- Shifted inferred native time step(s): {', '.join(f'{step:g}' for step in shifted_steps)} minutes",
        ]
    )
    (dataset_dir / "README.md").write_text(readme + "\n")


def regenerate_dataset(spec: DatasetPackageSpec) -> dict[str, object]:
    dataset_dir = DATASETS_ROOT / spec.name
    raw_dir = dataset_dir / "raw"
    shifted_dir = dataset_dir / "shifted"
    plots_dir = dataset_dir / "plots"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shifted_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw = build_raw_dataset(spec.name)
    shift_result = shift_dataframe_by_peak24(
        raw,
        id_col=spec.id_col,
        time_col="time_min",
        fit_value_col=spec.fit_value_col,
        value_cols=spec.raw_value_cols,
        output_value_cols=spec.shifted_value_cols,
        time_label_col="Time",
    )
    shifted_completed = _complete_native_grid(
        shift_result.shifted,
        id_col=spec.id_col,
        time_col="time_min",
        value_cols=spec.shifted_value_cols,
        time_label_col="Time",
    )
    shift_result = ShiftResult(shifted=shifted_completed, metadata=shift_result.metadata)

    raw_path = raw_dir / "data_raw.csv"
    shifted_path = shifted_dir / "data_shifted.csv"
    sidecar_path = shifted_dir / "shift_params.csv"
    raw.to_csv(raw_path, index=False)
    shift_result.shifted.to_csv(shifted_path, index=False)
    shift_result.metadata.to_csv(sidecar_path, index=False)

    _plot_series_grid(
        raw,
        id_col=spec.id_col,
        time_col="time_min",
        value_col=spec.fit_value_col,
        output_path=plots_dir / "raw_all_series_grid.pdf",
        title=f"{spec.name} raw",
    )
    _plot_series_grid(
        raw,
        id_col=spec.id_col,
        time_col="time_min",
        value_col=spec.fit_value_col,
        output_path=plots_dir / "raw_all_series_grid.png",
        title=f"{spec.name} raw",
    )
    _plot_series_grid(
        shift_result.shifted,
        id_col=spec.id_col,
        time_col="time_min",
        value_col=spec.fit_value_col,
        output_path=plots_dir / "shifted_all_series_grid.pdf",
        title=f"{spec.name} shifted",
    )
    _plot_series_grid(
        shift_result.shifted,
        id_col=spec.id_col,
        time_col="time_min",
        value_col=spec.fit_value_col,
        output_path=plots_dir / "shifted_all_series_grid.png",
        title=f"{spec.name} shifted",
    )
    _plot_overlay(
        raw,
        shift_result.shifted,
        id_col=spec.id_col,
        raw_time_col="time_min",
        shifted_time_col="time_min",
        value_col=spec.fit_value_col,
        output_path=plots_dir / "cortisol_original_vs_shifted_sinepeak.pdf",
    )
    _write_readme(spec, raw, shift_result)
    return {
        "dataset": spec.name,
        "raw_rows": len(raw),
        "shifted_rows": len(shift_result.shifted),
        "series": int(raw[spec.id_col].nunique()),
        "sidecar": str(sidecar_path),
    }


def regenerate_all_datasets() -> list[dict[str, object]]:
    return [regenerate_dataset(spec) for spec in PACKAGE_SPECS]
