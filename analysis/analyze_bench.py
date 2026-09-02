"""Minimal analysis script for vLLM benchmark results.

Reads from bench/bench_results/{model}/{gpu}/{benchmark_name}/result/*.json and produces:
  - tail_latency_workload_grid_2x2_best.png
  - tail_latency_qps_grid_2x2_best.png
  - improvement_summary_heatmap_all.png
  - marginal_effect_addons_workloads_2qps_bars.png
  - marginal_effect_addons_qps_sweep_bars.png
  - efficiency_frontier_workloads_2qps_logy.png
  - efficiency_frontier_qps_sweep_logy.png

Usage:
  python analysis/analyze_bench.py --bench-dir bench/bench_results --out-dir figures
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from utils import (
    OPT_ORDER,
    WORKLOAD_TICK,
    load_bench_results,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

MODEL_SIZES = ["3B", "8B"]
MODEL_DISPLAY = {"3B": "Llama 3.2-3B", "8B": "Llama 3.1-8B"}

# Colors per optimization variant (same order as OPT_ORDER)
_PALETTE = [
    "#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22",
]
OPT_COLOR: dict[str, str] = dict(zip(OPT_ORDER, _PALETTE))

# Workload order for comparisons (sorted roughly by output size):
#   768→32, 128→128, 768→192, 128→512
WORKLOAD_ORDER = ["long_short", "short_short", "long_long", "short_long"]

# Marker shapes per workload type (for efficiency frontier scatter)
WORKLOAD_MARKERS: dict[str, str] = {
    "qps_sweep":   "o",
    "short_short": "s",
    "long_short":  "^",
    "short_long":  "D",
    "long_long":   "P",
}
WORKLOAD_DISPLAY: dict[str, str] = {
    "qps_sweep":   "512 input → 128 output",
    "short_short": "128 input → 128 output",
    "long_short":  "768 input → 32 output",
    "short_long":  "128 input → 512 output",
    "long_long":   "768 input → 192 output",
}

OPT_DISPLAY: dict[str, str] = {
    "no-opts":      "No optimizations",
    "cb":           "Continuous batching",
    "cb+cp":        "Continuous batching + Chunked prefill",
    "cb+pc":        "Continuous batching + Prefix caching",
    "cb+as":        "Continuous batching + Async scheduling",
    "default":      "Default optimizations (CB+CP+PC+AS)",
    "all+q":        "Default + Quantization (BnB)",
    "all+sd(1B)":   "Default + Speculative decoding (1B draft)",
    "all+sd(1B+p)": "Default + Speculative decoding (1B parallel)",
    "all+sd(3B)":   "Default + Speculative decoding (3B draft)",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved -> {path}")
    plt.close(fig)


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _present_opts(df: pd.DataFrame) -> list[str]:
    """Ordered list of opt_labels present in df."""
    have = set(df["opt_label"].unique())
    return [o for o in OPT_ORDER if o in have]


def _opt_legend_handles(opts: list[str]) -> list[mpatches.Patch]:
    return [mpatches.Patch(color=OPT_COLOR[o], label=OPT_DISPLAY.get(o, o)) for o in opts]


def _set_log_y(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))


def _plot_grouped_bars(
    ax: plt.Axes,
    *,
    data: pd.DataFrame,
    categories: list,
    category_col: str,
    metric: str,
    opts: list[str],
    category_tick: list[str],
    log_y: bool,
    x_tick_rotation: float = 0.0,
) -> None:
    """Grouped bars for a single metric and grouping axis."""
    x = np.arange(len(categories))
    width = 0.85 / max(len(opts), 1)
    offsets = (np.arange(len(opts)) - (len(opts) - 1) / 2) * width

    for i, opt in enumerate(opts):
        odf = data[data["opt_label"] == opt]
        vals = []
        for c in categories:
            rows = odf[odf[category_col] == c]
            vals.append(float(rows[metric].mean()) if not rows.empty else np.nan)

        ax.bar(
            x + offsets[i],
            vals,
            width * 0.95,
            color=OPT_COLOR.get(opt, "#777777"),
            alpha=0.9,
        )

    if log_y:
        _set_log_y(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(category_tick, fontsize=9, rotation=x_tick_rotation,
                       ha="right" if x_tick_rotation else "center")


def plot_tail_latency_grids(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    suffix: str = "",
    log_y: bool = True,
) -> None:
    """Generate model-row 2×2 grids for tail latency (p99 TTFT + p99 E2EL).

    Produces:
      - tail_latency_workload_grid_2x2_best.png  (workloads @ 2 req/s)
      - tail_latency_qps_grid_2x2_best.png       (QPS sweep, 512×128)
    """
    metrics = [
        ("p99_ttft_ms", "P99 TTFT (ms)"),
        ("p99_e2el_ms", "P99 E2EL (ms)"),
    ]

    df = _num(df, [m for m, _ in metrics] + ["request_rate"]).copy()

    models = [m for m in MODEL_SIZES if m in df["model_size"].unique()]
    if not models:
        print("  [skip] tail_latency_grids: no data")
        return

    # Determine which opts are present (global, for consistent colors)
    opts_all = _present_opts(df)
    if "no-opts" not in opts_all:
        print("  [skip] tail_latency_grids: no 'no-opts' baseline")
        return
    opts = ["no-opts"] + [o for o in OPT_ORDER if o in opts_all and o != "no-opts"]

    def _make_grid(
        *,
        slice_df: pd.DataFrame,
        categories: list,
        category_col: str,
        category_tick: list[str],
        x_tick_rotation: float,
        metrics: list[tuple[str, str]],
        out_name: str,
        title: str,
    ) -> None:
        # Make the 2×2 grids wider (slide-friendly, similar to the frontier figs).
        if len(metrics) == 2:
            figsize = (18.0, 3.35 * len(models))
        else:
            figsize = (4.55 * len(metrics), 3.25 * len(models))

        fig, axes = plt.subplots(
            len(models),
            len(metrics),
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"wspace": 0.10, "hspace": 0.20},
        )

        for r, model in enumerate(models):
            mdf = slice_df[slice_df["model_size"] == model].copy()
            for c, (metric, metric_label) in enumerate(metrics):
                ax = axes[r][c]
                if metric not in mdf.columns:
                    ax.set_visible(False)
                    continue
                mdf2 = mdf.dropna(subset=[metric]).copy()
                if not mdf2.empty:
                    mdf2 = mdf2[mdf2[metric].astype(float) > 0]
                if not categories or mdf2.empty:
                    ax.set_visible(False)
                    continue

                _plot_grouped_bars(
                    ax,
                    data=mdf2,
                    categories=categories,
                    category_col=category_col,
                    metric=metric,
                    opts=opts,
                    category_tick=category_tick,
                    log_y=log_y,
                    x_tick_rotation=x_tick_rotation,
                )

                # Only show the category labels along the bottom row.
                show_x = r == (len(models) - 1)
                ax.tick_params(axis="x", labelbottom=show_x)
                if not show_x:
                    ax.set_xlabel("")

                if r == 0:
                    ax.set_title(metric_label, fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"{MODEL_DISPLAY[model]}\nTail latency (ms)")
                else:
                    ax.set_ylabel("")

        # Single legend inside top-left panel (overlap OK)
        handles = [
            mpatches.Patch(color=OPT_COLOR.get(o, "#777777"), label=OPT_DISPLAY.get(o, o))
            for o in opts
        ]
        axes[0][0].legend(
            handles=handles,
            title="Optimization",
            loc="upper left",
            framealpha=0.92,
            edgecolor="#cccccc",
            fancybox=True,
            fontsize=7.6,
            title_fontsize=8.2,
            ncol=2,
        )

        fig.suptitle(title, fontsize=13, y=0.985)
        # Slightly smaller bottom margin since only bottom row has x tick labels.
        fig.subplots_adjust(left=0.06, right=0.995, bottom=0.12, top=0.88)
        _savefig(fig, out_dir, f"{out_name}{suffix}.png")

    # Workload sweep figures
    wdf = df[df["workload"].isin(WORKLOAD_ORDER)].copy()
    workloads = [w for w in WORKLOAD_ORDER if w in set(wdf["workload"].unique())]
    workload_ticks = [WORKLOAD_TICK.get(w, w) for w in workloads]

    _make_grid(
        slice_df=wdf,
        categories=workloads,
        category_col="workload",
        category_tick=workload_ticks,
        x_tick_rotation=25,
        metrics=metrics,
        out_name="tail_latency_workload_grid_2x2_best",
        title="Tail Latency by Workload @ 2 req/s",
    )

    # QPS sweep figures
    qdf = df[df["workload"] == "qps_sweep"].copy()
    qps_levels = sorted({float(v) for v in qdf["request_rate"].dropna().unique()})
    qps_ticks = [f"{q:g}" for q in qps_levels]

    _make_grid(
        slice_df=qdf,
        categories=qps_levels,
        category_col="request_rate",
        category_tick=qps_ticks,
        x_tick_rotation=0,
        metrics=metrics,
        out_name="tail_latency_qps_grid_2x2_best",
        title="Tail Latency by QPS @ 512-in / 128-out",
    )


# ── Improvement summary heatmap ─────────────────────────────────────────────

def plot_improvement_summary_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    name: str,
    title: str,
    agg: str = "median",
    metrics: list[tuple[str, str, str]] | None = None,
    key_cols: list[str] | None = None,
    opts_exclude: set[str] | None = None,
    annotate: bool = True,
) -> None:
    """Summary heatmap of speedup vs no-opts, aggregated across many dots.

    Key property: speedup is computed 1-to-1 against the baseline for the exact
    same (model, QPS, workload) point.

    Steps:
      1) Reduce repeats by aggregating baseline and opt values per key.
      2) Inner-join opt to baseline on the key.
      3) Compute speedup ratio per joined row.
      4) Aggregate ratios with median/mean, then map to log2 for coloring.

    Layout: one subplot per model. Rows = opt variants, cols = metrics.
    """

    if metrics is None:
        metrics = [
            ("p50_ttft_ms", "P50 TTFT (ms)", "lower"),
            ("p99_ttft_ms", "P99 TTFT (ms)", "lower"),
            ("p50_tpot_ms", "P50 TPOT (ms)", "lower"),
            ("p99_tpot_ms", "P99 TPOT (ms)", "lower"),
            ("p50_e2el_ms", "P50 E2EL (ms)", "lower"),
            ("p99_e2el_ms", "P99 E2EL (ms)", "lower"),
            ("output_throughput", "Output tok/s", "higher"),
        ]
    if key_cols is None:
        # Enough to uniquely identify a benchmark setting in this repo.
        key_cols = ["workload", "request_rate"]

    metric_cols = [m for m, _, _ in metrics]
    # Only coerce numeric metric columns (+ request_rate if used).
    # Never coerce string key columns like 'workload' (it would become all-NaNs).
    numeric_cols = metric_cols.copy()
    if "request_rate" in df.columns:
        numeric_cols.append("request_rate")
    df = _num(df, numeric_cols).copy()

    models = [m for m in MODEL_SIZES if m in df["model_size"].unique()]
    if not models:
        print("  [skip] improvement summary heatmap: no data")
        return

    agg = str(agg).lower().strip()
    if agg not in {"median", "mean"}:
        raise ValueError("agg must be 'median' or 'mean'")

    # Stable numeric key for request_rate joins.
    if "request_rate" in key_cols and "request_rate" in df.columns:
        df["_qps_key"] = df["request_rate"].astype(float).round(6)
        key_cols = ["_qps_key" if c == "request_rate" else c for c in key_cols]

    def _reduce_per_key(frame: pd.DataFrame) -> pd.DataFrame:
        g = frame.groupby(key_cols, as_index=False)[metric_cols]
        return (g.median(numeric_only=True) if agg == "median" else g.mean(numeric_only=True))

    if opts_exclude is None:
        opts_exclude = set()

    opts_present = _present_opts(df)
    if "no-opts" not in opts_present:
        print("  [skip] improvement summary heatmap: no 'no-opts' baseline")
        return
    opts = [o for o in OPT_ORDER if o in opts_present and o != "no-opts" and o not in opts_exclude]
    if not opts:
        print("  [skip] improvement summary heatmap: no optimization variants")
        return

    matrices: dict[str, np.ndarray] = {}
    for model in models:
        mdf = df[df["model_size"] == model].copy()
        base_raw = mdf[mdf["opt_label"] == "no-opts"].copy()
        if base_raw.empty:
            continue
        base = _reduce_per_key(base_raw)

        log_mat = np.full((len(opts), len(metrics)), np.nan)
        for r, opt in enumerate(opts):
            odf_raw = mdf[mdf["opt_label"] == opt].copy()
            if odf_raw.empty:
                continue
            odf = _reduce_per_key(odf_raw)
            joined = odf.merge(base, on=key_cols, how="inner", suffixes=("_opt", "_base"))
            if joined.empty:
                print(
                    f"  [note] improvement summary: no 1:1 matches for {model} / '{opt}' "
                    f"on keys {key_cols} (this usually means baseline is missing some points, "
                    f"or key values don't align)"
                )
                continue

            for c, (metric, _, direction) in enumerate(metrics):
                ocol = f"{metric}_opt" if f"{metric}_opt" in joined.columns else metric
                bcol = f"{metric}_base" if f"{metric}_base" in joined.columns else metric
                if ocol not in joined.columns or bcol not in joined.columns:
                    continue

                base_vals = joined[bcol].astype(float)
                opt_vals = joined[ocol].astype(float)
                valid = (
                    np.isfinite(base_vals)
                    & np.isfinite(opt_vals)
                    & (base_vals != 0)
                    & (opt_vals != 0)
                )
                if not valid.any():
                    continue

                ratio = (base_vals[valid] / opt_vals[valid]) if direction == "lower" else (opt_vals[valid] / base_vals[valid])
                ratio = ratio[ratio > 0]
                if ratio.empty:
                    continue

                # Aggregate ratios arithmetically, then map to log₂ for coloring.
                ratio_agg = float(np.median(ratio) if agg == "median" else np.mean(ratio))
                if not np.isfinite(ratio_agg) or ratio_agg <= 0:
                    continue
                log_mat[r, c] = float(np.log2(ratio_agg))

        matrices[model] = log_mat

    if not matrices:
        print("  [skip] improvement summary heatmap: no baseline-aligned data")
        return

    all_log = np.concatenate([m.flatten() for m in matrices.values()])
    finite = all_log[np.isfinite(all_log)]
    if finite.size == 0:
        print("  [skip] improvement summary heatmap: no finite speedups")
        return

    # Color axis is log2(ratio), but we want fixed ratio ticks.
    # Use a diverging norm centered at 1× (log2=0) even though the limits are asymmetric.
    vmin = float(np.log2(0.25))
    vmax = float(np.log2(2000.0))
    vmin = min(vmin, float(np.nanmin(finite)))
    vmax = max(vmax, float(np.nanmax(finite)))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    span = max(abs(vmin), abs(vmax))

    n_rows = len(opts)
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(max(10, len(metrics) * 1.35) * len(models) / 2 + 1.2, max(4, n_rows * 0.75)),
        squeeze=False,
        gridspec_kw={"wspace": 0.012},
        layout="constrained",
    )

    last_im = None
    for col, model in enumerate(models):
        ax = axes[0][col]
        mat = matrices.get(model)
        if mat is None:
            ax.set_title(f"{MODEL_DISPLAY.get(model, model)}\n(no baseline)")
            continue

        im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
        last_im = im

        # Cell borders
        ax.set_xticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(opts) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Numeric annotations (speedup in x)
        if annotate:
            ann_fs = 7.4 if len(metrics) <= 5 else 6.8
            for yi in range(mat.shape[0]):
                for xi in range(mat.shape[1]):
                    v = mat[yi, xi]
                    if not np.isfinite(v):
                        continue
                    ratio = 2 ** float(v)
                    txt = f"{ratio:.2f}x" if ratio < 10 else (f"{ratio:.1f}x" if ratio < 100 else f"{ratio:.0f}x")
                    color = "#ffffff" if abs(float(v)) > (0.55 * span) else "#111111"
                    ax.text(xi, yi, txt, ha="center", va="center", fontsize=ann_fs, color=color)

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m[1] for m in metrics], rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(opts)))
        if col == 0:
            ax.set_yticklabels([OPT_DISPLAY.get(o, o) for o in opts], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[0][-1], pad=0.02)
        cbar.set_label(f"Speedup vs no optimizations (aggregate {agg} ratio)", fontsize=9)

        # More (log-spaced) ratio ticks so the scale reads less "jumpy".
        ratios = [
            0.25, 0.5, 1.0,
            2.0, 5.0, 10.0, 25.0,
            50.0, 100.0,
            250.0, 500.0,
            1000.0, 2000.0,
        ]
        ticks = [float(np.log2(r)) for r in ratios]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{r:g}x" for r in ratios])
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=13)
    _savefig(fig, out_dir, name)


# ── Student vs teacher (3B vs 8B) ─────────────────────────────────────────

def plot_student_vs_teacher_speedup_heatmaps(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    student_model_size: str = "3B",
    teacher_model_size: str = "8B",
    name: str = "student_vs_teacher_speedup",
    agg: str = "median",
    annotate: bool = True,
    metrics: list[tuple[str, str, str]] | None = None,
) -> None:
    """Student vs teacher speedup summary (not split by slice).

    For each metric we define a ratio so that ratio > 1 means the student is better:
      - latency metrics (direction='lower'):     ratio = teacher / student
      - throughput metrics (direction='higher'): ratio = student / teacher

    We aggregate the ratio per optimization variant with mean/median (arithmetic),
    then color the heatmap using log₂(mean_ratio) so the colorbar behaves like a log-scaled axis.

    Produces:
      - {name}_heatmap.png
    """

    if metrics is None:
        metrics = [
            ("p50_ttft_ms", "P50 TTFT", "lower"),
            ("p99_ttft_ms", "P99 TTFT", "lower"),
            ("p50_tpot_ms", "P50 TPOT", "lower"),
            ("p99_tpot_ms", "P99 TPOT", "lower"),
            ("p50_e2el_ms", "P50 E2EL", "lower"),
            ("p99_e2el_ms", "P99 E2EL", "lower"),
            ("output_throughput", "Output throughput", "higher"),
        ]

    name = str(name)
    if name.lower().endswith(".png"):
        name = name[:-4]

    metric_cols = [m for m, _, _ in metrics]
    numeric_cols = metric_cols.copy()
    if "request_rate" in df.columns:
        numeric_cols.append("request_rate")
    df = _num(df, numeric_cols).copy()

    agg = str(agg).lower().strip()
    if agg not in {"median", "mean"}:
        raise ValueError("agg must be 'median' or 'mean'")

    # Stable numeric key for QPS joins.
    if "request_rate" in df.columns:
        df["_qps_key"] = df["request_rate"].astype(float).round(6)

    key_cols = ["opt_label", "workload", "_qps_key"] if "_qps_key" in df.columns else ["opt_label", "workload"]

    def _reduce_per_key(frame: pd.DataFrame) -> pd.DataFrame:
        g = frame.groupby(key_cols, as_index=False)[metric_cols]
        return (g.median(numeric_only=True) if agg == "median" else g.mean(numeric_only=True))

    student_raw = df[df["model_size"] == student_model_size].copy()
    teacher_raw = df[df["model_size"] == teacher_model_size].copy()
    if student_raw.empty or teacher_raw.empty:
        print("  [skip] student vs teacher heatmap: missing student/teacher data")
        return

    student = _reduce_per_key(student_raw)
    teacher = _reduce_per_key(teacher_raw)
    joined = student.merge(teacher, on=key_cols, how="inner", suffixes=("_student", "_teacher"))
    if joined.empty:
        print(f"  [skip] student vs teacher heatmap: no 1:1 matches on keys {key_cols}")
        return

    opts_present = list(joined["opt_label"].dropna().unique())
    # Keep canonical ordering, then append any unexpected labels.
    opts = [o for o in OPT_ORDER if o in opts_present]
    opts += [o for o in opts_present if o not in set(opts)]
    if not opts:
        print("  [skip] student vs teacher heatmap: no optimization variants")
        return

    mat = np.full((len(opts), len(metrics)), np.nan)
    for r, opt in enumerate(opts):
        odf = joined[joined["opt_label"] == opt]
        if odf.empty:
            continue

        for c, (metric, _, direction) in enumerate(metrics):
            s_col = f"{metric}_student" if f"{metric}_student" in odf.columns else metric
            t_col = f"{metric}_teacher" if f"{metric}_teacher" in odf.columns else metric
            if s_col not in odf.columns or t_col not in odf.columns:
                continue

            s_vals = odf[s_col].astype(float)
            t_vals = odf[t_col].astype(float)
            valid = (
                np.isfinite(s_vals)
                & np.isfinite(t_vals)
                & (s_vals != 0)
                & (t_vals != 0)
            )
            if not bool(np.any(valid)):
                continue

            ratio = (t_vals / s_vals) if direction == "lower" else (s_vals / t_vals)
            ratio = ratio[pd.Series(valid, index=ratio.index)]
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            if ratio.empty:
                continue

            # Aggregate ratios arithmetically, then map to log₂ for coloring.
            ratio_agg = float(np.median(ratio) if agg == "median" else np.mean(ratio))
            if not np.isfinite(ratio_agg) or ratio_agg <= 0:
                continue
            mat[r, c] = float(np.log2(ratio_agg))

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        print("  [skip] student vs teacher heatmap: no finite speedups")
        return

    # Color axis is log2(ratio). Use a diverging norm centered at 1× (log2=0)
    # with fixed ratio ticks (as requested).
    vmin = float(np.log2(0.25))
    vmax = float(np.log2(2.5))
    vmin = min(vmin, float(np.nanmin(finite)))
    vmax = max(vmax, float(np.nanmax(finite)))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    span = max(abs(vmin), abs(vmax))

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(max(9.0, 1.25 * len(metrics) + 5.0), max(4.8, 0.62 * len(opts) + 2.0)),
    )

    im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")

    # Cell borders
    ax.set_xticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(opts) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        ann_fs = 7.4 if len(metrics) <= 5 else 6.8
        for yi in range(mat.shape[0]):
            for xi in range(mat.shape[1]):
                v = mat[yi, xi]
                if not np.isfinite(v):
                    continue
                ratio = 2 ** float(v)
                txt = f"{ratio:.2f}x" if ratio < 10 else (f"{ratio:.1f}x" if ratio < 100 else f"{ratio:.0f}x")
                color = "#ffffff" if abs(float(v)) > (0.55 * span) else "#111111"
                ax.text(xi, yi, txt, ha="center", va="center", fontsize=ann_fs, color=color)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(opts)))
    ax.set_yticklabels([OPT_DISPLAY.get(o, o) for o in opts], fontsize=9)

    student_disp = MODEL_DISPLAY.get(student_model_size, student_model_size)
    teacher_disp = MODEL_DISPLAY.get(teacher_model_size, teacher_model_size)
    ax.set_title(f"Student ({student_disp}) vs Teacher ({teacher_disp})", fontsize=12, pad=10)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Speedup ratio: student vs teacher")

    ticks = [
        float(np.log2(0.25)),
        float(np.log2(0.5)),
        0.0,
        float(np.log2(1.5)),
        float(np.log2(2.5)),
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["0.25x", "0.5x", "1x", "1.5x", "2.5x"])

    _savefig(fig, out_dir, f"{name}_heatmap.png")


def plot_student_vs_teacher_speedup_bars(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    student_model_size: str = "3B",
    teacher_model_size: str = "8B",
    name: str = "student_vs_teacher_speedup",
    agg: str = "mean",
    annotate: bool = False,
    metrics: list[tuple[str, str, str]] | None = None,
    opts: list[str] | None = None,
) -> None:
    """Student vs teacher speedup, per-instance (bars).

    Uses the same ratio definition as the heatmap, but reports percent deltas:
      Δ% = (ratio - 1) * 100

    Produces:
      - {name}_workloads_2qps_bars.png
      - {name}_qps_sweep_bars.png
    """

    if metrics is None:
        metrics = [
            ("p99_ttft_ms", "P99 TTFT", "lower"),
            ("p99_tpot_ms", "P99 TPOT", "lower"),
            ("p99_e2el_ms", "P99 E2EL", "lower"),
            ("output_throughput", "Output throughput", "higher"),
        ]

    name = str(name)
    if name.lower().endswith(".png"):
        name = name[:-4]

    metric_cols = [m for m, _, _ in metrics]
    numeric_cols = metric_cols.copy()
    if "request_rate" in df.columns:
        numeric_cols.append("request_rate")
    df = _num(df, numeric_cols).copy()

    agg = str(agg).lower().strip()
    if agg not in {"median", "mean"}:
        raise ValueError("agg must be 'median' or 'mean'")

    if "request_rate" in df.columns:
        df["_qps_key"] = df["request_rate"].astype(float).round(6)

    key_cols = ["opt_label", "workload", "_qps_key"] if "_qps_key" in df.columns else ["opt_label", "workload"]

    def _reduce_per_key(frame: pd.DataFrame) -> pd.DataFrame:
        g = frame.groupby(key_cols, as_index=False)[metric_cols]
        return (g.median(numeric_only=True) if agg == "median" else g.mean(numeric_only=True))

    student_raw_all = df[df["model_size"] == student_model_size].copy()
    teacher_raw_all = df[df["model_size"] == teacher_model_size].copy()
    if student_raw_all.empty or teacher_raw_all.empty:
        print("  [skip] student vs teacher bars: missing student/teacher data")
        return

    if opts is None:
        s_opts = set(student_raw_all["opt_label"].dropna().unique())
        t_opts = set(teacher_raw_all["opt_label"].dropna().unique())
        both = [o for o in OPT_ORDER if o in (s_opts & t_opts)]
        opts = both

    if not opts:
        print("  [skip] student vs teacher bars: no shared optimization variants")
        return

    qps_target = 2.0

    def _slice_workloads_2qps(frame: pd.DataFrame) -> pd.DataFrame:
        sdf = frame[frame["workload"].isin(WORKLOAD_ORDER)].copy()
        if "request_rate" in sdf.columns:
            sdf = sdf[np.isclose(sdf["request_rate"].astype(float), qps_target, rtol=0, atol=1e-6)]
        return sdf

    def _slice_qps_sweep(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["workload"] == "qps_sweep"].copy()

    slices = [
        {
            "key": "workloads_2qps",
            "title": "Workloads @ 2 req/s",
            "student_df": _slice_workloads_2qps(student_raw_all),
            "teacher_df": _slice_workloads_2qps(teacher_raw_all),
            "instance_col": "workload",
            "categories": WORKLOAD_ORDER,
            "ticks": [WORKLOAD_TICK.get(w, w) for w in WORKLOAD_ORDER],
            "rotate": 25,
        },
        {
            "key": "qps_sweep",
            "title": "QPS sweep (512-in / 128-out)",
            "student_df": _slice_qps_sweep(student_raw_all),
            "teacher_df": _slice_qps_sweep(teacher_raw_all),
            "instance_col": "_qps_key" if "_qps_key" in df.columns else "request_rate",
            "categories": None,
            "ticks": None,
            "rotate": 0,
        },
    ]

    student_disp = MODEL_DISPLAY.get(student_model_size, student_model_size)
    teacher_disp = MODEL_DISPLAY.get(teacher_model_size, teacher_model_size)

    for s in slices:
        s_student = s["student_df"]
        s_teacher = s["teacher_df"]
        if s_student.empty or s_teacher.empty:
            continue

        if s["key"] == "workloads_2qps":
            present = set(s_student["workload"].dropna().unique()) | set(s_teacher["workload"].dropna().unique())
            cats = [w for w in WORKLOAD_ORDER if w in present]
            s["categories"] = cats
            s["ticks"] = [WORKLOAD_TICK.get(w, w) for w in cats]

        if s["categories"] is None:
            inst_col = s["instance_col"]
            if inst_col not in s_student.columns or inst_col not in s_teacher.columns:
                continue
            cats = sorted({float(v) for v in pd.concat([s_student[inst_col], s_teacher[inst_col]]).dropna().unique()})
            s["categories"] = cats
            s["ticks"] = [f"{c:g}" for c in cats]

        cats = list(s["categories"])
        ticks = list(s["ticks"]) if s["ticks"] is not None else [str(c) for c in cats]

        # Reduce and join for the slice.
        student = _reduce_per_key(s_student[s_student["opt_label"].isin(opts)].copy())
        teacher = _reduce_per_key(s_teacher[s_teacher["opt_label"].isin(opts)].copy())
        joined = student.merge(teacher, on=key_cols, how="inner", suffixes=("_student", "_teacher"))
        if joined.empty:
            continue

        # values[metric][opt][cat] = pct
        values: dict[str, dict[str, dict[object, float]]] = {m: {o: {} for o in opts} for m, _, _ in metrics}

        inst_col = s["instance_col"]
        inst_vals = joined[inst_col]
        if inst_col != "workload":
            inst_vals = inst_vals.astype(float).round(6)

        for metric, _, direction in metrics:
            s_col = f"{metric}_student" if f"{metric}_student" in joined.columns else metric
            t_col = f"{metric}_teacher" if f"{metric}_teacher" in joined.columns else metric
            if s_col not in joined.columns or t_col not in joined.columns:
                continue

            s_vals = joined[s_col].astype(float)
            t_vals = joined[t_col].astype(float)
            valid = (
                np.isfinite(s_vals)
                & np.isfinite(t_vals)
                & (s_vals != 0)
                & (t_vals != 0)
            )
            if not bool(np.any(valid)):
                continue

            ratio = (t_vals / s_vals) if direction == "lower" else (s_vals / t_vals)
            ratio = ratio[pd.Series(valid, index=ratio.index)]
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            if ratio.empty:
                continue

            pct = (ratio - 1.0) * 100.0
            inst_aligned = inst_vals.loc[pct.index]
            opt_aligned = joined.loc[pct.index, "opt_label"].astype(str)

            for o, cat, v in zip(opt_aligned.tolist(), inst_aligned.tolist(), pct.tolist()):
                if o in values[metric] and cat in set(cats):
                    values[metric][o][cat] = float(v)

        # y-lims per metric
        ylims: dict[str, float] = {}
        for metric, _, _ in metrics:
            all_v: list[float] = []
            for o in opts:
                all_v.extend(values[metric][o].values())
            arr = np.array(all_v, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                ylims[metric] = 5.0
                continue
            lim = float(np.nanpercentile(np.abs(arr), 97))
            lim = max(5.0, lim)
            lim = float(np.ceil(lim / 5.0) * 5.0)
            ylims[metric] = lim

        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=(14.0, max(7.0, 1.50 * len(metrics))),
            squeeze=False,
            gridspec_kw={"hspace": 0.18},
        )

        x = np.arange(len(cats), dtype=float)
        group_w = 0.86
        bw = group_w / max(1, len(opts))
        offsets = (np.arange(len(opts)) - (len(opts) - 1) / 2) * bw

        for r, (metric, metric_label, _) in enumerate(metrics):
            ax = axes[r][0]
            ax.axhline(0.0, color="#000000", linewidth=1.8, alpha=0.92, zorder=2)

            for oi, opt in enumerate(opts):
                y = [values[metric][opt].get(cat, np.nan) for cat in cats]

                for j, v in enumerate(y):
                    if not np.isfinite(v):
                        continue
                    ax.bar(
                        x[j] + offsets[oi],
                        float(v),
                        bw * 0.95,
                        color=OPT_COLOR.get(opt, "#777777"),
                        edgecolor="#1a1a1a",
                        linewidth=0.6,
                        alpha=0.95,
                        zorder=3,
                    )
                    if annotate:
                        ax.text(
                            x[j] + offsets[oi],
                            float(v) + (1.0 if float(v) >= 0 else -1.0),
                            f"{float(v):+.0f}%",
                            ha="center",
                            va="bottom" if float(v) >= 0 else "top",
                            fontsize=6.4,
                            color="#111111",
                            zorder=4,
                        )

            lim = ylims.get(metric, 5.0)
            ax.set_ylim(-lim, lim)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))

            ax.set_ylabel(metric_label, fontsize=9, labelpad=2)
            ax.yaxis.set_label_coords(-0.10, 0.5)

            show_x = (r == len(metrics) - 1)
            ax.set_xticks(x)
            ax.set_xticklabels(
                ticks if show_x else [],
                rotation=s["rotate"],
                ha="right" if s["rotate"] else "center",
                fontsize=8.8,
            )

        fig.suptitle(
            f"Student ({student_disp}) vs Teacher ({teacher_disp}) — {s['title']}",
            fontsize=13,
            y=0.995,
            x=0.10,
            ha="left",
        )

        fig.text(
            0.022,
            0.50,
            "Δ vs teacher (%), 0 = equal, + = student better (lower latency / higher throughput)",
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )

        opt_handles = [
            mpatches.Patch(facecolor=OPT_COLOR.get(o, "#777777"), edgecolor="#1a1a1a", label=OPT_DISPLAY.get(o, o))
            for o in opts
        ]

        # Leave a larger right margin so the legend stays out of the plot area.
        fig.tight_layout(rect=[0.03, 0, 0.62, 0.94])
        fig.legend(
            handles=opt_handles,
            loc="upper left",
            bbox_to_anchor=(0.64, 0.998),
            framealpha=0.92,
            edgecolor="#cccccc",
            fancybox=True,
            fontsize=6.6,
            labelspacing=0.18,
            handletextpad=0.4,
            borderpad=0.32,
            handlelength=1.05,
        )

        _savefig(fig, out_dir, f"{name}_{s['key']}_bars.png")


# ── Marginal effect (add-on vs base) ───────────────────────────────────────

def plot_marginal_effect_addons_workloads_vs_qps(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    name: str = "marginal_effect_addons",
    agg: str = "mean",
    annotate: bool = False,
    metrics: list[tuple[str, str, str]] | None = None,
    pairs: list[tuple[str, str]] | None = None,
) -> None:
    """Marginal effect of add-on optimizations vs their base variant (per instance).

    This is meant to answer: "what does adding +pc / +cp / +sd(...) do *on top of* the base?"

    For each (add_on, base) pair, and for each instance (workload@2qps or QPS level), we compute
    a signed percent delta where positive means "better" and 0 means "equal":
      - define a per-point ratio ("speedup") the same way as the improvement summary:
          * latency metrics (direction='lower'):  ratio = base / add_on
          * throughput metrics (direction='higher'): ratio = add_on / base
      - then report Δ% = (ratio - 1) * 100

    Repeats are reduced with mean/median per (workload, request_rate) before the delta is computed.

    Produces two bar-chart grids:
      - {name}_workloads_2qps_bars.png
      - {name}_qps_sweep_bars.png

    Bars are colored by add-on; y-axis is Δ vs base (%), with a bold 0 line.
    """

    if metrics is None:
        metrics = [
            ("p99_ttft_ms", "P99 TTFT", "lower"),
            ("p99_tpot_ms", "P99 TPOT", "lower"),
            ("p99_e2el_ms", "P99 E2EL", "lower"),
            ("output_throughput", "Output throughput", "higher"),
        ]

    if pairs is None:
        pairs = [
            ("cb+cp", "cb"),
            ("cb+pc", "cb"),
            ("cb+as", "cb"),
            ("all+q", "default"),
            ("all+sd(1B)", "default"),
            ("all+sd(1B+p)", "default"),
        ]

    name = str(name)
    if name.lower().endswith(".png"):
        name = name[:-4]

    metric_cols = [m for m, _, _ in metrics]
    numeric_cols = metric_cols.copy()
    if "request_rate" in df.columns:
        numeric_cols.append("request_rate")
    df = _num(df, numeric_cols).copy()

    models = [m for m in MODEL_SIZES if m in df["model_size"].unique()]
    if not models:
        print("  [skip] marginal effect (bars): no data")
        return

    agg = str(agg).lower().strip()
    if agg not in {"median", "mean"}:
        raise ValueError("agg must be 'median' or 'mean'")

    # Build a stable join key.
    if "request_rate" in df.columns:
        df["_qps_key"] = df["request_rate"].astype(float).round(6)
    key_cols = ["workload", "_qps_key"] if "_qps_key" in df.columns else ["workload"]

    def _reduce_per_key(frame: pd.DataFrame) -> pd.DataFrame:
        g = frame.groupby(key_cols, as_index=False)[metric_cols]
        return (g.median(numeric_only=True) if agg == "median" else g.mean(numeric_only=True))

    # Only keep pairs where both variants exist somewhere in the dataset.
    opts_present = set(df["opt_label"].unique())
    pairs = [p for p in pairs if p[0] in opts_present and p[1] in opts_present and p[0] != p[1]]
    if not pairs:
        print("  [skip] marginal effect (bars): no (add_on, base) pairs present")
        return

    def _addon_label(add_on: str, base: str) -> str:
        # Prefer a "what got added" label like "+cp" or "+sd(1B+p)".
        if add_on.startswith(base + "+"):
            return add_on[len(base):]
        if base == "default" and add_on.startswith("all+"):
            return "+" + add_on[len("all+"):]
        # Special-case: sd(1B+p) vs sd(1B) => "+p"
        if base.startswith("all+sd(") and add_on.startswith("all+sd("):
            if add_on.replace("+p)", ")") == base:
                return "+p"
        return add_on

    def _base_label(base: str) -> str:
        # Shorten bases like all+sd(1B) -> +sd(1B) for display.
        if base == "default":
            return "default"
        if base.startswith("all+"):
            return "+" + base[len("all+"):]
        return base

    pair_labels: list[str] = []
    for add_on, base in pairs:
        addon = _addon_label(add_on, base)
        base_disp = _base_label(base)
        if addon.startswith("+"):
            pair_labels.append(f"{addon} ({base_disp})")
        else:
            pair_labels.append(f"{add_on} vs {base_disp}")

    pair_colors = {i: OPT_COLOR.get(pairs[i][0], "#1f77b4") for i in range(len(pairs))}

    qps_target = 2.0

    def _slice_workloads_2qps(frame: pd.DataFrame) -> pd.DataFrame:
        sdf = frame[frame["workload"].isin(WORKLOAD_ORDER)].copy()
        if "request_rate" in sdf.columns:
            sdf = sdf[np.isclose(sdf["request_rate"].astype(float), qps_target, rtol=0, atol=1e-6)]
        return sdf

    def _slice_qps_sweep(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["workload"] == "qps_sweep"].copy()

    work_sdf = _slice_workloads_2qps(df)
    qps_sdf = _slice_qps_sweep(df)

    work_present = set(work_sdf["workload"].unique()) if not work_sdf.empty else set()

    slices = [
        {
            "key": "workloads_2qps",
            "title": "Workloads @ 2 req/s",
            "df": work_sdf,
            "instance_col": "workload",
            "categories": [w for w in WORKLOAD_ORDER if w in work_present],
            "ticks": [WORKLOAD_TICK.get(w, w) for w in WORKLOAD_ORDER if w in work_present],
            "rotate": 25,
        },
        {
            "key": "qps_sweep",
            "title": "QPS sweep (512-in / 128-out)",
            "df": qps_sdf,
            "instance_col": "_qps_key" if "_qps_key" in qps_sdf.columns else "request_rate",
            "categories": None,
            "ticks": None,
            "rotate": 0,
        },
    ]

    for s in slices:
        sdf = s["df"]
        if sdf.empty:
            continue

        if s["categories"] is None:
            if s["instance_col"] not in sdf.columns:
                continue
            cats = sorted({float(v) for v in sdf[s["instance_col"]].dropna().unique()})
            s["categories"] = cats
            s["ticks"] = [f"{c:g}" for c in cats]

        # Precompute deltas: metric -> model -> pair_idx -> {category -> value}
        values: dict[str, dict[str, dict[int, dict[object, float]]]] = {}
        for metric, _, direction in metrics:
            values[metric] = {m: {pi: {} for pi in range(len(pairs))} for m in models}

        for model in models:
            mdf = sdf[sdf["model_size"] == model].copy()
            if mdf.empty:
                continue

            for pi, (add_on, base) in enumerate(pairs):
                add_raw = mdf[mdf["opt_label"] == add_on].copy()
                base_raw = mdf[mdf["opt_label"] == base].copy()
                if add_raw.empty or base_raw.empty:
                    continue

                add_df = _reduce_per_key(add_raw)
                base_df = _reduce_per_key(base_raw)
                joined = add_df.merge(base_df, on=key_cols, how="inner", suffixes=("_add", "_base"))
                if joined.empty:
                    continue

                inst_col = s["instance_col"]
                if inst_col not in joined.columns:
                    continue

                inst_vals = joined[inst_col]
                if inst_col != "workload":
                    inst_vals = inst_vals.astype(float).round(6)

                for metric, _, direction in metrics:
                    add_col = f"{metric}_add" if f"{metric}_add" in joined.columns else metric
                    base_col = f"{metric}_base" if f"{metric}_base" in joined.columns else metric
                    if add_col not in joined.columns or base_col not in joined.columns:
                        continue

                    add_vals = joined[add_col].astype(float)
                    base_vals = joined[base_col].astype(float)

                    valid = (
                        np.isfinite(add_vals)
                        & np.isfinite(base_vals)
                        & (add_vals != 0)
                        & (base_vals != 0)
                    )
                    if not bool(np.any(valid)):
                        continue

                    # Ratio is defined so that ratio > 1 means the add-on helps.
                    ratio = (base_vals / add_vals) if direction == "lower" else (add_vals / base_vals)
                    ratio = ratio[pd.Series(valid, index=ratio.index)]
                    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                    if ratio.empty:
                        continue

                    # Center at 0 (equal): Δ% = (ratio - 1) * 100.
                    pct = (ratio - 1.0) * 100.0
                    inst_aligned = inst_vals.loc[pct.index]

                    for cat, v in zip(inst_aligned.tolist(), pct.tolist()):
                        values[metric][model][pi][cat] = float(v)

        # Determine y-limits per metric row (symmetric around 0)
        ylims: dict[str, float] = {}
        for metric, _, _ in metrics:
            all_v = []
            for model in models:
                for pi in range(len(pairs)):
                    all_v.extend(values[metric][model][pi].values())
            all_v = np.array(all_v, dtype=float)
            all_v = all_v[np.isfinite(all_v)]
            if all_v.size == 0:
                ylims[metric] = 1.0
                continue
            lim = float(np.nanpercentile(np.abs(all_v), 97))
            lim = max(5.0, lim)
            # round up to a nice multiple
            lim = float(np.ceil(lim / 5.0) * 5.0)
            ylims[metric] = lim

        # Plot: rows=metrics, cols=models; x=instances; grouped bars = add-ons
        fig, axes = plt.subplots(
            len(metrics),
            len(models),
            figsize=(9.0 * len(models), max(7.5, 1.55 * len(metrics))),
            squeeze=False,
            gridspec_kw={"wspace": 0.08, "hspace": 0.18},
        )

        cats = list(s["categories"])
        ticks = list(s["ticks"]) if s["ticks"] is not None else [str(c) for c in cats]
        x = np.arange(len(cats), dtype=float)
        group_w = 0.82
        bw = group_w / max(1, len(pairs))
        offsets = (np.arange(len(pairs)) - (len(pairs) - 1) / 2) * bw

        for r, (metric, metric_label, _) in enumerate(metrics):
            for c, model in enumerate(models):
                ax = axes[r][c]
                ax.axhline(0.0, color="#000000", linewidth=1.8, alpha=0.92, zorder=2)

                for pi in range(len(pairs)):
                    y = []
                    for cat in cats:
                        y.append(values[metric][model][pi].get(cat, np.nan))

                    # Draw each bar separately; sign is conveyed by above/below the 0 line.
                    for j, v in enumerate(y):
                        if not np.isfinite(v):
                            continue
                        ax.bar(
                            x[j] + offsets[pi],
                            float(v),
                            bw * 0.95,
                            color=pair_colors[pi],
                            edgecolor="#1a1a1a",
                            linewidth=0.7,
                            alpha=0.95,
                            zorder=3,
                        )
                        if annotate:
                            ax.text(
                                x[j] + offsets[pi],
                                float(v) + (1.0 if float(v) >= 0 else -1.0),
                                f"{float(v):+.0f}%",
                                ha="center",
                                va="bottom" if float(v) >= 0 else "top",
                                fontsize=6.6,
                                color="#111111",
                                zorder=4,
                            )

                lim = ylims.get(metric, 5.0)
                ax.set_ylim(-lim, lim)

                # Labels
                if r == 0:
                    ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11)
                if c == 0:
                    ax.set_ylabel(metric_label, fontsize=9, labelpad=2)
                    # Pull the per-row metric labels slightly left to reduce the gap to the figure-level y label.
                    ax.yaxis.set_label_coords(-0.10, 0.5)

                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))

                show_x = (r == len(metrics) - 1)
                ax.set_xticks(x)
                ax.set_xticklabels(ticks if show_x else [], rotation=s["rotate"], ha="right" if s["rotate"] else "center", fontsize=8.6)

        # Legends (outside axes)
        addon_handles = [
            mpatches.Patch(facecolor=pair_colors[i], edgecolor="#1a1a1a", label=pair_labels[i])
            for i in range(len(pairs))
        ]

        fig.suptitle(
            f"Marginal Effect of Add-on Optimizations — {s['title']}\n"
            f"{agg} reduction per (workload,QPS); positive=better",
            fontsize=13,
            y=0.995,
        )

        # Figure-level y-axis meaning (applies to all subplots).
        # Keep this very close to the per-row metric labels to avoid a large dead zone on the left.
        fig.text(
            0.065,
            0.50,
            "Δ vs base (%), 0 = equal, + = better (lower latency / higher throughput)",
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )

        fig.tight_layout(rect=[0.03, 0, 0.78, 0.94])
        fig.legend(
            handles=addon_handles,
            title="Add-on (base)",
            loc="upper left",
            bbox_to_anchor=(0.80, 0.998),
            framealpha=0.92,
            edgecolor="#cccccc",
            fancybox=True,
            fontsize=8.2,
            title_fontsize=9,
        )

        out_name = f"{name}_{s['key']}_bars.png"
        _savefig(fig, out_dir, out_name)


# ── Efficiency frontier ─────────────────────────────────────────────────────

def _qps_marker_map(qps_levels: list[float]) -> dict[float, str]:
    """Stable marker assignment for QPS levels (per figure)."""
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "H"]
    qps_levels = [float(v) for v in qps_levels if np.isfinite(v)]
    qps_levels = sorted(set(qps_levels))
    return {qps: markers[i % len(markers)] for i, qps in enumerate(qps_levels)}


def plot_efficiency_frontier_workloads_2qps(
    df: pd.DataFrame,
    out_dir: Path,
    suffix: str = "",
    qps: float = 2.0,
    *,
    log_y: bool = False,
) -> None:
    """Efficiency frontier for the four input/output workloads at a fixed QPS."""
    df = _num(df, ["output_throughput", "p99_e2el_ms", "request_rate"])
    df = df[df["workload"].isin(WORKLOAD_ORDER)].copy()
    # Enforce the fixed-QPS slice (tolerant to float formatting).
    if "request_rate" in df.columns:
        df = df[np.isclose(df["request_rate"].astype(float), float(qps), rtol=0, atol=1e-6)]
    if log_y:
        df = df[df["p99_e2el_ms"].astype(float) > 0]

    models = [m for m in MODEL_SIZES if m in df["model_size"].unique()]
    if not models:
        print("  [skip] efficiency_frontier_workloads_2qps: no data")
        return

    fig, axes = plt.subplots(1, len(models), figsize=(9 * len(models), 6), squeeze=False)

    # For a clean slide-ready figure, keep legends outside the axes.
    # Collect legend items once per figure.
    opts_all = _present_opts(df)
    opt_handles = _opt_legend_handles(opts_all)
    wls_all = [w for w in WORKLOAD_ORDER if w in set(df["workload"].unique())]
    wl_handles = [
        Line2D([0], [0], marker=WORKLOAD_MARKERS[w], color="#555555",
               ms=7, linestyle="None", label=WORKLOAD_DISPLAY.get(w, w))
        for w in wls_all
    ]

    for col, model in enumerate(models):
        ax = axes[0][col]
        mdf = df[df["model_size"] == model].dropna(subset=["output_throughput", "p99_e2el_ms"])
        opts = _present_opts(mdf)
        wls_present = [w for w in WORKLOAD_MARKERS if w in mdf["workload"].unique() and w != "qps_sweep"]

        for opt in opts:
            odf = mdf[mdf["opt_label"] == opt].sort_values("output_throughput")
            if odf.empty:
                continue
            c = OPT_COLOR[opt]

            for wl in wls_present:
                wdf = odf[odf["workload"] == wl]
                if wdf.empty:
                    continue
                marker = WORKLOAD_MARKERS.get(wl, "o")
                ax.scatter(
                    wdf["output_throughput"].values,
                    wdf["p99_e2el_ms"].values,
                    color=c,
                    marker=marker,
                    s=95,
                    zorder=4,
                    edgecolors="#1a1a1a",
                    linewidths=0.4,
                    alpha=0.92,
                )

        ax.set_xlabel("Output Throughput (tokens/s)  →  higher is better")
        ax.set_ylabel("← lower is better  ·  P99 E2EL (ms)")
        ax.set_title(MODEL_DISPLAY[model], fontsize=11)

        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))

        # No per-axis legend (figure-level legends are added below).

    fig.suptitle(
        f"Throughput vs Tail Latency (workloads @ {qps:g} req/s)\n",
        fontsize=13,
        y=1.01,
    )

    # Leave a right margin for legends.
    fig.tight_layout(rect=[0, 0, 0.78, 0.95])
    leg1 = fig.legend(
        handles=opt_handles,
        title="Optimization variant",
        loc="upper left",
        bbox_to_anchor=(0.80, 0.93),
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=True,
        fontsize=8.5,
        title_fontsize=9,
    )
    fig.legend(
        handles=wl_handles,
        title="Workload (shape)",
        loc="upper left",
        bbox_to_anchor=(0.80, 0.48),
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=True,
        fontsize=8.5,
        title_fontsize=9,
    )
    name = f"efficiency_frontier_workloads_2qps{'_logy' if log_y else ''}{suffix}.png"
    _savefig(fig, out_dir, name)


def plot_efficiency_frontier_qps_sweep(
    df: pd.DataFrame,
    out_dir: Path,
    suffix: str = "",
    *,
    log_y: bool = False,
    warn_if_missing_no_opts: bool = True,
) -> None:
    """Efficiency frontier for the 512-in/128-out workload across QPS levels."""
    df = _num(df, ["output_throughput", "p99_e2el_ms", "request_rate"])
    df = df[df["workload"] == "qps_sweep"].copy()

    models = [m for m in MODEL_SIZES if m in df["model_size"].unique()]
    if not models:
        print("  [skip] efficiency_frontier_qps_sweep: no data")
        return

    fig, axes = plt.subplots(1, len(models), figsize=(9 * len(models), 6), squeeze=False)

    # Figure-level legends (outside axes).
    opts_all = _present_opts(df)
    opt_handles = _opt_legend_handles(opts_all)

    # One mapping across the whole figure so shapes mean the same QPS in both panels.
    all_qps_levels = sorted({float(v) for v in df["request_rate"].dropna().unique()})
    marker_map = _qps_marker_map(all_qps_levels)

    for col, model in enumerate(models):
        ax = axes[0][col]
        mdf = df[df["model_size"] == model].dropna(subset=["output_throughput", "p99_e2el_ms", "request_rate"])
        if log_y:
            # Log scale requires positive values.
            mdf = mdf[mdf["p99_e2el_ms"].astype(float) > 0]
        opts = _present_opts(mdf)
        if mdf.empty:
            ax.set_title(f"{MODEL_DISPLAY[model]}\n(no positive latency values)")
            continue

        if warn_if_missing_no_opts and "no-opts" not in set(opts):
            # If you're generating an "excl_no_opts" figure, this is expected.
            # Otherwise, it likely means the no-opts run is missing or got
            # dropped by filtering (e.g., NaNs in output_throughput/p99_e2el_ms).
            print(
                f"  [warn] efficiency_frontier_qps_sweep: 'no-opts' not present after filtering "
                f"for model={model} (suffix='{suffix}')"
            )

        # Draw baseline last so it doesn't get visually covered by other variants.
        if "no-opts" in opts:
            opts = [o for o in opts if o != "no-opts"] + ["no-opts"]

        for opt in opts:
            odf = mdf[mdf["opt_label"] == opt]
            if odf.empty:
                continue
            c = OPT_COLOR[opt]

            # Workload is constant here; use marker shape to encode QPS.
            for qps, qdf in odf.groupby("request_rate"):
                marker = marker_map.get(float(qps), "o")
                is_baseline = (opt == "no-opts")
                ax.scatter(
                    qdf["output_throughput"].values,
                    qdf["p99_e2el_ms"].values,
                    color=c,
                    marker=marker,
                    s=140 if is_baseline else 95,
                    zorder=7 if is_baseline else 4,
                    edgecolors="#000000" if is_baseline else "#1a1a1a",
                    linewidths=1.2 if is_baseline else 0.4,
                    alpha=1.0 if is_baseline else 0.88,
                )

        ax.set_xlabel("Output Throughput (tokens/s)  →  higher is better")
        ax.set_ylabel("← lower is better  ·  P99 E2EL (ms)")
        ax.set_title(MODEL_DISPLAY[model], fontsize=11)

        if log_y:
            ax.set_yscale("log")
            # Make log ticks readable on slides.
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))

        # No per-axis legend (figure-level legends are added below).

    fig.suptitle(
        ("Throughput vs Tail Latency (QPS @ 512-in / 128-out)\n"),
        fontsize=13,
        y=1.01,
    )

    # QPS-shape legend handles
    qps_handles = [
        Line2D([0], [0], marker=marker_map.get(float(qps), "o"), color="#555555",
               ms=7, linestyle="None", label=f"{float(qps):g} req/s")
        for qps in all_qps_levels
    ]

    # Leave a right margin for legends.
    fig.tight_layout(rect=[0, 0, 0.78, 0.95])
    fig.legend(
        handles=opt_handles,
        title="Optimization variant",
        loc="upper left",
        bbox_to_anchor=(0.80, 0.93),
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=True,
        fontsize=8.5,
        title_fontsize=9,
    )
    fig.legend(
        handles=qps_handles,
        title="QPS (shape)",
        loc="upper left",
        bbox_to_anchor=(0.80, 0.48),
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=True,
        fontsize=8.5,
        title_fontsize=9,
    )
    name = f"efficiency_frontier_qps_sweep{'_logy' if log_y else ''}{suffix}.png"
    _savefig(fig, out_dir, name)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the minimal set of benchmark figures used in this repo"
    )
    parser.add_argument(
        "--bench-dir", default="bench/bench_results",
        help="Root of the bench_results tree",
    )
    parser.add_argument(
        "--out-dir", default="figures",
        help="Output directory for figures (relative paths are resolved from repo root)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    print(f"Loading results from {bench_dir} ...")
    df = load_bench_results(bench_dir)
    if df.empty:
        raise SystemExit("ERROR: no results found. Check --bench-dir.")

    print(f"Generating figures in {out_dir} ...")

    # Tail latency grids (2×2 best)
    plot_tail_latency_grids(df, out_dir, log_y=True)

    # Improvement summary heatmap (all points)
    plot_improvement_summary_heatmap(
        df,
        out_dir,
        name="improvement_summary_heatmap_all.png",
        title="Improvement Summary",
        agg="median",
        key_cols=["workload", "request_rate"],
        opts_exclude={"all+sd(3B)"},
        annotate=True,
    )

    # Marginal cost/benefit of add-ons vs their base variant
    plot_marginal_effect_addons_workloads_vs_qps(df, out_dir)

    # Marginal effect (extra metrics): include P50 + P99 rows as well.
    plot_marginal_effect_addons_workloads_vs_qps(
        df,
        out_dir,
        name="marginal_effect_addons_p50p99",
        metrics=[
            ("p50_ttft_ms", "P50 TTFT", "lower"),
            ("p99_ttft_ms", "P99 TTFT", "lower"),
            ("p50_tpot_ms", "P50 TPOT", "lower"),
            ("p99_tpot_ms", "P99 TPOT", "lower"),
            ("p50_e2el_ms", "P50 E2EL", "lower"),
            ("p99_e2el_ms", "P99 E2EL", "lower"),
            ("output_throughput", "Output throughput", "higher"),
        ],
    )

    # Student vs teacher (3B student vs 8B teacher)
    plot_student_vs_teacher_speedup_heatmaps(df, out_dir)
    plot_student_vs_teacher_speedup_bars(df, out_dir)

    # Efficiency frontiers (log y)
    plot_efficiency_frontier_workloads_2qps(df, out_dir, log_y=True)
    plot_efficiency_frontier_qps_sweep(df, out_dir, log_y=True)

    print("Done.")


if __name__ == "__main__":
    main()
