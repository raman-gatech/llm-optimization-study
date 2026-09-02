"""Shared utilities for loading and processing vLLM benchmark result JSONs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# All latency metric columns present in result JSONs
LATENCY_COLS = [
    "mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p50_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms",  "p50_itl_ms",  "p95_itl_ms",  "p99_itl_ms",
    "mean_e2el_ms", "p50_e2el_ms", "p95_e2el_ms", "p99_e2el_ms",
]
THROUGHPUT_COLS = ["request_throughput", "output_throughput", "total_token_throughput"]


def load_results_dir(result_dir: str | Path, stage_override: str | None = None) -> pd.DataFrame:
    """Load all JSON result files from a single directory into a DataFrame."""
    records = []
    for p in Path(result_dir).glob("*.json"):
        with open(p) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: could not parse {p}")
                continue
        data["_file"] = p.name
        if stage_override and "stage" not in data:
            data["stage"] = stage_override
        records.append(data)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_results_tree(base_dir: str | Path) -> pd.DataFrame:
    """
    Recursively load all JSON results under base_dir.
    Adds '_subdir' column from the immediate parent directory name
    (e.g. 'baseline', 'apc', 'quantization').
    """
    frames = []
    for json_path in Path(base_dir).rglob("*.json"):
        with open(json_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: could not parse {json_path}")
                continue
        data["_file"] = json_path.name
        data["_subdir"] = json_path.parent.name
        # If no stage field, infer from subdir
        if "stage" not in data:
            data["stage"] = json_path.parent.name
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


def infer_model_size(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'model_size' column ('3B' or '8B') based on model_id."""
    def _size(model_id):
        if pd.isna(model_id):
            return "unknown"
        s = str(model_id)
        if "3B" in s or "3b" in s:
            return "3B"
        if "7B" in s or "7b" in s or "8B" in s or "8b" in s:
            return "8B"
        return s.split("/")[-1]
    df = df.copy()
    df["model_size"] = df["model_id"].apply(_size)
    return df


def get_qps_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are part of a QPS sweep (have numeric request_rate)."""
    df = df.copy()
    df["request_rate"] = pd.to_numeric(df["request_rate"], errors="coerce")
    # QPS sweep rows: workload contains 'random' or 'qps', or run field present
    mask = (
        df["workload"].str.contains("random|qps", case=False, na=False)
        | df.get("run", pd.Series(dtype=str)).notna()
    )
    return df[mask].dropna(subset=["request_rate"]).sort_values("request_rate")


def get_workload_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with named workloads (short_short, long_long, etc.)."""
    named = ["long_short", "short_short", "long_long", "short_long"]
    return df[df["workload"].isin(named)].copy()


# ── New bench_results tree loader ─────────────────────────────────────────────

# Maps (input_len, output_len) → semantic workload name
_WORKLOAD_MAP: dict[tuple[int, int], str] = {
    (128, 128): "short_short",
    (768,  32): "long_short",
    (128, 512): "short_long",
    (768, 192): "long_long",
    (512, 128): "qps_sweep",
}

# Maps full benchmark directory name → short display label
BENCHMARK_LABELS: dict[str, str] = {
    "vllm-0.18.0-no-opts":     "no-opts",
    "vllm-0.18.0-cb":          "cb",
    "vllm-0.18.0-cb+cp":       "cb+cp",
    "vllm-0.18.0-cb+pc":       "cb+pc",
    "vllm-0.18.0-cb+as":       "cb+as",
    "vllm-0.18.0-default":     "default",
    "vllm-0.18.0-all+q":       "all+q",
    "vllm-0.18.0-all+sd(1B)":  "all+sd(1B)",
    "vllm-0.18.0-all+sd(1B+p)":"all+sd(1B+p)",
    "vllm-0.18.0-all+sd(3B)":  "all+sd(3B)",
}

# Canonical display order for optimization labels
OPT_ORDER = [
    "no-opts", "cb", "cb+cp", "cb+pc", "cb+as", "default",
    "all+q", "all+sd(1B)", "all+sd(1B+p)", "all+sd(3B)",
]

# Human-readable workload tick labels
WORKLOAD_TICK = {
    "short_short": "128 input → 128 output",
    "long_short":  "768 input → 32 output",
    "short_long":  "128 input → 512 output",
    "long_long":   "768 input → 192 output",
}

_FNAME_RE = re.compile(
    r"^(?P<config>[^-]+)-(?P<dataset>[^-]+)"
    r"-(?P<in>\d+)x(?P<out>\d+)"
    r"-(?P<qps>[\d.]+)qps"
    r"-(?P<np>\d+)p\.json$"
)


def load_bench_results(base_dir: str | Path) -> pd.DataFrame:
    """
    Load all results from the bench_results directory tree:
      {base_dir}/{model}/{gpu}/{benchmark_name}/result/*.json

    Adds columns: model_dir, gpu, benchmark_name, opt_label, model_size,
                  input_len, output_len, workload.
    """
    records = []
    base = Path(base_dir)
    for json_path in base.rglob("result/*.json"):
        m = _FNAME_RE.match(json_path.name)
        if not m:
            continue
        bench_dir  = json_path.parent.parent   # .../benchmark_name/
        gpu_dir    = bench_dir.parent           # .../gpu/
        model_dir  = gpu_dir.parent             # .../model/

        try:
            with open(json_path) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: could not parse {json_path}")
            continue

        input_len      = int(m.group("in"))
        output_len     = int(m.group("out"))
        benchmark_name = bench_dir.name

        data["_file"]          = json_path.name
        data["model_dir"]      = model_dir.name
        data["gpu"]            = gpu_dir.name
        data["benchmark_name"] = benchmark_name
        data["opt_label"]      = BENCHMARK_LABELS.get(benchmark_name, benchmark_name)
        data["input_len"]      = input_len
        data["output_len"]     = output_len
        data["workload"]       = _WORKLOAD_MAP.get((input_len, output_len),
                                                   f"{input_len}x{output_len}")
        data["request_rate"]   = float(m.group("qps"))

        mdir = model_dir.name
        if "3B" in mdir or "3b" in mdir:
            data["model_size"] = "3B"
        elif "8B" in mdir or "8b" in mdir or "7B" in mdir or "7b" in mdir:
            data["model_size"] = "8B"
        else:
            data["model_size"] = mdir

        records.append(data)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def best_run_per_qps(df: pd.DataFrame, metric: str = "p99_e2el_ms") -> pd.DataFrame:
    """
    When multiple runs exist at the same QPS level, keep the one with the
    lowest median value of `metric` (most representative run).
    """
    if df.empty or metric not in df.columns:
        return df
    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return (
        df.sort_values(metric)
        .groupby(["model_size", "request_rate"], as_index=False)
        .first()
    )
