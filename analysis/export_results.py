"""Export durable, resume-oriented summaries from the checked-in benchmark corpus.

Outputs (under ``results/`` by default):

* ``benchmark_results.csv``: one normalized row per vLLM result JSON.
* ``gpu_telemetry_summary.csv``: one aggregate row per matching dmon trace.
* ``summary.json``: experiment scale, peak values, paired medians, and caveats.
* ``CHECKSUMS.sha256``: integrity hashes for the three generated data files.

The exporter uses only the Python standard library so it can run before the
plotting environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FILENAME_RE = re.compile(
    r"^(?P<config>[^-]+)-(?P<dataset>[^-]+)"
    r"-(?P<input_len>\d+)x(?P<output_len>\d+)"
    r"-(?P<qps>[\d.]+)qps"
    r"-(?P<num_prompts>\d+)p\.json$"
)

BENCHMARK_PREFIX = "vllm-0.18.0-"
MODEL_ORDER = ["Llama-3.2-3B", "Llama-3.1-8B"]
VARIANT_ORDER = [
    "no-opts",
    "cb",
    "cb+cp",
    "cb+pc",
    "cb+as",
    "default",
    "all+q",
    "all+sd(1B)",
    "all+sd(1B+p)",
    "all+sd(3B)",
]

LATENCY_FIELDS = [
    f"{stat}_{metric}_ms"
    for metric in ("ttft", "tpot", "itl", "e2el")
    for stat in ("mean", "median", "std", "p50", "p95", "p99")
]

RESULT_FIELDS = [
    "source_file",
    "date",
    "model",
    "model_size",
    "gpu",
    "benchmark_name",
    "variant",
    "config",
    "dataset",
    "input_len",
    "output_len",
    "offered_qps",
    "num_prompts",
    "completed",
    "failed",
    "duration_s",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "total_token_throughput",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
] + LATENCY_FIELDS

DMON_FIELDS = [
    "gpu_id",
    "power_w",
    "gpu_temp_c",
    "memory_temp_c",
    "sm_util_pct",
    "memory_util_pct",
    "encoder_util_pct",
    "decoder_util_pct",
    "jpeg_util_pct",
    "ofa_util_pct",
    "memory_clock_mhz",
    "processor_clock_mhz",
    "power_violation_pct",
    "thermal_violation",
    "framebuffer_mb",
    "bar1_mb",
    "confidential_compute_mb",
    "single_bit_ecc_errors",
    "double_bit_ecc_errors",
    "pci_errors",
    "pcie_rx_mb_s",
    "pcie_tx_mb_s",
]


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def median(values: Iterable[float]) -> float:
    return float(statistics.median(values))


def mean(values: Iterable[float]) -> float:
    return float(statistics.fmean(values))


def round_floats(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, digits) for item in value]
    return value


def load_results(bench_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(bench_dir.glob("*/*/*/result/*.json")):
        match = FILENAME_RE.match(path.name)
        if match is None:
            continue

        benchmark_dir = path.parent.parent
        gpu_dir = benchmark_dir.parent
        model_dir = gpu_dir.parent
        benchmark_name = benchmark_dir.name
        variant = (
            benchmark_name[len(BENCHMARK_PREFIX) :]
            if benchmark_name.startswith(BENCHMARK_PREFIX)
            else benchmark_name
        )
        raw = json.loads(path.read_text(encoding="utf-8"))

        row: dict[str, Any] = {
            "source_file": str(path),
            "date": raw.get("date"),
            "model": model_dir.name,
            "model_size": "3B" if "3B" in model_dir.name else "8B",
            "gpu": gpu_dir.name,
            "benchmark_name": benchmark_name,
            "variant": variant,
            "config": match.group("config"),
            "dataset": match.group("dataset"),
            "input_len": int(match.group("input_len")),
            "output_len": int(match.group("output_len")),
            "offered_qps": float(match.group("qps")),
            "num_prompts": int(match.group("num_prompts")),
            "completed": raw.get("completed"),
            "failed": raw.get("failed"),
            "duration_s": raw.get("duration"),
            "total_input_tokens": raw.get("total_input_tokens"),
            "total_output_tokens": raw.get("total_output_tokens"),
            "request_throughput": raw.get("request_throughput"),
            "request_goodput": raw.get("request_goodput"),
            "output_throughput": raw.get("output_throughput"),
            "total_token_throughput": raw.get("total_token_throughput"),
            "max_output_tokens_per_s": raw.get("max_output_tokens_per_s"),
            "max_concurrent_requests": raw.get("max_concurrent_requests"),
            "_benchmark_dir": benchmark_dir,
            "_result_stem": path.stem,
        }
        for field in LATENCY_FIELDS:
            row[field] = raw.get(field)
        rows.append(row)

    return rows


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in RESULT_FIELDS}


def result_key(row: dict[str, Any]) -> tuple[str, str, int, int, float]:
    return (
        str(row["model"]),
        str(row["variant"]),
        int(row["input_len"]),
        int(row["output_len"]),
        float(row["offered_qps"]),
    )


def context(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": row["model"],
        "variant": row["variant"],
        "input_len": row["input_len"],
        "output_len": row["output_len"],
        "offered_qps": row["offered_qps"],
        "source_file": row["source_file"],
    }


def peak(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    row = max(rows, key=lambda item: float(item[field]))
    return {"value": row[field], **context(row)}


def paired_speedup_summary(
    rows: list[dict[str, Any]],
    baseline_variant: str,
) -> dict[str, Any]:
    index = {result_key(row): row for row in rows}
    metrics = {
        "p50_ttft_ms": "lower",
        "p99_ttft_ms": "lower",
        "p50_tpot_ms": "lower",
        "p99_tpot_ms": "lower",
        "p50_e2el_ms": "lower",
        "p99_e2el_ms": "lower",
        "output_throughput": "higher",
    }
    output: dict[str, Any] = {}

    for model in MODEL_ORDER:
        output[model] = {}
        present_variants = [
            variant
            for variant in VARIANT_ORDER
            if any(row["model"] == model and row["variant"] == variant for row in rows)
        ]
        for variant in present_variants:
            if variant == baseline_variant:
                continue
            paired: dict[str, list[float]] = {metric: [] for metric in metrics}
            for row in rows:
                if row["model"] != model or row["variant"] != variant:
                    continue
                baseline_key = (
                    model,
                    baseline_variant,
                    int(row["input_len"]),
                    int(row["output_len"]),
                    float(row["offered_qps"]),
                )
                baseline = index.get(baseline_key)
                if baseline is None:
                    continue
                for metric, direction in metrics.items():
                    optimized_value = float(row[metric])
                    baseline_value = float(baseline[metric])
                    ratio = (
                        baseline_value / optimized_value
                        if direction == "lower"
                        else optimized_value / baseline_value
                    )
                    paired[metric].append(ratio)

            if not any(paired.values()):
                continue
            output[model][variant] = {
                "paired_points": len(next(iter(paired.values()))),
                "median_speedup": {metric: median(values) for metric, values in paired.items()},
                "median_percent_improvement": {
                    metric: (
                        100.0 * (1.0 - 1.0 / median(values))
                        if metrics[metric] == "lower"
                        else 100.0 * (median(values) - 1.0)
                    )
                    for metric, values in paired.items()
                },
            }
    return output


def addon_changes_vs_default(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = {result_key(row): row for row in rows}
    addons = {"all+q", "all+sd(1B)", "all+sd(1B+p)", "all+sd(3B)"}
    output: dict[str, Any] = {}
    for model in MODEL_ORDER:
        output[model] = {}
        for variant in VARIANT_ORDER:
            if variant not in addons:
                continue
            changes: list[dict[str, float]] = []
            for row in rows:
                if row["model"] != model or row["variant"] != variant:
                    continue
                baseline = index.get(
                    (
                        model,
                        "default",
                        int(row["input_len"]),
                        int(row["output_len"]),
                        float(row["offered_qps"]),
                    )
                )
                if baseline is None:
                    continue
                changes.append(
                    {
                        "output_throughput_pct": 100.0
                        * (
                            float(row["output_throughput"]) / float(baseline["output_throughput"])
                            - 1.0
                        ),
                        "p99_e2el_reduction_pct": 100.0
                        * (1.0 - float(row["p99_e2el_ms"]) / float(baseline["p99_e2el_ms"])),
                        "p99_ttft_reduction_pct": 100.0
                        * (1.0 - float(row["p99_ttft_ms"]) / float(baseline["p99_ttft_ms"])),
                    }
                )
            if changes:
                output[model][variant] = {
                    "paired_points": len(changes),
                    "median_output_throughput_pct": median(
                        item["output_throughput_pct"] for item in changes
                    ),
                    "median_p99_e2el_reduction_pct": median(
                        item["p99_e2el_reduction_pct"] for item in changes
                    ),
                    "median_p99_ttft_reduction_pct": median(
                        item["p99_ttft_reduction_pct"] for item in changes
                    ),
                }
    return output


def high_load_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = {result_key(row): row for row in rows}
    output: dict[str, Any] = {}
    for model in MODEL_ORDER:
        candidates = [
            row
            for row in rows
            if row["model"] == model
            and row["input_len"] == 512
            and row["output_len"] == 128
            and row["offered_qps"] == 16.0
            and row["variant"] != "no-opts"
        ]
        best_latency = min(candidates, key=lambda item: float(item["p99_e2el_ms"]))
        best_throughput = max(candidates, key=lambda item: float(item["request_throughput"]))
        baseline = index[(model, "no-opts", 512, 128, 16.0)]

        def comparison(row: dict[str, Any], baseline_row: dict[str, Any]) -> dict[str, Any]:
            return {
                **context(row),
                "achieved_request_throughput": row["request_throughput"],
                "output_throughput": row["output_throughput"],
                "total_token_throughput": row["total_token_throughput"],
                "offered_load_attained_pct": 100.0 * float(row["request_throughput"]) / 16.0,
                "request_throughput_speedup_vs_control": float(row["request_throughput"])
                / float(baseline_row["request_throughput"]),
                "p99_e2el_ms": row["p99_e2el_ms"],
                "p99_e2el_speedup_vs_control": float(baseline_row["p99_e2el_ms"])
                / float(row["p99_e2el_ms"]),
                "p99_e2el_reduction_pct_vs_control": 100.0
                * (1.0 - float(row["p99_e2el_ms"]) / float(baseline_row["p99_e2el_ms"])),
            }

        output[model] = {
            "serialized_control": {
                **context(baseline),
                "achieved_request_throughput": baseline["request_throughput"],
                "output_throughput": baseline["output_throughput"],
                "p99_e2el_ms": baseline["p99_e2el_ms"],
            },
            "lowest_p99_e2el": comparison(best_latency, baseline),
            "highest_request_throughput": comparison(best_throughput, baseline),
        }
    return output


def model_size_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = {result_key(row): row for row in rows}
    metrics = {
        "p50_ttft_ms": "lower",
        "p99_ttft_ms": "lower",
        "p50_tpot_ms": "lower",
        "p99_tpot_ms": "lower",
        "p50_e2el_ms": "lower",
        "p99_e2el_ms": "lower",
        "output_throughput": "higher",
    }
    changes: dict[str, list[float]] = {metric: [] for metric in metrics}
    for row in rows:
        if row["model"] != "Llama-3.2-3B" or row["variant"] != "default":
            continue
        teacher = index[
            (
                "Llama-3.1-8B",
                "default",
                int(row["input_len"]),
                int(row["output_len"]),
                float(row["offered_qps"]),
            )
        ]
        for metric, direction in metrics.items():
            student_value = float(row[metric])
            teacher_value = float(teacher[metric])
            change = (
                100.0 * (1.0 - student_value / teacher_value)
                if direction == "lower"
                else 100.0 * (student_value / teacher_value - 1.0)
            )
            changes[metric].append(change)
    return {
        "comparison": "Llama-3.2-3B versus Llama-3.1-8B under vLLM default",
        "paired_points": len(next(iter(changes.values()))),
        "nominal_parameter_reduction_pct": 62.5,
        "nominal_size_ratio": 8.0 / 3.0,
        "median_3b_improvement_pct": {metric: median(values) for metric, values in changes.items()},
    }


def observed_upper_bounds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = {result_key(row): row for row in rows}
    output: dict[str, Any] = {}
    for model in MODEL_ORDER:
        candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for row in rows:
            if row["model"] != model or row["variant"] == "no-opts":
                continue
            baseline = index[
                (
                    model,
                    "no-opts",
                    int(row["input_len"]),
                    int(row["output_len"]),
                    float(row["offered_qps"]),
                )
            ]
            speedup = float(baseline["p99_e2el_ms"]) / float(row["p99_e2el_ms"])
            candidates.append((speedup, row, baseline))
        speedup, row, baseline = max(candidates, key=lambda item: item[0])
        output[model] = {
            **context(row),
            "serialized_control_p99_e2el_ms": baseline["p99_e2el_ms"],
            "optimized_p99_e2el_ms": row["p99_e2el_ms"],
            "p99_e2el_speedup": speedup,
            "p99_e2el_reduction_pct": 100.0 * (1.0 - 1.0 / speedup),
            "interpretation": "Queue-dominated upper bound versus serialized control",
        }
    return output


def parse_dmon(path: Path) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    if not path.exists():
        return samples
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != len(DMON_FIELDS):
            continue
        try:
            sample = {field: float(value) for field, value in zip(DMON_FIELDS, parts, strict=True)}
        except ValueError:
            continue
        samples.append(sample)
    return samples


def telemetry_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "power_w",
        "gpu_temp_c",
        "memory_temp_c",
        "sm_util_pct",
        "memory_util_pct",
        "framebuffer_mb",
        "bar1_mb",
        "pcie_rx_mb_s",
        "pcie_tx_mb_s",
    ]
    summaries: list[dict[str, Any]] = []
    for row in rows:
        path = row["_benchmark_dir"] / "mon" / f"{row['_result_stem']}.txt"
        samples = parse_dmon(path)
        summary: dict[str, Any] = {
            "source_file": str(path),
            "model": row["model"],
            "variant": row["variant"],
            "input_len": row["input_len"],
            "output_len": row["output_len"],
            "offered_qps": row["offered_qps"],
            "sample_count": len(samples),
        }
        for field in fields:
            values = [sample[field] for sample in samples]
            summary[f"{field}_mean"] = mean(values) if values else None
            summary[f"{field}_median"] = median(values) if values else None
            summary[f"{field}_max"] = max(values) if values else None
        summaries.append(summary)
    return summaries


def build_summary(rows: list[dict[str, Any]], telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    completed = sum(int(row["completed"]) for row in rows)
    failed = sum(int(row["failed"]) for row in rows)
    telemetry_samples = sum(int(row["sample_count"]) for row in telemetry)
    summary = {
        "schema_version": 1,
        "source": {
            "bench_directory": "bench/bench_results",
            "result_filename_pattern": "default-random-{input}x{output}-{qps}qps-200p.json",
            "date_range": {
                "first": min(str(row["date"]) for row in rows),
                "last": max(str(row["date"]) for row in rows),
            },
            "environment": {
                "gpu": "NVIDIA H200 (one visible device per run)",
                "python": "3.12.5",
                "pytorch": "2.10.0+cu126",
                "cuda": "12.6",
                "vllm": "0.18.0",
            },
        },
        "experiment_scale": {
            "models": len({row["model"] for row in rows}),
            "serving_variants_overall": len({row["variant"] for row in rows}),
            "model_variant_combinations": len({(row["model"], row["variant"]) for row in rows}),
            "workload_points_per_complete_combination": 9,
            "result_json_files": len(rows),
            "completed_requests": completed,
            "failed_requests": failed,
            "recorded_completion_rate_pct": 100.0 * completed / (completed + failed),
            "input_tokens": sum(int(row["total_input_tokens"]) for row in rows),
            "output_tokens": sum(int(row["total_output_tokens"]) for row in rows),
            "total_tokens": sum(
                int(row["total_input_tokens"]) + int(row["total_output_tokens"]) for row in rows
            ),
            "aggregate_benchmark_duration_s": sum(float(row["duration_s"]) for row in rows),
            "parseable_gpu_telemetry_samples": telemetry_samples,
            "gpu_telemetry_fields_per_sample": len(DMON_FIELDS) - 1,
            "input_token_range": [
                min(int(row["input_len"]) for row in rows),
                max(int(row["input_len"]) for row in rows),
            ],
            "output_token_range": [
                min(int(row["output_len"]) for row in rows),
                max(int(row["output_len"]) for row in rows),
            ],
            "offered_qps_range": [
                min(float(row["offered_qps"]) for row in rows),
                max(float(row["offered_qps"]) for row in rows),
            ],
        },
        "peak_measurements": {
            "request_throughput": peak(rows, "request_throughput"),
            "output_throughput": peak(rows, "output_throughput"),
            "total_token_throughput": peak(rows, "total_token_throughput"),
            "max_output_tokens_per_s": peak(rows, "max_output_tokens_per_s"),
            "max_concurrent_requests": peak(rows, "max_concurrent_requests"),
            "telemetry": {
                "power_w": max(
                    float(row["power_w_max"]) for row in telemetry if row["power_w_max"] is not None
                ),
                "sm_util_pct": max(
                    float(row["sm_util_pct_max"])
                    for row in telemetry
                    if row["sm_util_pct_max"] is not None
                ),
                "memory_util_pct": max(
                    float(row["memory_util_pct_max"])
                    for row in telemetry
                    if row["memory_util_pct_max"] is not None
                ),
                "framebuffer_mb": max(
                    float(row["framebuffer_mb_max"])
                    for row in telemetry
                    if row["framebuffer_mb_max"] is not None
                ),
            },
        },
        "median_speedups_vs_serialized_control": paired_speedup_summary(rows, "no-opts"),
        "high_load_512x128_at_16_qps": high_load_summary(rows),
        "model_size_comparison": model_size_comparison(rows),
        "addon_changes_vs_default": addon_changes_vs_default(rows),
        "observed_upper_bounds": observed_upper_bounds(rows),
        "caveats": [
            "One run is available per model/variant/workload point; "
            "no confidence intervals are implied.",
            "The no-opts control uses --max-num-seqs 1 and is a serialized "
            "control, not a complete removal of continuous batching.",
            "Workloads use synthetic random tokens rather than production or GSM8K request traces.",
            "The runtime corpus benchmarks base 3B and 8B models, not the distilled checkpoints.",
            "No checked-in model-quality summaries support an accuracy-retention claim.",
        ],
    }
    return round_floats(summary)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(output_dir: Path, names: list[str]) -> None:
    lines = []
    for name in names:
        digest = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (output_dir / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export normalized benchmark data and resume-oriented metrics"
    )
    parser.add_argument("--bench-dir", default="bench/bench_results")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench_dir = Path(args.bench_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(bench_dir)
    if not rows:
        raise SystemExit(f"No benchmark results found under {bench_dir}")
    telemetry = telemetry_summary(rows)
    summary = build_summary(rows, telemetry)

    result_name = "benchmark_results.csv"
    telemetry_name = "gpu_telemetry_summary.csv"
    summary_name = "summary.json"

    write_csv(output_dir / result_name, [public_row(row) for row in rows], RESULT_FIELDS)
    telemetry_fields = list(telemetry[0])
    write_csv(output_dir / telemetry_name, telemetry, telemetry_fields)
    (output_dir / summary_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_checksums(output_dir, [result_name, telemetry_name, summary_name])

    print(f"Exported {len(rows)} benchmark rows to {output_dir / result_name}")
    print(
        f"Exported {sum(row['sample_count'] for row in telemetry)} telemetry samples "
        f"across {len(telemetry)} summaries to {output_dir / telemetry_name}"
    )
    print(f"Wrote aggregate metrics to {output_dir / summary_name}")
    print(f"Wrote integrity hashes to {output_dir / 'CHECKSUMS.sha256'}")


if __name__ == "__main__":
    main()
