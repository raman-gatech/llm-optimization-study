from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exporter_reconstructs_complete_corpus(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "analysis" / "export_results.py"),
            "--bench-dir",
            str(ROOT / "bench" / "bench_results"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )

    with (tmp_path / "benchmark_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    with (tmp_path / "gpu_telemetry_summary.csv").open() as handle:
        telemetry = list(csv.DictReader(handle))
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert len(rows) == 171
    assert len(telemetry) == 171
    assert sum(int(row["completed"]) for row in rows) == 34_200
    assert sum(int(row["failed"]) for row in rows) == 0
    assert summary["experiment_scale"]["total_tokens"] == 22_218_600
    assert summary["experiment_scale"]["parseable_gpu_telemetry_samples"] == 18_435


def test_checked_in_summary_has_expected_headline_metrics() -> None:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    high_load = summary["high_load_512x128_at_16_qps"]
    assert high_load["Llama-3.1-8B"]["lowest_p99_e2el"]["variant"] == "default"
    assert high_load["Llama-3.2-3B"]["lowest_p99_e2el"]["variant"] == "cb+as"
    assert summary["experiment_scale"]["recorded_completion_rate_pct"] == 100.0
