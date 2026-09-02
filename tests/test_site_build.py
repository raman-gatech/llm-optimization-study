from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_static_site_build_contains_all_public_assets(tmp_path: Path) -> None:
    output = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_site.py"),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    required = {
        "index.html",
        "styles.css",
        "app.js",
        "favicon.svg",
        "data/summary.json",
        "data/figures.json",
        "downloads/benchmark_results.csv",
        "downloads/gpu_telemetry_summary.csv",
        "downloads/project_proposal.pdf",
    }
    for relative_path in required:
        path = output / relative_path
        assert path.is_file(), f"missing site artifact: {relative_path}"
        assert path.stat().st_size > 0, f"empty site artifact: {relative_path}"

    figure_names = json.loads((output / "data" / "figures.json").read_text())
    assert len(figure_names) == 12
    assert sorted(figure_names) == sorted(path.name for path in (ROOT / "figures").glob("*.png"))
    assert all((output / "figures" / name).is_file() for name in figure_names)


def test_site_headline_metrics_match_versioned_summary() -> None:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    scale = summary["experiment_scale"]
    html = (ROOT / "site" / "index.html").read_text()

    assert f">{scale['result_json_files']}<" in html
    assert f">{scale['completed_requests']:,}<" in html
    assert f">{scale['recorded_completion_rate_pct']:.0f}%<" in html
