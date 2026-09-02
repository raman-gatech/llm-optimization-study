"""Build the static portfolio site from versioned source assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="dist")
    args = parser.parse_args()
    output_arg = Path(args.output_dir)
    output = output_arg if output_arg.is_absolute() else (ROOT / output_arg).resolve()

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for name in ("index.html", "styles.css", "app.js", "favicon.svg"):
        copy_file(ROOT / "site" / name, output / name)
    copy_file(ROOT / "results" / "summary.json", output / "data" / "summary.json")
    copy_file(
        ROOT / "results" / "benchmark_results.csv",
        output / "downloads" / "benchmark_results.csv",
    )
    copy_file(
        ROOT / "results" / "gpu_telemetry_summary.csv",
        output / "downloads" / "gpu_telemetry_summary.csv",
    )
    copy_file(ROOT / "Group8-LLMOptimization.pdf", output / "downloads" / "project_proposal.pdf")

    figures = []
    for source in sorted((ROOT / "figures").glob("*.png")):
        copy_file(source, output / "figures" / source.name)
        figures.append(source.name)
    (output / "data" / "figures.json").write_text(
        json.dumps(figures, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Built {output} with {len(figures)} figures")


if __name__ == "__main__":
    main()
