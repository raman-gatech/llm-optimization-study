.PHONY: install test test-torch lint format typecheck export figures site serve docker-build check

install:
	python3 -m pip install -e '.[analysis,test,dev]'

test:
	python3 -m pytest

test-torch:
	python3 -m pytest -m torch

lint:
	python3 -m ruff check llm_optimization analysis/export_results.py scripts tests

format:
	python3 -m ruff format llm_optimization analysis/export_results.py scripts tests
	python3 -m ruff check --fix llm_optimization analysis/export_results.py scripts tests

typecheck:
	python3 -m mypy llm_optimization analysis/export_results.py scripts

export:
	python3 analysis/export_results.py --bench-dir bench/bench_results --output-dir results

figures:
	python3 analysis/analyze_bench.py --bench-dir bench/bench_results --out-dir figures

site:
	python3 scripts/build_site.py --output-dir dist

serve: site
	python3 -m http.server 8000 --directory dist

docker-build:
	docker build -t llm-optimization-portfolio .

check: lint typecheck test export site
