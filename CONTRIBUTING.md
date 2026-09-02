# Contributing

Contributions should preserve reproducibility and distinguish measured evidence
from planned or inferred behavior.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,test,dev]'
make check
```

Training dependencies are intentionally optional:

```bash
python -m pip install -e '.[training]'
```

The vLLM benchmark has a separate CUDA-specific lock file at
`bench/requirements.txt`.

## Pull requests

1. Open an issue for changes that alter prompts, scoring, benchmark baselines,
   workload generation, or stored-result schemas.
2. Add or update tests with behavior changes.
3. Run `make format`, then `make check`.
4. Regenerate `results/` after adding raw benchmark data.
5. Document hardware, software versions, seeds, and exact commands for new runs.
6. Never commit credentials, gated model weights, caches, or private datasets.

One run per benchmark point is not sufficient for a statistical claim. New
performance claims should include repeated runs and uncertainty where feasible.
