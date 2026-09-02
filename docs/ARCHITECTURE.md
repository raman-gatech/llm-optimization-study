# Architecture

The repository separates expensive GPU experimentation from lightweight,
reproducible reporting.

```text
GSM8K + teacher/student models
        |
        v
Model_Optimizations/src       bench/bench.py + vLLM
        |                             |
checkpoints/eval summaries      raw JSON + dmon telemetry
        |                             |
        +-----------> results/ <------+
                           |
             analysis + static site
                           |
                GitHub Pages / Nginx
```

## Components

- `llm_optimization/`: tested reusable prompt, scoring, and KD primitives.
- `Model_Optimizations/`: GPU training, teacher-trace generation, evaluation,
  legacy configs, and Slurm recipes.
- `bench/`: isolated vLLM server/load orchestration and raw evidence.
- `analysis/`: normalized exports and publication figures.
- `results/`: compact versioned data products and integrity hashes.
- `site/`: dependency-free portfolio presentation source.
- `scripts/build_site.py`: deterministic assembly into `dist/`.

## Deployment model

The public application is static by design. It does not require model weights,
GPU capacity, secrets, or a writable backend, and it exposes no model-generation
endpoint. GitHub Actions builds the site and deploys it to Pages. The same build
is served by an unprivileged Nginx container on port 8080.

## Trust boundaries

Raw results are immutable evidence. Derived CSV/JSON files are reproducible and
checksummed. Figures and the site consume only versioned derived data. Training
and serving credentials remain outside the repository.
