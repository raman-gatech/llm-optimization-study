# Stored Results and Resume Metrics

This directory is the durable, local results layer for the LLM Optimization
project. It preserves normalized benchmark data, derived headline metrics, GPU
telemetry summaries, integrity hashes, and resume-ready language without
duplicating model weights or the full raw logs.

All numbers below are derived from the checked-in files under
`bench/bench_results/`. Regenerate them with:

```bash
python3 analysis/export_results.py \
  --bench-dir bench/bench_results \
  --output-dir results
```

## Files

| File | Purpose |
|---|---|
| [`benchmark_results.csv`](benchmark_results.csv) | One normalized row for each of the 171 vLLM result JSONs, including experiment context, throughput, concurrency, and latency statistics |
| [`gpu_telemetry_summary.csv`](gpu_telemetry_summary.csv) | One row per benchmark with aggregate power, temperature, SM utilization, memory, framebuffer, and PCIe telemetry |
| [`summary.json`](summary.json) | Machine-readable scale metrics, peak measurements, paired median speedups, high-load results, add-on tradeoffs, model-size comparisons, and caveats |
| [`CHECKSUMS.sha256`](CHECKSUMS.sha256) | SHA-256 integrity hashes for the generated CSV and JSON files |

The raw source of truth remains `bench/bench_results/`; the files here are
derived artifacts designed for analysis, reporting, and resume preparation.

## Experiment scale

| Metric | Measured value |
|---|---:|
| Models | 2 |
| Serving variants overall | 10 |
| Model/variant combinations | 19 |
| Workload points per complete combination | 9 |
| Result JSONs | 171 |
| Completed requests | 34,200 |
| Failed requests | 0 |
| Recorded request completion | 100% |
| Input tokens | 16,503,400 |
| Output tokens | 5,715,200 |
| Total tokens | 22,218,600 |
| Aggregate measured benchmark duration | 16,860.45 seconds / 4.68 hours |
| Parseable one-second GPU telemetry samples | 18,435 |
| GPU telemetry fields per sample | 21 |
| Input-length range | 128–768 tokens |
| Output-length range | 32–512 tokens |
| Offered-load range | 1–16 requests/s |

Runtime measurements used one NVIDIA H200 with Python 3.12.5, PyTorch
2.10.0+cu126, CUDA 12.6, and vLLM 0.18.0.

## Peak measurements

| Metric | Peak | Configuration |
|---|---:|---|
| Achieved request throughput | 15.57 requests/s | Llama 3.1-8B, parallel 1B speculative drafting, 512×128 at 16 req/s |
| Output throughput | 1,993.21 tokens/s | Same configuration |
| Total token throughput | 9,950.47 tokens/s | Same configuration |
| Instantaneous output rate | 4,684 tokens/s | Llama 3.2-3B default, 512×128 at 16 req/s |
| Concurrent requests | 185 | Llama 3.1-8B serialized control, 512×128 at 16 req/s |
| SM utilization | 100% | Observed in GPU telemetry |
| Power | 672 W | Observed in GPU telemetry |
| Memory-utilization counter | 78% | Observed in GPU telemetry |
| Framebuffer allocation | 130,763 MB | Observed in GPU telemetry |

## Strongest high-load results

These results use the 512-input/128-output workload at 16 offered requests per
second. The control uses `--max-num-seqs 1` and must be described as a
**serialized control**, not as a complete removal of continuous batching.

| Model/configuration | Achieved req/s | Output tok/s | Offered load attained | Throughput gain | P99 E2E | P99 reduction |
|---|---:|---:|---:|---:|---:|---:|
| 3B serialized control | 1.98 | 253.68 | 12.4% | baseline | 87.45 s | baseline |
| 3B continuous batching + async | 15.38 | 1,969.1 | 96.2% | **7.76×** | 695 ms | **99.21% / 125.8×** |
| 8B serialized control | 1.28 | 163.68 | 8.0% | baseline | 142.39 s | baseline |
| 8B vLLM default | 15.07 | 1,929.44 | 94.2% | **11.79×** | 1.10 s | **99.23% / 129.1×** |
| 8B parallel 1B drafting | **15.57** | **1,993.21** | **97.3%** | **12.18×** | 1.30 s | **99.08% / 109.2×** |

## Median results across nine paired workload points

These values compare the standard vLLM default configuration with the
serialized control. Medians are more defensible than the largest single-point
speedups because only one run exists per benchmark point.

| Metric | Llama 3.2-3B | Llama 3.1-8B |
|---|---:|---:|
| Output-throughput speedup | 1.51× / +50.9% | 2.33× / +133.4% |
| P50 E2E speedup | 42.4× | 62.7× |
| P99 E2E speedup | 27.9× | 106.8× |
| P99 E2E reduction | 96.4% | 99.1% |
| P50 TTFT speedup | 1,272.5× | 1,785.3× |
| P99 TTFT speedup | 163.6× | 2,171.6× |
| P99 TTFT reduction | 99.4% | 99.95% |
| P50 TPOT reduction | 9.6% | 6.0% |

The extreme TTFT ratios primarily measure queueing avoidance versus the
serialized control; they should not be presented as kernel-level acceleration.

## Model-size comparison

Across nine paired vLLM-default workload points, Llama 3.2-3B versus Llama
3.1-8B produced these median changes:

| Metric | 3B improvement over 8B |
|---|---:|
| P50 TTFT | 34.7% lower |
| P99 TTFT | 29.8% lower |
| P50 TPOT | 39.2% lower |
| P99 TPOT | 37.6% lower |
| P50 E2E | 39.1% lower |
| P99 E2E | 36.7% lower |
| Output throughput | 0.4% higher |

The named 8B-to-3B teacher/student design represents a nominal **62.5%
parameter-count reduction** or **2.67× smaller model class**. The small observed
throughput difference is expected because most optimized tests were constrained
by offered load rather than maximum model capacity.

## Optimization tradeoffs versus vLLM default

| Add-on | Model | Median output-throughput change | Median P99 E2E change |
|---|---|---:|---:|
| 4-bit BitsAndBytes | 3B | −0.70% | 121.99% worse |
| 4-bit BitsAndBytes | 8B | −1.62% | 158.49% worse |
| Standard 1B speculative draft | 3B | −0.82% | 308.57% worse |
| Standard 1B speculative draft | 8B | −0.57% | 181.03% worse |
| Parallel 1B speculative draft | 3B | −0.07% | 35.95% worse |
| Parallel 1B speculative draft | 8B | **+0.75%** | **8.53% better** |
| 3B speculative draft | 8B | −1.17% | 201.86% worse |

This is a useful negative-result story: optimizations are hardware- and
workload-sensitive, and additional model work can increase latency even when it
reduces precision or speculates future tokens.

## Resume-ready bullets

Use only statements that match your personal contribution. If this was shared
team work, use `co-developed`, `contributed to`, or `led <specific component>`
instead of claiming the entire system.

### Systems/performance version

- Built an NVIDIA H200 LLM benchmarking platform spanning **2 Llama models, 10
  serving configurations, and 9 workload/load profiles**, executing **171
  isolated experiments, 34,200 requests, and 22.2M tokens with 100% recorded
  request completion**.
- Optimized Llama 3.1-8B inference at **16 req/s**, increasing achieved
  throughput **11.8×** to **15.1 req/s / 1,929 output tokens/s** while reducing
  p99 end-to-end latency **99.2%**, from **142.4 seconds to 1.10 seconds**, versus
  a serialized control.
- Sustained **15.4 req/s and 1,969 output tokens/s** for Llama 3.2-3B—**96.2% of
  offered load**—while reducing p99 latency **125.8×**, from **87.4 seconds to
  695 ms**, using continuous batching and asynchronous scheduling.

### ML systems version

- Developed CE, full-logit KD, top-k KD, hidden-state alignment, and
  sequence-distillation pipelines for an **8B-to-3B teacher/student design**, a
  nominal **62.5% parameter reduction / 2.67× smaller model class**.
- Quantified model-size effects across nine paired workloads, finding that the
  3B model reduced median p99 end-to-end latency **36.7%** and p99 token latency
  **37.6%** relative to the 8B model under the same default serving stack.
- Evaluated quantization and speculative-decoding tradeoffs on H200 hardware,
  finding that 4-bit quantization increased median p99 latency **122–158%**, while
  parallel 1B drafting selectively improved 8B p99 latency **8.5%**.

### Measurement/analytics version

- Built an automated performance-analysis pipeline covering **16 latency and 3
  throughput metrics**, producing **12 performance visualizations** across TTFT,
  TPOT, ITL, end-to-end latency, workload length, and request rate.
- Instrumented **171 GPU experiments** with one-second NVIDIA telemetry and
  normalized **18.4K samples across 21 fields**, including utilization, memory,
  power, clocks, temperature, PCIe traffic, and ECC status.

## Claims not supported by the stored evidence

Do not claim any of the following unless new result artifacts are added:

- a specific GSM8K accuracy or accuracy improvement;
- a measured percentage of teacher-quality retention after distillation;
- that a distilled checkpoint produced the runtime speedups in this corpus;
- statistical significance or confidence intervals;
- production traffic or customer impact;
- multi-GPU/DDP/FSDP training;
- that the serialized control fully disables continuous batching.

The runtime corpus benchmarks base 3B and 8B models. No trained checkpoints or
model-quality summaries are stored in this copy.

## How to preserve future results

1. Keep immutable raw benchmark output under `bench/bench_results/` using the
   existing model/GPU/variant directory convention.
2. Regenerate this directory with `analysis/export_results.py` after adding raw
   runs.
3. Regenerate plots with:

   ```bash
   python analysis/analyze_bench.py \
     --bench-dir bench/bench_results \
     --out-dir figures
   ```

4. Verify generated artifacts before sharing:

   ```bash
   cd results
   shasum -a 256 -c CHECKSUMS.sha256
   ```

5. Store future model-quality summaries as lightweight JSON/CSV files containing
   at least: model/checkpoint identifier, dataset and split, seed, prompt format,
   exact-match accuracy, extraction failures, generation settings, training
   time, peak memory, and the source run configuration.
6. Do not store Hugging Face tokens, model weights, caches, or private data in
   this directory. Keep weights in approved artifact storage and record only a
   stable model identifier/checksum here.

## Version-control policy

The generated CSV/JSON files are intentionally versioned because they are
compact, reviewable evidence and their checksums make regeneration drift
visible in CI. Raw result JSON, environment metadata, and telemetry are also
versioned as source evidence. Large checkpoints, caches, and verbose process
logs remain ignored and belong in approved model/artifact storage.
