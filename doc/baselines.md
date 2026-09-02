# vLLM Benchmarking: Baseline Experiments (Reproducibility Guide)

## Objective

This document provides exact instructions to reproduce baseline LLM inference benchmarks using vLLM. These baselines will be used for comparison against runtime optimizations such as batching, KV caching, quantization, and scheduling strategies.

---

## 1. System Requirements

### Hardware

* GPU: NVIDIA H100 / H200 (recommended)
* VRAM:

  * 3B model: ≥ 16 GB
  * 8B model: ≥ 40 GB

### Software

* Python ≥ 3.11 and ≤ 3.13
* CUDA-compatible environment
* vLLM installed

---

## 2. Environment Setup

### Create environment

```bash
conda create -n llm-serve python=3.12 -y
conda activate llm-serve
```

(Using pure pip + virtualenv is also ok, see scripts in `bench` directory.)

### Install dependencies

```bash
pip install vllm
```

---

## 3. Start vLLM Server

### 3B Model

```bash
vllm serve meta-llama/Llama-3.2-3B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16
```

### 8B Model

```bash
vllm serve meta-llama/Llama-3.1-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16
```

---

## 4. Verify Server

```bash
curl http://127.0.0.1:8000/v1/models
```

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B",
    "prompt": "Hello",
    "max_tokens": 16,
    "temperature": 0
  }'
```

---

## 5. Directory Setup

```bash
mkdir -p bench_results/{baseline,batching,chunked_prefill,apc,quantization,scheduling,cumulative,logs,setup}
mkdir -p bench_results_8b/{baseline,batching,chunked_prefill,apc,quantization,scheduling,cumulative,logs,setup}
```

---

## 6. Save Environment Metadata

```bash
python --version > bench_results/setup/env_info.txt
pip list >> bench_results/setup/env_info.txt
nvidia-smi >> bench_results/setup/env_info.txt
```

---

## 7. Baseline Experiments

### 7.1 Workload Definitions

| Name        | Input Tokens | Output Tokens | RPS |
| ----------- | ------------ | ------------- | --- |
| short_short | 128          | 128           | 2   |
| long_short  | 768          | 32            | 2   |
| short_long  | 128          | 512           | 2   |
| long_long   | 768          | 192           | 1   |

---

### 7.2 Run Baselines (3B or 8B)

Replace model accordingly.

#### short_short

```bash
vllm bench serve \
  --backend vllm \
  --model meta-llama/Llama-3.2-3B \
  --served-model-name meta-llama/Llama-3.2-3B \
  --endpoint /v1/completions \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random \
  --num-prompts 100 \
  --request-rate 2 \
  --random-input-len 128 \
  --random-output-len 128 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result \
  --result-dir ./bench_results/baseline \
  --metadata stage=baseline workload=short_short
```

#### long_short

```bash
--random-input-len 768 \
--random-output-len 32
```

#### short_long

```bash
--random-input-len 128 \
--random-output-len 512
```

#### long_long

```bash
--random-input-len 768 \
--random-output-len 192 \
--request-rate 1
```

---

## 8. QPS Scaling Experiments

### Configuration

* Input: 512 tokens
* Output: 128 tokens

### RPS values

```bash
1, 2, 4, 8, 16
```

---

### Example (1 RPS)

```bash
vllm bench serve \
  --backend vllm \
  --model meta-llama/Llama-3.2-3B \
  --served-model-name meta-llama/Llama-3.2-3B \
  --endpoint /v1/completions \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random \
  --num-prompts 150 \
  --request-rate 1 \
  --random-input-len 512 \
  --random-output-len 128 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result \
  --result-dir ./bench_results/baseline \
  --metadata stage=baseline run=qps_curve rps=1 workload=random512x128
```

Repeat for other RPS values.

---

## 9. Logging GPU Metrics

Run in parallel:

```bash
nvidia-smi dmon -s pucvmet -d 1 > bench_results/logs/gpu_log_baseline.txt
```

---

## 10. Output Format

Each run produces a JSON file containing:

### Throughput

* request_throughput
* output_throughput
* total_token_throughput

### Latency

* TTFT
* TPOT
* ITL
* E2EL

### Percentiles

* p50, p95, p99

---

## 11. Important Notes

* Do NOT rely on filenames to identify workloads

* Always use JSON fields:

  * run
  * rps
  * workload

* Keep 3B and 8B results separate

* Ensure no other GPU processes are running

---

## 12. Reproducibility Checklist

Before running:

* [ ] Correct conda environment activated
* [ ] vLLM server running
* [ ] GPU free
* [ ] Output directories created

After running:

* [ ] JSON files generated
* [ ] Logs captured
* [ ] Metadata saved

---

## 13. Next Steps

After baseline reproduction:

1. Apply runtime optimizations:

   * Continuous batching
   * Chunked prefill
   * Prefix caching (APC)
   * Quantization
   * Scheduling strategies

2. Compare results with baselines:

   * Throughput vs load
   * Latency vs load
   * Model scaling (3B vs 8B)

---

## Contact / Notes

For issues, verify:

* server is running
* correct model name is used
* port is reachable

---
