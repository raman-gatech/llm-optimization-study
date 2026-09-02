# vLLM Benchmark

## Setup and Run

1. Ensure that the working directory is the directory containing this `README.md` file.
2. If Slurm is available, run `. module.sh`.
   If module not found or Slurm is not available, install compatible versions of Python and CUDA as specified in `module.sh`.
3. If virtualenv has not been set up, run `. init.sh`.
4. Export your Hugging Face token in the shell: `export HF_TOKEN="hf_..."`.
   Never write the token into a tracked file.
5. Run `. env.sh`; it preserves the token already present in your environment.
6. If Python dependencies have not been installed, run `. install.sh`.
7. Run benchmark.
   - Usage: `python bench.py <model_name> <gpu_name> <benchmark_name> [-- <vllm_serve_args> ...]`.
   - `<model_name>` is the model name on Hugging Face, e.g. `meta-llama/Llama-3.2-3B`.
   - `<gpu_name>` is a representative substring of the full GPU device name, e.g. `'NVIDIA A100'`.
   - `<benchmark_name>` is the name of the benchmark (e.g. to indicate which optimizations are enabled).
   - `<vllm_serve_args>` are additional command line arguments passed to `vllm serve` (e.g. to enable or disable certain optimizations).
   - It is recommended to redirect the standard output and standard error to files when running actual benchmarks, e.g. `> bench.out 2> bench.err`.

## Examples

```shell
. module.sh
. env.sh
python bench.py meta-llama/Llama-3.2-3B 'NVIDIA H200' vllm-0.18.0-default > bench.out 2> bench.err
```

## Notations

The meaning of suffixes in benchmark names:

- `default`: Default settings of `vllm serve` without explicitly enable or disable any optimizations.
  By default, vLLM 0.18.0 enables continuous batching + chunked prefill + prefix caching + async scheduling (along with some other optimizations not studied by this project).
- `no-opts`: None of the four optimizations mentioned in the `default` setting is enabled.
  Notice that there is no easy way to completely disable continuous batching without changing vLLM code, so batch inference is disabled instead (by setting `--max-num-seqs 1`).
- `cb`: Enables only continuous batching.
- `cb+cp`: Enables continuous batching and chunked prefill.
- `cb+pc`: Enables continuous batching and prefix caching.
- `cb+as`: Enables continuous batching and async scheduling.
- `all+q`: Enables all four optimizations and inflight 4-bit quantization with BitsAndBytes.
