#!/bin/bash

. .venv/bin/activate

export HF_HOME=$(pwd)/cache/hf
mkdir -p "$HF_HOME"

export VLLM_CONFIG_ROOT=$(pwd)/cache/vllm_config
mkdir -p "$VLLM_CONFIG_ROOT"

export VLLM_CACHE_ROOT=$(pwd)/cache/vllm_cache
mkdir -p "$VLLM_CACHE_ROOT"

# Preserve a token supplied by the shell or a local, ignored `.env` file.
# Never place credentials in this tracked script.
export HF_TOKEN="${HF_TOKEN:-}"
