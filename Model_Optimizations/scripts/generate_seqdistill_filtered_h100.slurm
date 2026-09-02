#!/bin/bash
#SBATCH --job-name=seqdistill-gen
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=16:00:00
#SBATCH --output=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/seqdistill_teacher_traces_2026-04-10/logs/gen-%j.out
#SBATCH --error=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/seqdistill_teacher_traces_2026-04-10/logs/gen-%j.err

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Meta-Llama-3.1-8B}"

echo "=== JOB START ==="
echo "hostname: $(hostname)"
echo "time: $(date)"
echo "teacher_model: ${TEACHER_MODEL}"

source /home/hice1/rswaminathan38/scratch/miniconda3/etc/profile.d/conda.sh
conda activate /home/hice1/rswaminathan38/scratch/conda_envs/llm-train

echo "python: $(which python)"
python -V
nvidia-smi

cd /storage/ice1/3/3/rswaminathan38/LLM_Bench/LLMOptimization/Model_Optimizations

mkdir -p /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/seqdistill_teacher_traces_2026-04-10/logs
mkdir -p /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/seqdistill_teacher_traces_2026-04-10/data

echo "=== START GENERATION (SEQUENCE DISTILLATION DATA) ==="
python -m src.generate_seqdistill_dataset \
  --teacher_model "${TEACHER_MODEL}" \
  --output_jsonl /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/seqdistill_teacher_traces_2026-04-10/data/teacher_traces_all.jsonl \
  --filtered_output_jsonl /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/seqdistill_teacher_traces_2026-04-10/data/teacher_traces_filtered.jsonl \
  --summary_json /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/seqdistill_teacher_traces_2026-04-10/data/teacher_traces_summary.json \
  --split train \
  --max_new_tokens 256

echo "=== JOB END ==="
echo "time: $(date)"
