#!/bin/bash
#SBATCH --job-name=hf-fullkd-strong-eval
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/hf_full_kd_strong_2026-04-09/eval/eval-%j.out
#SBATCH --error=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/hf_full_kd_strong_2026-04-09/eval/eval-%j.err

set -e

echo "=== JOB START ==="
echo "hostname: $(hostname)"
echo "time: $(date)"

source /home/hice1/rswaminathan38/scratch/miniconda3/etc/profile.d/conda.sh
conda activate /home/hice1/rswaminathan38/scratch/conda_envs/llm-train

echo "python: $(which python)"
python -V
nvidia-smi

cd /storage/ice1/3/3/rswaminathan38/LLM_Bench/LLMOptimization/Model_Optimizations

mkdir -p /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/hf_full_kd_strong_2026-04-09/eval

echo "=== START EVALUATION (RELAXED, HF FULL KD STRONG) ==="
python -m src.evaluate_hf_relaxed \
  --checkpoint_dir /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/hf_full_kd_strong_2026-04-09/final \
  --student_model meta-llama/Llama-3.2-3B \
  --output_dir /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/hf_full_kd_strong_2026-04-09/eval \
  --max_new_tokens 256

echo "=== JOB END ==="
echo "time: $(date)"
