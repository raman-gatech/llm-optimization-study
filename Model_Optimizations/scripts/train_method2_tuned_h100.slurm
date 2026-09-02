#!/bin/bash
#SBATCH --job-name=method2-tuned-h100
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/method2_topk_kd_tuned_2026-04-07/logs/train-%j.out
#SBATCH --error=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/method2_topk_kd_tuned_2026-04-07/logs/train-%j.err

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== JOB START ==="
echo "hostname: $(hostname)"
echo "time: $(date)"

source /home/hice1/rswaminathan38/scratch/miniconda3/etc/profile.d/conda.sh
conda activate /home/hice1/rswaminathan38/scratch/conda_envs/llm-train

echo "python: $(which python)"
python -V
nvidia-smi

cd /storage/ice1/3/3/rswaminathan38/LLM_Bench/LLMOptimization/Model_Optimizations

echo "=== START TRAINING (METHOD 2 TUNED) ==="
python -m src.train --config configs/method2_topk_kd_tuned.yaml

echo "=== JOB END ==="
echo "time: $(date)"
