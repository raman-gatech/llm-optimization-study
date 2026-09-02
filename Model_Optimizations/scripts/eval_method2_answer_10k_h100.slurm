#!/bin/bash
#SBATCH --job-name=method2-answer-10k-eval-h100
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/method2_topk_kd_answer_10k_2026-04-09/eval/eval-%j.out
#SBATCH --error=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/method2_topk_kd_answer_10k_2026-04-09/eval/eval-%j.err

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

echo "=== START EVALUATION (METHOD 2 TOP-K KD ANSWER-ONLY, 10K) ==="
python -m src.evaluate --config configs/method2_topk_kd_answer_10k.yaml

echo "=== JOB END ==="
echo "time: $(date)"
