#!/bin/bash
#SBATCH --job-name=method0-eval
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/method0_ce/eval/eval-%j.out
#SBATCH --error=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/method0_ce/eval/eval-%j.err

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

echo "=== START EVALUATION ==="
python -m src.evaluate

echo "=== JOB END ==="
echo "time: $(date)"