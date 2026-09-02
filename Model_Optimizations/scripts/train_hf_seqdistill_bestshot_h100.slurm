#!/bin/bash
#SBATCH --job-name=hf-seqdistill
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=16:00:00
#SBATCH --output=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/hf_seqdistill_bestshot_2026-04-10/logs/train-%j.out
#SBATCH --error=/home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/hf_seqdistill_bestshot_2026-04-10/logs/train-%j.err

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

mkdir -p /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/hf_seqdistill_bestshot_2026-04-10/logs

echo "=== START TRAINING (HF SEQUENCE DISTILLATION BEST SHOT) ==="
python -m src.train_hf_seqdistill \
  --student_model meta-llama/Llama-3.2-3B \
  --distill_data_path /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/seqdistill_teacher_traces_2026-04-10/data/teacher_traces_filtered.jsonl \
  --output_dir /home/hice1/rswaminathan38/scratch/LLM_Bench/LLMOptimization/Model_Optimizations/outputs/hf_seqdistill_bestshot_2026-04-10 \
  --num_train_epochs 6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --val_ratio 0.05 \
  --require_correct \
  --mix_gold_data \
  --max_seq_length 1024 \
  --metric_max_new_tokens 256 \
  --save_steps 500 \
  --eval_steps 500 \
  --logging_steps 25 \
  --save_total_limit 1 \
  --bf16 \
  --gradient_checkpointing

echo "=== JOB END ==="
echo "time: $(date)"
