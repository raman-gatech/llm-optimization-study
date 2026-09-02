# GSM8K Model Optimization

This directory contains the GPU training, knowledge-distillation,
teacher-trace generation, evaluation, and cluster-launch code for the
Llama 3.1-8B teacher / Llama 3.2-3B student portion of the project.

The preferred workflows use Hugging Face Trainer. The config-driven pipeline
is retained to reproduce the original Method 0/1/2 experiments.

## Layout

```text
Model_Optimizations/
├── configs/          # portable YAMLs for CE, full KD, and top-k KD
├── env/              # archived project Conda/pip environment exports
├── sanity_checks/    # gated-model download and compatibility checks
├── scripts/          # Slurm recipes, plots, and Hub publishing
└── src/
    ├── data/         # GSM8K dataset, prompting, and collation
    ├── losses/       # CE, full-KL, and top-k KL objectives
    ├── models/       # model/tokenizer loading
    ├── trainers/     # original config-driven CE trainer
    └── utils/        # answer extraction
```

Shared, unit-tested prompting, numeric scoring, and causally shifted KD
primitives live in the root `llm_optimization/` package and are imported by the
active HF workflows.

## Environment

From the repository root, create a Linux/CUDA environment compatible with your
hardware and install the training extra:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[training]'
export HF_TOKEN='hf_...'
cd Model_Optimizations
```

Meta Llama access is gated. Accept the applicable model license through
Hugging Face before running. Never write the token into a tracked file.

The files under `env/` record the original PACE software environment. They are
useful provenance, but their account-specific Conda prefixes are not intended
as portable installation instructions.

## Workflows

| Workflow | Command | Primary output |
|---|---|---|
| Config-driven CE/full/top-k KD | `python -m src.train --config <yaml>` | `checkpoints/best` and `checkpoints/final` |
| HF CE fine-tuning | `python -m src.train_hf_ce ...` | checkpoints and `final/` |
| HF full/top-k/hidden KD | `python -m src.train_hf_kd ...` | checkpoints and `final/` |
| Teacher traces | `python -m src.generate_seqdistill_dataset ...` | all and correct-only JSONL |
| Sequence distillation | `python -m src.train_hf_seqdistill ...` | checkpoints and `final/` |
| Config-driven evaluation | `python -m src.evaluate --config <yaml>` | prediction JSONL and summary JSON |
| HF evaluation | `python -m src.evaluate_hf_relaxed ...` | prediction JSONL and summary JSON |

### Cross-entropy baseline

```bash
python -m src.train_hf_ce \
  --model_name meta-llama/Llama-3.2-3B \
  --output_dir outputs/hf_student_ce \
  --num_train_epochs 10 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --max_seq_length 1024
```

The trainer derives validation data from the GSM8K training split and leaves
the official test split untouched for final evaluation.

### Top-k knowledge distillation

```bash
python -m src.train_hf_kd \
  --student_model meta-llama/Llama-3.2-3B \
  --teacher_model /path/to/teacher/final \
  --output_dir outputs/hf_topk_kd \
  --kd_method topk \
  --top_k 64 \
  --alpha 0.7 \
  --temperature 2.0 \
  --kd_on_answer_only \
  --num_train_epochs 10 \
  --max_seq_length 1024
```

`--alpha` is the CE weight; KD receives `1 - alpha`. Use `--kd_method full`
for full-vocabulary KL. Top-k is generally the more memory-practical
single-GPU method for an 8B teacher.

Optional hidden alignment:

```text
--hidden_match --hidden_match_weight 0.05 --hidden_match_layer -1 \
--hidden_match_on_answer_only
```

When teacher and student hidden sizes differ, the trainer learns a projection
as part of the optimization module.

### Sequence distillation

```bash
python -m src.generate_seqdistill_dataset \
  --teacher_model /path/to/teacher/final \
  --output_jsonl outputs/traces/all.jsonl \
  --filtered_output_jsonl outputs/traces/correct.jsonl

python -m src.train_hf_seqdistill \
  --student_model meta-llama/Llama-3.2-3B \
  --distill_data_path outputs/traces/correct.jsonl \
  --output_dir outputs/hf_seqdistill \
  --require_correct \
  --mix_gold_data
```

Teacher traces are filtered using completion-only numeric answers, not numbers
copied from the question.

### Evaluation

```bash
python -m src.evaluate_hf_relaxed \
  --checkpoint_dir outputs/hf_topk_kd/final \
  --output_dir outputs/hf_topk_kd/eval \
  --max_new_tokens 256
```

Evaluation uses the same real-newline prompt as training, decodes generated
tokens only, and scores robust numeric equality with tolerance. It writes:

```text
<output>/
├── test_predictions.jsonl
└── test_summary.json
```

Numeric GSM8K exact match is the primary metric. BLEU/ROUGE are secondary text
comparisons and should not replace mathematical correctness.

## Config-driven experiments

Available YAMLs:

- `configs/method0_ce.yaml`
- `configs/method1_full_kd_tuned.yaml`
- `configs/method1_full_kd_answer.yaml`
- `configs/method1_full_kd_answer_10k.yaml`
- `configs/method2_topk_kd.yaml`
- `configs/method2_topk_kd_tuned.yaml`
- `configs/method2_topk_kd_answer.yaml`
- `configs/method2_topk_kd_answer_10k.yaml`

All use repository-relative `outputs/...` directories. Example:

```bash
python -m src.train --config configs/method0_ce.yaml
python -m src.evaluate --config configs/method0_ce.yaml
```

The legacy config uses `kd_alpha` as the KD weight, the inverse of the HF
trainer's `--alpha` convention.

## Cluster recipes

The `.slurm` files under `scripts/` preserve the original H100 experiment
matrix. Their partitions, accounts, Conda paths, log locations, and some output
paths are PACE-specific. Review and adapt every launcher before submitting on a
different account or cluster.

The Hugging Face Hub helper is read-only unless an upload command is explicitly
run:

```bash
python scripts/push_models_to_hub.py --list
python scripts/push_models_to_hub.py --preset core --dry-run
```

## Results and artifact policy

Training outputs belong under `outputs/` and remain ignored by Git. Do not
commit checkpoints or model caches. Publish weights to approved model storage
and version lightweight evidence here:

- `run_config.json` with seed, code version, hyperparameters, and model IDs;
- `test_summary.json` with dataset/split, prompt/decoding settings, exact match,
  extraction failures, training time, and peak memory;
- optional prediction JSONL only when data licensing and privacy permit it.

This repository currently includes no trained checkpoints or GSM8K summary
artifacts, so model-quality comparisons are not yet evidence-backed. Runtime
results and their limitations are documented in the root README and
`results/README.md`.
