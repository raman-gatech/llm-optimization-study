#!/usr/bin/env python3
"""Publish local GSM8K model artifacts to the Hugging Face Hub.

This script is tailored to the model outputs in this repository. It can:
  - list the known publishable artifacts
  - recover a missing ``final/`` export from a selected checkpoint
  - generate a lightweight model card for each upload
  - push model files plus run/eval metadata to the Hugging Face Hub
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import HfApi


MODEL_OPT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    repo_slug: str
    title: str
    role: str
    variant: str
    base_model: str
    local_dir: Path
    run_config: Optional[Path] = None
    run_config_repo_path: str = "run_config.json"
    eval_summary: Optional[Path] = None
    best_checkpoint: Optional[Path] = None
    trainer_state: Optional[Path] = None
    teacher_label: Optional[str] = None
    notes: Optional[str] = None


ARTIFACTS: dict[str, ArtifactSpec] = {
    "teacher-8b-ce-2026-04-10": ArtifactSpec(
        key="teacher-8b-ce-2026-04-10",
        repo_slug="teacher-8b-gsm8k-ce-20260410",
        title="Teacher 8B CE",
        role="teacher",
        variant="cross-entropy fine-tune",
        base_model="meta-llama/Meta-Llama-3.1-8B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "hf_teacher_8b_gsm8k_2026-04-10" / "final",
        run_config=MODEL_OPT_ROOT / "outputs" / "hf_teacher_8b_gsm8k_2026-04-10" / "run_config.json",
        eval_summary=MODEL_OPT_ROOT / "outputs" / "hf_teacher_8b_gsm8k_2026-04-10" / "eval" / "test_summary.json",
        notes="Teacher model fine-tuned on GSM8K with the repository's chain-of-thought prompt format.",
    ),
    "student-3b-ce-2026-04-10": ArtifactSpec(
        key="student-3b-ce-2026-04-10",
        repo_slug="student-3b-gsm8k-ce-20260410",
        title="Student 3B CE",
        role="student",
        variant="cross-entropy fine-tune",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-10" / "final",
        run_config=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-10" / "run_config.json",
        eval_summary=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-10" / "eval" / "test_summary.json",
        notes="Direct CE baseline student exported from the successful 2026-04-10 run.",
    ),
    "student-3b-ce-2026-04-14-best": ArtifactSpec(
        key="student-3b-ce-2026-04-14-best",
        repo_slug="student-3b-gsm8k-ce-20260414-best",
        title="Student 3B CE",
        role="student",
        variant="cross-entropy fine-tune (best recovered checkpoint)",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-14" / "final",
        run_config=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-14" / "run_config.json",
        best_checkpoint=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-14" / "checkpoint-2000",
        trainer_state=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-14" / "checkpoint-2000" / "trainer_state.json",
        notes=(
            "Recovered from the best checkpoint because the 2026-04-14 training job hit the Slurm time limit "
            "before the script wrote output_dir/final."
        ),
    ),
    "student-3b-full-kd-ft-teacher-2026-04-10": ArtifactSpec(
        key="student-3b-full-kd-ft-teacher-2026-04-10",
        repo_slug="student-3b-gsm8k-full-kd-ft-teacher-20260410",
        title="Student 3B Full KD",
        role="student",
        variant="full-logit KD",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "hf_full_kd_teacher_2026-04-10" / "final",
        run_config=MODEL_OPT_ROOT / "outputs" / "hf_full_kd_teacher_2026-04-10" / "run_config.json",
        eval_summary=MODEL_OPT_ROOT / "outputs" / "hf_full_kd_teacher_2026-04-10" / "eval" / "test_summary.json",
        teacher_label="Fine-tuned GSM8K teacher 8B (2026-04-10)",
    ),
    "student-3b-topk-kd-ft-teacher-2026-04-10": ArtifactSpec(
        key="student-3b-topk-kd-ft-teacher-2026-04-10",
        repo_slug="student-3b-gsm8k-topk-kd-ft-teacher-20260410",
        title="Student 3B Top-K KD",
        role="student",
        variant="top-k KD",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "hf_topk_kd_teacher_2026-04-10" / "final",
        run_config=MODEL_OPT_ROOT / "outputs" / "hf_topk_kd_teacher_2026-04-10" / "run_config.json",
        eval_summary=MODEL_OPT_ROOT / "outputs" / "hf_topk_kd_teacher_2026-04-10" / "eval" / "test_summary.json",
        teacher_label="Fine-tuned GSM8K teacher 8B (2026-04-10)",
    ),
    "student-3b-topk-hidden-ft-teacher-2026-04-13": ArtifactSpec(
        key="student-3b-topk-hidden-ft-teacher-2026-04-13",
        repo_slug="student-3b-gsm8k-topk-hidden-ft-teacher-20260413",
        title="Student 3B Top-K KD + Hidden Match",
        role="student",
        variant="top-k KD with hidden-state matching",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "hf_topk_hidden_teacher_2026-04-13" / "final",
        run_config=MODEL_OPT_ROOT / "outputs" / "hf_topk_hidden_teacher_2026-04-13" / "run_config.json",
        eval_summary=MODEL_OPT_ROOT / "outputs" / "hf_topk_hidden_teacher_2026-04-13" / "eval" / "test_summary.json",
        teacher_label="Fine-tuned GSM8K teacher 8B (2026-04-10)",
    ),
    "student-3b-full-kd-strong-2026-04-09": ArtifactSpec(
        key="student-3b-full-kd-strong-2026-04-09",
        repo_slug="student-3b-gsm8k-full-kd-strong-20260409",
        title="Student 3B Full KD Strong",
        role="student",
        variant="full-logit KD",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_full_kd_strong_2026-04-09" / "final",
        run_config=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_full_kd_strong_2026-04-09" / "run_config.json",
        eval_summary=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_full_kd_strong_2026-04-09" / "eval" / "test_summary.json",
        teacher_label="Fine-tuned GSM8K teacher 8B (2026-04-10)",
        notes="The directory name says 'strong', but the Slurm launch script points to the fine-tuned GSM8K teacher export.",
    ),
    "student-3b-topk-kd-strong-2026-04-09": ArtifactSpec(
        key="student-3b-topk-kd-strong-2026-04-09",
        repo_slug="student-3b-gsm8k-topk-kd-strong-20260409",
        title="Student 3B Top-K KD Strong",
        role="student",
        variant="top-k KD",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_topk_kd_strong_2026-04-09" / "final",
        run_config=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_topk_kd_strong_2026-04-09" / "run_config.json",
        eval_summary=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_topk_kd_strong_2026-04-09" / "eval" / "test_summary.json",
        teacher_label="Fine-tuned GSM8K teacher 8B (2026-04-10)",
        notes="The directory name says 'strong', but the Slurm launch script points to the fine-tuned GSM8K teacher export.",
    ),
    "student-3b-ce-2026-04-14-ckpt-2000": ArtifactSpec(
        key="student-3b-ce-2026-04-14-ckpt-2000",
        repo_slug="student-3b-gsm8k-ce-20260414-checkpoint-2000",
        title="Student 3B CE Checkpoint 2000",
        role="student",
        variant="training checkpoint at step 2000",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-14" / "checkpoint-2000",
        run_config=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-14" / "run_config.json",
        trainer_state=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-14" / "checkpoint-2000" / "trainer_state.json",
        notes="Preserves the exact checkpoint directory from the interrupted 2026-04-14 CE run.",
    ),
    "student-3b-ce-2026-04-14-ckpt-6000": ArtifactSpec(
        key="student-3b-ce-2026-04-14-ckpt-6000",
        repo_slug="student-3b-gsm8k-ce-20260414-checkpoint-6000",
        title="Student 3B CE Checkpoint 6000",
        role="student",
        variant="training checkpoint at step 6000",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-14" / "checkpoint-6000",
        run_config=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-14" / "run_config.json",
        trainer_state=MODEL_OPT_ROOT / "outputs" / "hf_student_ce_3b_2026-04-14" / "checkpoint-6000" / "trainer_state.json",
        notes=(
            "Preserves the later 2026-04-14 CE checkpoint even though the trainer state still records "
            "checkpoint-2000 as the best checkpoint."
        ),
    ),
    "student-3b-full-kd-strong-2026-04-09-ckpt-2000": ArtifactSpec(
        key="student-3b-full-kd-strong-2026-04-09-ckpt-2000",
        repo_slug="student-3b-gsm8k-full-kd-strong-20260409-checkpoint-2000",
        title="Student 3B Full KD Strong Checkpoint 2000",
        role="student",
        variant="training checkpoint at step 2000",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_full_kd_strong_2026-04-09" / "checkpoint-2000",
        run_config=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_full_kd_strong_2026-04-09" / "run_config.json",
        trainer_state=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_full_kd_strong_2026-04-09" / "checkpoint-2000" / "trainer_state.json",
        teacher_label="Base Meta-Llama-3.1-8B teacher",
        notes="Includes optimizer, scheduler, and RNG state files in addition to the model checkpoint.",
    ),
    "student-3b-topk-kd-strong-2026-04-09-ckpt-2000": ArtifactSpec(
        key="student-3b-topk-kd-strong-2026-04-09-ckpt-2000",
        repo_slug="student-3b-gsm8k-topk-kd-strong-20260409-checkpoint-2000",
        title="Student 3B Top-K KD Strong Checkpoint 2000",
        role="student",
        variant="training checkpoint at step 2000",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_topk_kd_strong_2026-04-09" / "checkpoint-2000",
        run_config=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_topk_kd_strong_2026-04-09" / "run_config.json",
        trainer_state=MODEL_OPT_ROOT / "outputs" / "strong" / "hf_topk_kd_strong_2026-04-09" / "checkpoint-2000" / "trainer_state.json",
        teacher_label="Base Meta-Llama-3.1-8B teacher",
        notes="Includes optimizer, scheduler, and RNG state files in addition to the model checkpoint.",
    ),
    "student-3b-method0-ce-best": ArtifactSpec(
        key="student-3b-method0-ce-best",
        repo_slug="student-3b-gsm8k-method0-ce-best",
        title="Student 3B Method0 CE Best",
        role="student",
        variant="best saved checkpoint from method0 CE training",
        base_model="meta-llama/Llama-3.2-3B",
        local_dir=MODEL_OPT_ROOT / "outputs" / "method0_ce" / "checkpoints" / "best",
        eval_summary=MODEL_OPT_ROOT / "outputs" / "method0_ce" / "eval" / "test_summary.json",
        trainer_state=MODEL_OPT_ROOT / "outputs" / "method0_ce" / "checkpoints" / "best" / "trainer_state.json",
        notes="Legacy method0 baseline checkpoint from the non-HF training pipeline.",
    ),
}


DEFAULT_KEYS = list(ARTIFACTS)
CORE_KEYS = [
    "teacher-8b-ce-2026-04-10",
    "student-3b-ce-2026-04-14-best",
    "student-3b-full-kd-ft-teacher-2026-04-10",
    "student-3b-topk-kd-ft-teacher-2026-04-10",
    "student-3b-topk-hidden-ft-teacher-2026-04-13",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        help="HF namespace/org. Defaults to the authenticated user's namespace.",
    )
    parser.add_argument(
        "--repo-prefix",
        default="llmbench-",
        help="Prefix applied to every repo slug.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Explicit artifact keys to publish.",
    )
    parser.add_argument(
        "--preset",
        choices=["core", "all"],
        default="all",
        help="Artifact selection preset when --only is not provided.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known artifacts and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without creating repos or uploading files.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create/update public repos. Default is private for safety.",
    )
    parser.add_argument(
        "--overwrite-final",
        action="store_true",
        help="Rebuild a recovered final/ export even if it already exists.",
    )
    return parser.parse_args()


def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def copy_with_hardlink_fallback(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def recover_final_from_checkpoint(spec: ArtifactSpec, overwrite: bool) -> Optional[Path]:
    if spec.best_checkpoint is None:
        return None
    ensure_exists(spec.best_checkpoint, "Best checkpoint")
    if spec.local_dir.exists() and not overwrite:
        return None

    if spec.local_dir.exists():
        shutil.rmtree(spec.local_dir)
    spec.local_dir.mkdir(parents=True, exist_ok=True)

    for item in spec.best_checkpoint.iterdir():
        target = spec.local_dir / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                target,
                copy_function=lambda src, dst: copy_with_hardlink_fallback(Path(src), Path(dst)),
            )
        else:
            copy_with_hardlink_fallback(item, target)
    return spec.local_dir


def resolve_namespace(api: HfApi, namespace: Optional[str], token: Optional[str]) -> str:
    if namespace:
        return namespace
    whoami = api.whoami(token=token)
    return whoami["name"]


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def build_model_card(spec: ArtifactSpec, repo_id: str) -> str:
    run_cfg = load_json(spec.run_config) or {}
    eval_summary = load_json(spec.eval_summary) if spec.eval_summary else None
    trainer_state = load_json(spec.trainer_state) if spec.trainer_state else None

    tags = ["gsm8k", "transformers", "vllm", "text-generation"]
    if spec.role == "teacher":
        tags.append("teacher-model")
    else:
        tags.append("student-model")
    if "kd" in spec.variant.lower():
        tags.append("knowledge-distillation")

    yaml_block = textwrap.dedent(
        f"""\
        ---
        library_name: transformers
        pipeline_tag: text-generation
        base_model: {spec.base_model}
        datasets:
        - gsm8k
        tags:
        {chr(10).join(f"- {tag}" for tag in tags)}
        ---
        """
    ).strip()

    metrics_lines = []
    if eval_summary:
        metrics_lines.extend(
            [
                f"- Test relaxed exact-match accuracy: `{format_float(eval_summary.get('exact_match_accuracy'))}`",
                f"- Correct / examples: `{eval_summary.get('correct', 'n/a')}` / `{eval_summary.get('examples', 'n/a')}`",
                f"- Avg generated tokens: `{format_float(eval_summary.get('avg_generated_tokens'))}`",
                f"- Prompt style used during evaluation: `{eval_summary.get('prompt_style', 'n/a')}`",
            ]
        )
    elif trainer_state:
        metrics_lines.extend(
            [
                f"- Best validation relaxed accuracy: `{format_float(trainer_state.get('best_metric'))}`",
                f"- Best checkpoint: `{Path(trainer_state.get('best_model_checkpoint', 'n/a')).name}`",
                "- Full held-out test summary is not available because the source training job ended before the planned post-train evaluation step.",
            ]
        )
    else:
        metrics_lines.append("- No evaluation summary was bundled with this export.")

    training_lines = [
        f"- Base model: `{spec.base_model}`",
        f"- Variant: `{spec.variant}`",
        f"- Output source: `{spec.local_dir.parent}`",
    ]
    if spec.teacher_label:
        training_lines.append(f"- Teacher used for distillation: `{spec.teacher_label}`")
    for field in (
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "warmup_ratio",
        "max_seq_length",
        "alpha",
        "temperature",
        "kd_method",
        "top_k",
        "hidden_match",
        "hidden_match_weight",
    ):
        if field in run_cfg:
            training_lines.append(f"- `{field}`: `{run_cfg[field]}`")

    notes_lines = []
    if spec.notes:
        notes_lines.append(f"- {spec.notes}")
    notes_lines.extend(
        [
            "- Prompt format in this repo is `question + \"\\n\\nLet's think step by step.\\n\"`.",
            "- Original Meta Llama license and access requirements still apply to downstream use.",
            "- Training metadata and evaluation summaries are uploaded alongside the weights when available.",
        ]
    )

    return "\n".join(
        [
            yaml_block,
            "",
            f"# {spec.title}",
            "",
            f"This repo contains the `{spec.variant}` export for the `{spec.role}` model from the GSM8K workflow in this project.",
            "",
            "## Quick Use",
            "",
            f"Transformers:",
            "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'model_id = "{repo_id}"',
            "tokenizer = AutoTokenizer.from_pretrained(model_id)",
            "model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=\"auto\")",
            "```",
            "",
            "vLLM:",
            "```bash",
            f"vllm serve {repo_id} --dtype auto",
            "```",
            "",
            "## Metrics",
            *metrics_lines,
            "",
            "## Training Details",
            *training_lines,
            "",
            "## Notes",
            *notes_lines,
        ]
    ).strip() + "\n"


def select_artifacts(args: argparse.Namespace) -> list[ArtifactSpec]:
    if args.only:
        unknown = [key for key in args.only if key not in ARTIFACTS]
        if unknown:
            raise KeyError(f"Unknown artifact key(s): {', '.join(unknown)}")
        keys = args.only
    elif args.preset == "core":
        keys = CORE_KEYS
    else:
        keys = DEFAULT_KEYS
    return [ARTIFACTS[key] for key in keys]


def print_artifact_list() -> None:
    print("Known publishable artifacts:\n")
    for spec in ARTIFACTS.values():
        exists = spec.local_dir.exists() or (spec.best_checkpoint and spec.best_checkpoint.exists())
        status = "ready" if exists else "missing"
        print(f"- {spec.key}")
        print(f"  repo slug: {spec.repo_slug}")
        print(f"  local dir: {spec.local_dir}")
        if spec.best_checkpoint:
            print(f"  recovery checkpoint: {spec.best_checkpoint}")
        print(f"  status: {status}")


def upload_single_artifact(
    api: HfApi,
    spec: ArtifactSpec,
    namespace: str,
    repo_prefix: str,
    token: Optional[str],
    private: bool,
    dry_run: bool,
    overwrite_final: bool,
) -> None:
    recovered = recover_final_from_checkpoint(spec, overwrite=overwrite_final)
    if recovered is not None:
        print(f"[recover] {spec.key}: rebuilt {recovered} from {spec.best_checkpoint}")

    ensure_exists(spec.local_dir, f"Artifact directory for {spec.key}")
    if spec.run_config is not None:
        ensure_exists(spec.run_config, f"Run config for {spec.key}")

    repo_id = f"{namespace}/{repo_prefix}{spec.repo_slug}"
    model_card = build_model_card(spec, repo_id=repo_id)

    print(f"[plan] {spec.key} -> {repo_id}")
    if dry_run:
        return

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(spec.local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload model files for {spec.key}",
        token=token,
    )
    if spec.run_config is not None and spec.run_config.exists():
        api.upload_file(
            path_or_fileobj=str(spec.run_config),
            path_in_repo=spec.run_config_repo_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add run config for {spec.key}",
            token=token,
        )
    if spec.eval_summary and spec.eval_summary.exists():
        api.upload_file(
            path_or_fileobj=str(spec.eval_summary),
            path_in_repo="eval/test_summary.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add eval summary for {spec.key}",
            token=token,
        )
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
        handle.write(model_card)
        temp_readme = Path(handle.name)
    try:
        api.upload_file(
            path_or_fileobj=str(temp_readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add model card for {spec.key}",
            token=token,
        )
    finally:
        temp_readme.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.list:
        print_artifact_list()
        return

    selected = select_artifacts(args)
    api = HfApi()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    namespace = resolve_namespace(api, args.namespace, token)
    private = not args.public

    print(f"Using namespace: {namespace}")
    print(f"Repo visibility: {'public' if args.public else 'private'}")
    print(f"Repo prefix: {args.repo_prefix}")
    print("")

    for spec in selected:
        upload_single_artifact(
            api=api,
            spec=spec,
            namespace=namespace,
            repo_prefix=args.repo_prefix,
            token=token,
            private=private,
            dry_run=args.dry_run,
            overwrite_final=args.overwrite_final,
        )


if __name__ == "__main__":
    main()
