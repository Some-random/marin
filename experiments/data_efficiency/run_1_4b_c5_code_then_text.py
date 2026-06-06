# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B model, matched-budget Aryabumi code->text replica (C5).
#
# This is a TEMPLATE — cache paths for code+markup need to be filled in once
# the data source is decided (StarCoderData if access is granted, or
# codeparrot/github-code fallback). See C5_BLOCKER_2026-06-05.md.
#
# Recipe (faithful to Aryabumi et al 2408.10914 §3.1, scaled to our budget):
#   Stage 1 (steps 0-14671): 80% multi-language Stack + 20% markup
#   Stage 2 (steps 14672-29342): 90% DCLM + 10% (80/20 Stack+markup)
#
# Hyperparameters identical to A5/B4 (run_1_4b_1ep_dclm.py / run_1_4b_1ep_code25.py):
#   LR=3e-4 cosine to 0, wd=0.1, β=(0.9, 0.95), warmup=0.01, batch=256 x seq=4096,
#   29,343 steps, 30.77B total trained tokens.
#
# Hardware: 8 nodes (8 x 8 = 64 A100) with PER_DEVICE_PARALLELISM=4 so that
# global batch=256 stays IDENTICAL to A5 (4-node x per_device=8). Wall-clock
# expected ~20-24h.
#
# LR schedule: SINGLE continuous cosine across both stages. The data weights
# swap at step 14672 via Levanter's MixtureDataset staged-weights mechanism
# (LmDataConfig.train_weights accepts list[(start_seq_index, dict)]).
# This matches Aryabumi's framing of continual pretraining where "to avoid a
# full distribution shift" they keep 10% code in stage 2.

import os
from datetime import timedelta
from pathlib import Path

_secrets = Path(__file__).resolve().parents[2] / ".secrets"
if _secrets.exists():
    for line in _secrets.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig, DatasetComponent
from levanter.distributed import DistributedConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.trainer import TrainerConfig

from experiments.data_efficiency.models import model_dict

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"

# === DCLM (text, stage 2) — same 7 shards as A5/B4 ===
DCLM_SHARDS = [
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00006",
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00020",
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00026",
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00035",
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00042",
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00047",
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00071",
]
DCLM_VAL = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_val-415aea"

# === Code (Stack-style, stages 1+2) and Markup (stages 1+2) ===
# Tokenization caches produced by `experiments/data_efficiency/tokenize_starcoderdata.py`
# Cache prefixes: stack_<safe_lang>-<hash>, markup_<lang>-<hash>.
# default_tokenize replaces "-" with "_" in step names, so c-sharp -> stack_c_sharp.
_TOKENIZED_BASE = Path(BASE_TOKENIZED)


def _resolve_cache(prefix: str) -> str:
    matches = sorted(_TOKENIZED_BASE.glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(
            f"No tokenized cache for prefix '{prefix}' in {BASE_TOKENIZED}. "
            f"Run `MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python "
            f"-m experiments.data_efficiency.tokenize_starcoderdata` first."
        )
    return str(matches[-1])


_STACK_LANGS = ["java", "javascript", "php", "python", "c-sharp",
                "typescript", "c", "cpp", "go", "ruby"]
_MARKUP_LANGS = ["markdown", "yaml", "json", "html", "css"]


def _stack_caches() -> dict[str, str]:
    return {lang: _resolve_cache(f"stack_{lang.replace('-', '_')}")
            for lang in _STACK_LANGS}


def _markup_caches() -> dict[str, str]:
    return {lang: _resolve_cache(f"markup_{lang}") for lang in _MARKUP_LANGS}


# Auto-resolve at import (raises FileNotFoundError if tokenization incomplete).
try:
    STACK_LANG_CACHES: dict[str, str] = _stack_caches()
    MARKUP_LANG_CACHES: dict[str, str] = _markup_caches()
except FileNotFoundError as _e:
    STACK_LANG_CACHES = {}
    MARKUP_LANG_CACHES = {}
    print(f"[c5-prod] WARN: {_e}")

# === Aryabumi Table 3 ratios (Stack), top-10 re-normalized ===
STACK_RATIOS = {
    "java": 15.54 / 86.83,
    "javascript": 15.29 / 86.83,
    "php": 12.46 / 86.83,
    "python": 9.60 / 86.83,
    "c-sharp": 8.30 / 86.83,
    "typescript": 7.92 / 86.83,
    "c": 6.63 / 86.83,
    "cpp": 4.91 / 86.83,
    "go": 3.49 / 86.83,
    "ruby": 2.69 / 86.83,
}
# === Aryabumi Table 4 ratios (Markup), top-5 re-normalized ===
MARKUP_RATIOS = {
    "markdown": 54.23 / 90.40,
    "yaml": 10.77 / 90.40,
    "json": 9.97 / 90.40,
    "html": 8.57 / 90.40,
    "css": 6.86 / 90.40,
}

# === Paloma eval components (same as A5/B4) ===
PALOMA_SUBSETS = [
    "4chan",
    "c4_100_domains",
    "c4_en",
    "dolma-v1_5",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
]


def _paloma_components() -> dict[str, DatasetComponent]:
    paloma_dir = Path(f"{BASE_TOKENIZED}/paloma")
    found = {}
    for sub in PALOMA_SUBSETS:
        matches = sorted(paloma_dir.glob(f"{sub}-*"))
        if not matches:
            raise FileNotFoundError(f"Paloma subset '{sub}' not tokenized.")
        found[f"paloma_{sub}"] = DatasetComponent(cache_dir=str(matches[-1]))
    return found


paloma_components = _paloma_components()


def _distributed_from_env() -> DistributedConfig:
    num_proc = os.environ.get("JAX_DIST_NUM_PROCESSES")
    if num_proc is None:
        return DistributedConfig()
    return DistributedConfig(
        num_processes=int(num_proc),
        process_id=int(os.environ["JAX_DIST_PROCESS_ID"]),
        coordinator_address=os.environ["JAX_DIST_COORDINATOR"],
        local_device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    )


# === Batch and step layout — match A5 exactly regardless of node count ===
# A5 was 4 nodes, global batch 256, per_device_parallelism=8.
# C5 on 8 nodes: keep global batch 256, set per_device_parallelism=4 so
# per-step compute and step count are IDENTICAL.
NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
TRAIN_BATCH_SIZE = 256  # fixed across all node counts to match A5/B4
PER_DEVICE_PARALLELISM = max(1, TRAIN_BATCH_SIZE // TOTAL_GPUS)
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096  # 1,048,576
TARGET_TOKENS = 30_770_000_000
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP  # 29,343


# === Stage boundary in SEQUENCE index ===
# MixtureDataset takes start_seq_index (sequence count), not step count.
# seq_per_step = batch * 1 sequence per item = batch = 256
# Stage 2 starts halfway: at step 14672 = seq index 14672 * 256 = 3,756,032
STAGE2_START_STEP = NUM_TRAIN_STEPS // 2  # 14671 (integer division)
STAGE2_START_SEQ = STAGE2_START_STEP * TRAIN_BATCH_SIZE


def _stage1_weights() -> dict[str, float]:
    """80% Stack + 20% markup, distributed per Aryabumi Tables 3/4."""
    w: dict[str, float] = {}
    for lang, r in STACK_RATIOS.items():
        w[f"stack_{lang}"] = 0.80 * r
    for lang, r in MARKUP_RATIOS.items():
        w[f"markup_{lang}"] = 0.20 * r
    # No DCLM in stage 1, no eval-only components.
    for k in [
        *[f"dclm_shard{i}" for i in range(len(DCLM_SHARDS))],
        "dclm_200m_val",
        *[f"paloma_{s}" for s in PALOMA_SUBSETS],
    ]:
        w[k] = 0.0
    return w


def _stage2_weights() -> dict[str, float]:
    """90% DCLM (7 shards equal) + 10% code (same 80/20 Stack+markup split)."""
    w: dict[str, float] = {}
    dclm_share = 0.90 / len(DCLM_SHARDS)
    for i in range(len(DCLM_SHARDS)):
        w[f"dclm_shard{i}"] = dclm_share
    for lang, r in STACK_RATIOS.items():
        w[f"stack_{lang}"] = 0.10 * 0.80 * r
    for lang, r in MARKUP_RATIOS.items():
        w[f"markup_{lang}"] = 0.10 * 0.20 * r
    for k in ["dclm_200m_val", *[f"paloma_{s}" for s in PALOMA_SUBSETS]]:
        w[k] = 0.0
    return w


# === Build LmDataConfig components ===
_dclm_components = {f"dclm_shard{i}": DatasetComponent(cache_dir=p) for i, p in enumerate(DCLM_SHARDS)}
_stack_components = {f"stack_{k}": DatasetComponent(cache_dir=v) for k, v in STACK_LANG_CACHES.items()}
_markup_components = {f"markup_{k}": DatasetComponent(cache_dir=v) for k, v in MARKUP_LANG_CACHES.items()}

if not _stack_components or not _markup_components:
    # Hard fail before launching — caches must be wired before this script can run.
    import warnings

    warnings.warn(
        "STACK_LANG_CACHES and MARKUP_LANG_CACHES are empty. "
        "Fill them in once tokenization completes - see C5_BLOCKER_2026-06-05.md.",
        stacklevel=2,
    )

_shard_val_sizes = {k: 0 for k in {**_dclm_components, **_stack_components, **_markup_components}}

# Levanter's MixtureDataset accepts staged weights as list[(start_seq_index, dict)].
# At seq index 0, use stage1_weights; at STAGE2_START_SEQ, swap to stage2_weights.
data_config = LmDataConfig(
    components={
        **_dclm_components,
        **_stack_components,
        **_markup_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights=[
        (0, _stage1_weights()),
        (STAGE2_START_SEQ, _stage2_weights()),
    ],
    num_validation_sequences=_shard_val_sizes,
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=False,
    shuffle_before_trainval_split=False,
    stop_strategy="restart",
    enforce_eos=True,
)

model_config = model_dict["1_4b4k"]

train_config = TrainLmConfig(
    data=data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["1.4b", "1ep", "c5", "aryabumi-code-then-text", "wd-0.1", f"nodes-{NUM_PROC}", "8node"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_1ep_c5_code_then_text/",
            save_interval=timedelta(minutes=30),
            keep=[{"every": NUM_TRAIN_STEPS // 4}],
        ),
        ray=RayConfig(auto_start_cluster=False),
        distributed=_distributed_from_env(),
        jax_compilation_cache_dir="/fsx/users/dongweij/marin/outputs/jax_compile_cache",
    ),
    model=model_config,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=3e-4,
        weight_decay=0.1,
        lr_schedule="cosine",
        min_lr_ratio=0.0,
        beta1=0.9,
        beta2=0.95,
        warmup=0.01,
        max_grad_norm=1.0,
    ),
    data_seed=0,
)

if __name__ == "__main__":
    print("=== 1.4B 1-epoch C5 (Aryabumi code->text) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  train_batch_size (FIXED): {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(
        f"  num_train_steps: {NUM_TRAIN_STEPS:,}  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B"
    )
    print(f"  stage 2 starts at step {STAGE2_START_STEP} / seq {STAGE2_START_SEQ:,}")
    print("  stage 1 weights:")
    for k, v in _stage1_weights().items():
        if v > 0:
            print(f"    {k}: {v:.4f}")
    print("  stage 2 weights:")
    for k, v in _stage2_weights().items():
        if v > 0:
            print(f"    {k}: {v:.4f}")
    print(f"  LR={3e-4}, WD=0.1, cosine to 0 (continuous across both stages)")
    if not STACK_LANG_CACHES or not MARKUP_LANG_CACHES:
        print("\nERROR: STACK_LANG_CACHES and MARKUP_LANG_CACHES are empty.")
        print("Fill them in after tokenization completes. See C5_BLOCKER_2026-06-05.md.")
        raise SystemExit(1)
    from levanter.main import train_lm

    train_lm.main(train_config)
