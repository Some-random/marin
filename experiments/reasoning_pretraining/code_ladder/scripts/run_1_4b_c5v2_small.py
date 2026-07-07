# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B C5-v2 SMALL: 1/10 budget of C5-v2 production.
#
# SAME data sources, SAME 80/20 + 90/10 recipe structure as run_1_4b_c5v2_clean_code.py.
# ONLY differences:
#   - batch=64 (vs 256) — fits single-node, matches base x16 / code25 v2 / 4B
#   - 12,800 steps (vs 29,343) — 3.36 B trained tokens total
#   - Stage 2 starts at step 6,400 (vs 14,672)
#   - LR cosine still 3e-4 → 0 across the reduced step count
#   - 1 node × 8 GPUs × per_device=8 (matches base x16 / code25 v2 hparams)
#
# Goal: probe whether clean-code (Stack-Edu + Nemotron) still beats raw-code
# (StarCoderData) at 10× smaller training budget.
#   - Direct A/B against 1.4B base x16 (DCLM-only @ 3.36 B trained)
#   - Direct A/B against 1.4B code25 v2 (DCLM + 25% opc_algorithmic @ 3.36 B)
#   - C5-v2 full = 30.77 B; this = 3.36 B (1/9.2)

import os
from datetime import timedelta
from pathlib import Path

_secrets = Path(__file__).resolve().parents[4] / ".secrets"
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

from experiments.reasoning_pretraining.code_ladder.models.models import model_dict

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"
_TOKENIZED_BASE = Path(BASE_TOKENIZED)


def _resolve_cache(prefix: str) -> str:
    matches = sorted(_TOKENIZED_BASE.glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(
            f"No tokenized cache for prefix '{prefix}'. "
            f"Run `MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python "
            f"-m experiments.reasoning_pretraining.code_ladder.data.code_data_c5v2` first."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple tokenized caches match prefix '{prefix}': {[m.name for m in matches]}. "
            f"Pin one explicitly by including the hash, e.g. '{matches[0].name}' instead of bare prefix."
        )
    return str(matches[0])


# === Code (clean) and Markup (clean) — same caches as C5-v2 production ===
try:
    SE_PYTHON_CACHE = _resolve_cache("c5v2_stack_edu_python_clean")
    SE_MARKDOWN_CACHE = _resolve_cache("c5v2_stack_edu_markdown_clean")
    NEMOTRON_CC_CACHE = _resolve_cache("c5v2_nemotron_code_concepts")
    NEMOTRON_UA_CACHE = _resolve_cache("c5v2_nemotron_unconditional_algorithmic")
except FileNotFoundError as _e:
    SE_PYTHON_CACHE = ""
    SE_MARKDOWN_CACHE = ""
    NEMOTRON_CC_CACHE = ""
    NEMOTRON_UA_CACHE = ""
    print(f"[c5v2-small] WARN: {_e}")


# === DCLM (text, stage 2) — same 7 shards as A5/B4/C5/C5-v2 ===
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


# === Code mix proportions within the 80% code slot (same as C5-v2 production) ===
CODE_RATIOS = {
    "se_python": 8.8 / 16.3,             # ~54.0%
    "nemotron_cc": 7.3 / 16.3,           # ~44.8%
    "nemotron_ua": 0.2 / 16.3,           # ~1.2%
}
MARKUP_RATIOS = {"se_markdown": 1.0}


# === Paloma eval components (same as C5-v2) ===
PALOMA_SUBSETS = [
    "4chan", "c4_100_domains", "c4_en", "dolma-v1_5",
    "dolma_100_programing_languages", "dolma_100_subreddits",
    "falcon-refinedweb", "gab", "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit", "manosphere_meta_sep", "mc4",
    "ptb", "redpajama", "twitterAAE_HELM_fixed", "wikitext_103",
]


def _paloma_components() -> dict[str, DatasetComponent]:
    paloma_dir = Path(f"{BASE_TOKENIZED}/paloma")
    found = {}
    for sub in PALOMA_SUBSETS:
        matches = sorted(paloma_dir.glob(f"{sub}-*"))
        if not matches:
            raise FileNotFoundError(f"Paloma subset '{sub}' not tokenized.")
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple caches match {sub!r}: " + str([p.name for p in matches]) + ". "
                "Pin one explicitly (include hash in prefix)."
            )
        found[f"paloma_{sub}"] = DatasetComponent(cache_dir=str(matches[0]))
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


# === Batch and step layout — small-budget single-node ===
NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
TRAIN_BATCH_SIZE = 64  # matches base x16 / code25 v2 / 4B exactly
PER_DEVICE_PARALLELISM = max(1, TRAIN_BATCH_SIZE // TOTAL_GPUS)
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096  # 262,144
TARGET_TOKENS = 3_360_000_000              # 3.36 B (= 12,800 steps × 262,144)
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP  # 12,816 → round down to 12,800
NUM_TRAIN_STEPS = 12_800

STAGE2_START_STEP = NUM_TRAIN_STEPS // 2  # 6,400
STAGE2_START_SEQ = STAGE2_START_STEP * TRAIN_BATCH_SIZE


_code_key_to_cache = {
    "se_python": SE_PYTHON_CACHE,
    "nemotron_cc": NEMOTRON_CC_CACHE,
    "nemotron_ua": NEMOTRON_UA_CACHE,
}
_markup_key_to_cache = {
    "se_markdown": SE_MARKDOWN_CACHE,
}


def _stage1_weights() -> dict[str, float]:
    """80% code + 20% markup, token-proportional within each slot."""
    w: dict[str, float] = {}
    for k, r in CODE_RATIOS.items():
        w[f"code_{k}"] = 0.80 * r
    for k, r in MARKUP_RATIOS.items():
        w[f"markup_{k}"] = 0.20 * r
    for k in [
        *[f"dclm_shard{i}" for i in range(len(DCLM_SHARDS))],
        "dclm_200m_val",
        *[f"paloma_{s}" for s in PALOMA_SUBSETS],
    ]:
        w[k] = 0.0
    return w


def _stage2_weights() -> dict[str, float]:
    """90% DCLM (7 shards equal) + 10% (80% code + 20% markup)."""
    w: dict[str, float] = {}
    dclm_share = 0.90 / len(DCLM_SHARDS)
    for i in range(len(DCLM_SHARDS)):
        w[f"dclm_shard{i}"] = dclm_share
    for k, r in CODE_RATIOS.items():
        w[f"code_{k}"] = 0.10 * 0.80 * r
    for k, r in MARKUP_RATIOS.items():
        w[f"markup_{k}"] = 0.10 * 0.20 * r
    for k in ["dclm_200m_val", *[f"paloma_{s}" for s in PALOMA_SUBSETS]]:
        w[k] = 0.0
    return w


# === Build LmDataConfig components ===
_dclm_components = {f"dclm_shard{i}": DatasetComponent(cache_dir=p) for i, p in enumerate(DCLM_SHARDS)}
_code_components = {f"code_{k}": DatasetComponent(cache_dir=v) for k, v in _code_key_to_cache.items() if v}
_markup_components = {f"markup_{k}": DatasetComponent(cache_dir=v) for k, v in _markup_key_to_cache.items() if v}

_shard_val_sizes = {k: 0 for k in {**_dclm_components, **_code_components, **_markup_components}}

data_config = LmDataConfig(
    components={
        **_dclm_components,
        **_code_components,
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
            tags=["1.4b", "c5v2-small", "1/10-budget", "clean-code-then-text", "wd-0.1", f"nodes-{NUM_PROC}"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_c5v2_small/",
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
    print("=== 1.4B C5-v2 SMALL (1/10 budget) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  stage 2 starts at step {STAGE2_START_STEP} / seq {STAGE2_START_SEQ:,}")
    print(f"  LR=3e-4, WD=0.1, cosine to 0 (continuous across both stages)")
    print()
    print("  caches:")
    print(f"    SE_PYTHON   : {SE_PYTHON_CACHE or '(MISSING)'}")
    print(f"    SE_MARKDOWN : {SE_MARKDOWN_CACHE or '(MISSING)'}")
    print(f"    NEMOTRON_CC : {NEMOTRON_CC_CACHE or '(MISSING)'}")
    print(f"    NEMOTRON_UA : {NEMOTRON_UA_CACHE or '(MISSING)'}")
    if not SE_PYTHON_CACHE or not SE_MARKDOWN_CACHE or not NEMOTRON_CC_CACHE or not NEMOTRON_UA_CACHE:
        print("\nERROR: caches missing. Run code_data_c5v2 tokenize first.")
        raise SystemExit(1)
    from levanter.main import train_lm
    train_lm.main(train_config)
