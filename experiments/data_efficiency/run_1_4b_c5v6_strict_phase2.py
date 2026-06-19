# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B C5-v6 STRICT PHASE 2: same as run_1_4b_c5v6_phase2.py except phase 2's
# components dict puts code+markup BEFORE dclm so the Feistel shuffle keys
# match phase 1's. This delivers true strict prefix replay (what the original
# c5v6_phase2 docstring already promised). Compare against the existing
# c5v6_phase2 final to isolate strict-replay vs. same-cache-different-shuffle.
#
# Original 1.4B C5-v6 PHASE 2 docstring (preserved for context):
#
# Same recipe as C5-v3 phase 2 (separate fresh-cosine init from C5-v3 phase 1's
# step-14671) EXCEPT the phase 2 code+markup share goes from 10% → 30%. Tests
# whether keeping code circuits "warm" via heavier replay during text training
# preserves code performance AND helps NL elicitation.
#
# Phase 1: REUSE c5v3_phase1_step14671 (no retrain — same code+markup checkpoint).
# Phase 2: 70% DCLM + 30% (80% code + 20% markup) = 70% DCLM + 24% code + 6% markup.
#
# IMPORTANT — STRICT REPLAY, NOT NEW CODE:
# Levanter's MixtureDataset re-indexes each component starting at doc-index 0
# whenever a fresh `MixtureDataset` is built. Because phase 2 uses the same
# `data_seed=0` and the same code+markup caches as phase 1, and because
# `initialize_from_checkpoint_path` loads model weights only (no loader state),
# the per-component doc index at block T is `block_id * counts_per_block`,
# where counts_per_block is proportional to the component's share in that phase.
# Phase 1 (100% code+markup): SE-Python counts_per_block ≈ 1760.
# Phase 2 (30% code+markup):  SE-Python counts_per_block ≈ 532.
# Phase 2's SE-Python doc range [0..14672×532] ⊂ phase 1's [0..14672×1760].
# Same for Nemotron-CC, Nemotron-UA, Stack-Edu-Markdown.
# → Every code+markup token phase 2 sees is a token phase 1 already saw,
#   in the same shuffled order. This is approximately strict replay of the
#   first ~30% of phase 1's code+markup data.
# The "new code" version is C5-v6-NEW (separate script) which uses an
# explicit per-component doc-offset to start phase 2 where phase 1 ended.

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
_TOKENIZED_BASE = Path(BASE_TOKENIZED)

# === Phase 1 final checkpoint (REUSE C5-v3 phase 1) ===
PHASE1_INIT_FROM = "checkpoints/1_4b_c5v3_phase1/8dtdcear/step-14671"


def _resolve_cache(prefix: str) -> str:
    matches = sorted(_TOKENIZED_BASE.glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(f"No tokenized cache for prefix '{prefix}'.")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple tokenized caches match prefix '{prefix}': {[m.name for m in matches]}. "
            f"Pin one explicitly by including the hash, e.g. '{matches[0].name}' instead of bare prefix."
        )
    return str(matches[0])


try:
    SE_PYTHON_CACHE = _resolve_cache("c5v2_stack_edu_python_clean")
    SE_MARKDOWN_CACHE = _resolve_cache("c5v2_stack_edu_markdown_clean")
    NEMOTRON_CC_CACHE = _resolve_cache("c5v2_nemotron_code_concepts")
    NEMOTRON_UA_CACHE = _resolve_cache("c5v2_nemotron_unconditional_algorithmic")
except FileNotFoundError as _e:
    SE_PYTHON_CACHE = SE_MARKDOWN_CACHE = NEMOTRON_CC_CACHE = NEMOTRON_UA_CACHE = ""
    print(f"[c5v6-p2] WARN: {_e}")


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


CODE_RATIOS = {
    "se_python": 8.8 / 16.3,
    "nemotron_cc": 7.3 / 16.3,
    "nemotron_ua": 0.2 / 16.3,
}
MARKUP_RATIOS = {"se_markdown": 1.0}


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


NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
TRAIN_BATCH_SIZE = 256
PER_DEVICE_PARALLELISM = max(1, TRAIN_BATCH_SIZE // TOTAL_GPUS)
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096
NUM_TRAIN_STEPS = 14_672  # Same as C5-v3 phase 2 — 15.39 B tokens


_code_key_to_cache = {
    "se_python": SE_PYTHON_CACHE,
    "nemotron_cc": NEMOTRON_CC_CACHE,
    "nemotron_ua": NEMOTRON_UA_CACHE,
}
_markup_key_to_cache = {"se_markdown": SE_MARKDOWN_CACHE}


def _phase2_weights() -> dict[str, float]:
    """70% DCLM (7 shards equal) + 30% (80% code + 20% markup) = HIGH-replay phase 2."""
    w: dict[str, float] = {}
    dclm_share = 0.70 / len(DCLM_SHARDS)
    for i in range(len(DCLM_SHARDS)):
        w[f"dclm_shard{i}"] = dclm_share
    for k, r in CODE_RATIOS.items():
        w[f"code_{k}"] = 0.30 * 0.80 * r
    for k, r in MARKUP_RATIOS.items():
        w[f"markup_{k}"] = 0.30 * 0.20 * r
    for k in ["dclm_200m_val", *[f"paloma_{s}" for s in PALOMA_SUBSETS]]:
        w[k] = 0.0
    return w


_dclm_components = {f"dclm_shard{i}": DatasetComponent(cache_dir=p) for i, p in enumerate(DCLM_SHARDS)}
_code_components = {f"code_{k}": DatasetComponent(cache_dir=v) for k, v in _code_key_to_cache.items() if v}
_markup_components = {f"markup_{k}": DatasetComponent(cache_dir=v) for k, v in _markup_key_to_cache.items() if v}

_shard_val_sizes = {k: 0 for k in {**_dclm_components, **_code_components, **_markup_components}}

data_config = LmDataConfig(
    components={
        # Code + markup FIRST to match phase-1 insertion order so Levanter's
        # per-component shuffle keys (taken sequentially from key_iter in
        # datasets.py:786) match phase 1, giving true strict prefix replay.
        # Original c5v6_phase2 had DCLM first → different shuffle keys →
        # "same-cache, different-stream" replay, not strict replay.
        **_code_components,
        **_markup_components,
        **_dclm_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights=_phase2_weights(),
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
    initialize_from_checkpoint_path=os.environ.get("C5V3_PHASE1_CKPT", PHASE1_INIT_FROM),
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["1.4b", "c5v6_strict", "phase2-30pct-code-replay", "dclm", "strict-prefix-replay", "wd-0.1", f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_c5v6_strict_phase2/",
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
    print("=== 1.4B C5-v6 PHASE 2 (continued text from code-LM, 30% code+markup replay) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  initialize_from_checkpoint_path: {train_config.initialize_from_checkpoint_path}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  LR=3e-4 (FRESH), WD=0.1, cosine to 0 (warmup 1% ≈ 147 steps)")
    print(f"  data: 70% DCLM + 30% (80% code + 20% markup)")
    if not SE_PYTHON_CACHE or not SE_MARKDOWN_CACHE or not NEMOTRON_CC_CACHE or not NEMOTRON_UA_CACHE:
        print("\nERROR: code caches missing.")
        raise SystemExit(1)
    from levanter.main import train_lm
    train_lm.main(train_config)
