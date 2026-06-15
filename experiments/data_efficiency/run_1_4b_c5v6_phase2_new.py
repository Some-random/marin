# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B C5-v6-NEW PHASE 2: continued text pretraining with NEW (NOT REPLAY) code+markup.
#
# Same data MIX as C5-v6 phase 2 (70% DCLM + 30% code+markup at 80/20 inner split),
# same separate-cosine init from C5-v3 phase 1 step-14671, same total token budget.
# The difference: phase 2's code+markup are DISJOINT from phase 1's, NOT a strict
# prefix replay. This isolates the effect of *new* code data from the effect of
# *replay-style* code-circuit reactivation observed in C5-v6.
#
# Mechanism — Levanter's MixtureDataset re-indexes each component starting at
# sequence-index 0. C5-v6 (replay) shares caches and re-reads phase 1's first ~30%.
# C5-v6-NEW uses a per-component `offset` (added in this PR) to skip past phase 1's
# consumption point:
#
#   Phase 1 consumption per component (sequences @ 4096 tokens; from
#   mixture_block_size=2048, num_blocks=1834, counts_per_block at phase 1 weights):
#     code_se_python (Stack-Edu Python score≥3.0):  1834 × 886 = 1,624,924  (cache ≈ 1,657,408 — FULL)
#     code_nemotron_cc:                              1834 × 733 = 1,344,322  (cache ≈ 1,713,867)
#     code_nemotron_ua:                              1834 ×  20 =    36,680  (cache ≈    46,387)
#     markup_se_markdown:                            1834 × 409 =   750,106  (cache ≈ 2,417,968)
#
#   Since SE-Python cache is essentially exhausted by phase 1, phase 2 uses a
#   FRESH cache (c5v6new_stack_edu_python_low) tokenized from Stack-Edu Python
#   blobs in score-range [2.8, 3.0) (initially fetched [2.7, 3.0); filtered down to [2.8, 3.0) before tokenization per Dongwei's quality concern) — disjoint by construction. offset=0.
#
#   For the other 3 components we reuse the existing caches and set
#   `offset = phase 1 consumption`. With Levanter's RESTART strategy, the
#   slice wraps modulo (cache_len - offset), so phase 1's docs at indices
#   [0, offset) are NEVER revealed to phase 2. Minor (~2%) wrap-around remains
#   for nemotron_cc (cache 1.71 M vs need 1.75 M); negligible weight impact.

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
    # NEW (phase 1 never saw): Stack-Edu Python score in [2.8, 3.0) (initially fetched [2.7, 3.0); filtered down to [2.8, 3.0) before tokenization per Dongwei's quality concern)
    SE_PYTHON_LOW_CACHE = _resolve_cache("c5v6new_stack_edu_python_low")
    # Existing caches — phase 2 uses with explicit offsets
    SE_MARKDOWN_CACHE = _resolve_cache("c5v2_stack_edu_markdown_clean")
    NEMOTRON_CC_CACHE = _resolve_cache("c5v2_nemotron_code_concepts")
    NEMOTRON_UA_CACHE = _resolve_cache("c5v2_nemotron_unconditional_algorithmic")
except FileNotFoundError as _e:
    SE_PYTHON_LOW_CACHE = SE_MARKDOWN_CACHE = NEMOTRON_CC_CACHE = NEMOTRON_UA_CACHE = ""
    print(f"[c5v6new-p2] WARN: {_e}")


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


# Ratios within the 80% code + 20% markup slice. Identical to C5-v6 phase 2.
CODE_RATIOS = {
    "se_python": 8.8 / 16.3,
    "nemotron_cc": 7.3 / 16.3,
    "nemotron_ua": 0.2 / 16.3,
}
MARKUP_RATIOS = {"se_markdown": 1.0}


# Phase 1 (C5-v3 phase 1) sequence-offsets per component. Phase 2 uses these
# `offset`s on the existing caches so it reads docs phase 1 DID NOT see.
# Derivation: mixture_block_size=2048, num_blocks=14672*256/2048=1834, and
# counts_per_block from phase 1's weights (100% code+markup; remainder of 2
# from rounding goes to the largest = se_python which makes its count 886).
PHASE1_COUNTS_PER_BLOCK = {
    "se_python": 886,        # (unused in phase 2 — new cache w/ offset=0)
    "nemotron_cc": 733,
    "nemotron_ua": 20,
    "se_markdown": 409,
}
NUM_BLOCKS_PHASE1 = 1834
PHASE1_OFFSETS = {k: NUM_BLOCKS_PHASE1 * v for k, v in PHASE1_COUNTS_PER_BLOCK.items()}


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
NUM_TRAIN_STEPS = 14_672  # Same as C5-v3/v6 phase 2 — 15.39 B tokens


def _phase2_weights() -> dict[str, float]:
    """70% DCLM (7 shards equal) + 30% (80% code + 20% markup). Identical to C5-v6 phase 2."""
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

# code/markup components: SE-Python uses the NEW cache (no offset); the other 3
# reuse C5-v3 phase 1's caches with offset = phase 1 sequence-consumption.
_code_components = {
    "code_se_python": DatasetComponent(cache_dir=SE_PYTHON_LOW_CACHE, offset=0),
    "code_nemotron_cc": DatasetComponent(cache_dir=NEMOTRON_CC_CACHE, offset=PHASE1_OFFSETS["nemotron_cc"]),
    "code_nemotron_ua": DatasetComponent(cache_dir=NEMOTRON_UA_CACHE, offset=PHASE1_OFFSETS["nemotron_ua"]),
}
_markup_components = {
    "markup_se_markdown": DatasetComponent(cache_dir=SE_MARKDOWN_CACHE, offset=PHASE1_OFFSETS["se_markdown"]),
}

_shard_val_sizes = {k: 0 for k in {**_dclm_components, **_code_components, **_markup_components}}

data_config = LmDataConfig(
    components={
        **_dclm_components,
        **_code_components,
        **_markup_components,
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
            tags=["1.4b", "c5v6new", "phase2-30pct-code-NEW", "dclm", "wd-0.1", f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_c5v6new_phase2/",
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
    print("=== 1.4B C5-v6-NEW PHASE 2 (continued text from code-LM, 30% NEW code+markup) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  initialize_from_checkpoint_path: {train_config.initialize_from_checkpoint_path}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  LR=3e-4 (FRESH), WD=0.1, cosine to 0 (warmup 1% ≈ 147 steps)")
    print(f"  data: 70% DCLM + 30% (80% code + 20% markup) — code+markup are NEW (no overlap with phase 1)")
    print(f"  Phase 1 sequence-offsets applied per component:")
    for k, v in PHASE1_OFFSETS.items():
        print(f"    {k}: offset={v:,}")
    print(f"  SE-Python: fresh cache c5v6new_stack_edu_python_low (score in [2.8, 3.0) (initially fetched [2.7, 3.0); filtered down to [2.8, 3.0) before tokenization per Dongwei's quality concern))")
    print(f"    → {SE_PYTHON_LOW_CACHE}")
    if not SE_PYTHON_LOW_CACHE or not SE_MARKDOWN_CACHE or not NEMOTRON_CC_CACHE or not NEMOTRON_UA_CACHE:
        print("\nERROR: code caches missing.")
        raise SystemExit(1)
    from levanter.main import train_lm
    train_lm.main(train_config)
