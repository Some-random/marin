# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B code25b_clean PHASE 2: continued text pretraining over the curated-code base.
#
# Tests the "code-as-reasoning-prior" hypothesis: train extensively on curated
# code (24.65 B in code25b_clean phase 1), then continue with a SMALL text
# budget — does the strong code prior survive while NL/reasoning emerges?
#
# Phase 1: REUSE code25b_clean step-23511 (24.65 B at threshold ≥2.7 = 5 SE-Py
#   bands + Nemotron-CC + Nemotron-UA + Stack-Edu Markdown, 80% code + 20% markup).
# Phase 2: 70% DCLM + 30% (80% code + 20% markup), code+markup at the SAME mix
#   as phase 1 so replay re-activates the same circuits.
# Recipe: separate cosine 3e-4 → 0 over 4,768 steps = 5.0 B tokens.
#   Total compute = phase 1 (24.65 B) + phase 2 (5.0 B) ≈ 29.65 B — close to A5's
#   30.77 B matched-compute, but with the code-prior shape inverted (code-heavy
#   then text-light, vs A5's text-only).
#
# DCLM text source (NOT SP-NL) because Dongwei picked DCLM for direct
# accordance with the prior C5-v6 30%-replay-DCLM finding.
#
# Replay structure:
# 30% code+markup at code25b_clean's exact 5-band SE-Python + Nemotron + Stack-
# Edu Markdown mix, row-proportional within slots. Same shuffle/data_seed as
# phase 1, so replay is approximately the first ~30%-worth of phase 1's
# (already-shuffled) code+markup tokens — strict replay, not new code.

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
PHASE1_INIT_FROM = "checkpoints/1_4b_code25b_clean/1k836xz4/step-23511"


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
    SE_PYTHON_CLEAN_CACHE = _resolve_cache("c5v2_stack_edu_python_clean")  # ≥3.0
    SE_PYTHON_LOW_CACHE = _resolve_cache("c5v6new_stack_edu_python_low")    # [2.8, 3.0)
    SE_PYTHON_MID_CACHE = _resolve_cache("c5_25b_se_python_mid")            # [2.7, 2.8)
    SE_MARKDOWN_CACHE = _resolve_cache("c5v2_stack_edu_markdown_clean")
    NEMOTRON_CC_CACHE = _resolve_cache("c5v2_nemotron_code_concepts")
    NEMOTRON_UA_CACHE = _resolve_cache("c5v2_nemotron_unconditional_algorithmic")
except FileNotFoundError as _e:
    SE_PYTHON_CLEAN_CACHE = SE_PYTHON_LOW_CACHE = SE_PYTHON_MID_CACHE = ""
    SE_MARKDOWN_CACHE = NEMOTRON_CC_CACHE = NEMOTRON_UA_CACHE = ""
    print(f"[code25b_clean-p2] WARN: {_e}")


# === Row counts for token-proportional weighting within the 5-band code slot ===
# Matches code25b_clean's training mix exactly.
import json as _json
def _rows(cache: str) -> int:
    return int(_json.loads((Path(cache) / "train" / "shard_ledger.json").read_text())["total_num_rows"])

if SE_PYTHON_CLEAN_CACHE:
    _CODE_ROWS = {
        "se_python_clean": _rows(SE_PYTHON_CLEAN_CACHE),
        "se_python_low":   _rows(SE_PYTHON_LOW_CACHE),
        "se_python_mid":   _rows(SE_PYTHON_MID_CACHE),
        "nemotron_cc":     _rows(NEMOTRON_CC_CACHE),
        "nemotron_ua":     _rows(NEMOTRON_UA_CACHE),
    }
    _TOTAL_CODE_ROWS = sum(_CODE_ROWS.values()) or 1
else:
    _CODE_ROWS = {}
    _TOTAL_CODE_ROWS = 1


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


# CODE_RATIOS: 5-band threshold-≥2.7 mix matching code25b_clean phase 1 exactly,
# row-proportional within the code slot (yields token-proportional sampling).
CODE_RATIOS = {k: _CODE_ROWS.get(k, 0) / _TOTAL_CODE_ROWS for k in (
    "se_python_clean", "se_python_low", "se_python_mid", "nemotron_cc", "nemotron_ua"
)}
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
NUM_TRAIN_STEPS = 4_768  # ~5.0 B tokens — small text-continuation budget over the code25b_clean base


_code_key_to_cache = {
    "se_python_clean": SE_PYTHON_CLEAN_CACHE,
    "se_python_low":   SE_PYTHON_LOW_CACHE,
    "se_python_mid":   SE_PYTHON_MID_CACHE,
    "nemotron_cc":     NEMOTRON_CC_CACHE,
    "nemotron_ua":     NEMOTRON_UA_CACHE,
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
        # Code + markup FIRST to match phase-1 insertion order so the Feistel
        # shuffle keys line up and replay is strict prefix, not new draw.
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
    initialize_from_checkpoint_path=os.environ.get("CODE25B_CLEAN_BASE_CKPT", PHASE1_INIT_FROM),
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["1.4b", "code25b_clean", "phase2-30pct-replay-DCLM", "code-as-prior", "5B-text", "wd-0.1", f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_code25b_clean_p2/",
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
    print("=== 1.4B code25b_clean PHASE 2 (5B-text continuation over curated code base, 30% replay + DCLM) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  initialize_from_checkpoint_path: {train_config.initialize_from_checkpoint_path}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  LR=3e-4 (FRESH), WD=0.1, cosine to 0 (warmup 1% ≈ 47 steps)")
    print(f"  data: 70% DCLM + 30% (80% code + 20% markup, code mix matches code25b_clean phase 1)")
    print(f"  code mix (within 80% slot, row-proportional):")
    for k, r in CODE_RATIOS.items():
        print(f"    {k:<22} w={r*100:>5.2f}%")
    if not SE_PYTHON_CLEAN_CACHE or not SE_PYTHON_LOW_CACHE or not SE_PYTHON_MID_CACHE \
       or not SE_MARKDOWN_CACHE or not NEMOTRON_CC_CACHE or not NEMOTRON_UA_CACHE:
        print("\nERROR: caches missing.")
        raise SystemExit(1)
    from levanter.main import train_lm
    train_lm.main(train_config)
