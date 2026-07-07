# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Small-scale (300M) C5-v2 continuous-cosine variant. Mirror of run_1_4b_c5v2_clean_code.py
# at 300M / Chinchilla-optimal 6B-token budget.
#
# Single-script run: phase-1 (code+markup) → phase-2 (DCLM + replay) in ONE
# continuous cosine LR schedule across all 22,886 steps (3B phase 1 + 3B phase 2).
# Data weights swap mid-run via LmDataConfig's step-keyed `train_weights` list
# at STAGE2_START_SEQ. Optimizer state is continuous — no fresh cosine reset.
#
# Compare against `run_smallscale_phase2.py` with MODEL_KEY=300m4k +
# PHASE1_CKPT=smallscale_300m4k_code_p1/... (separate cosine, fresh optimizer).
#
# Env vars:
#   MODEL_KEY      — "300m4k" (default) or "600m4k"
#   TEXT_SOURCE    — "dclm" or "sp_nl"
#   REPLAY_FRAC    — fraction code+markup in phase 2 (default 0.30 = C5-v6 mix)
#   TARGET_TOKENS  — total tokens (default 6e9 for 300m4k Chinchilla-optimal)
#   RUN_TAG        — wandb tag + ckpt dir suffix

import os
from datetime import timedelta
from pathlib import Path

_secrets = Path(__file__).resolve().parents[4] / ".secrets"
if _secrets.exists():
    for line in _secrets.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

import json
import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig, DatasetComponent
from levanter.distributed import DistributedConfig, RayConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.reasoning_pretraining.code_ladder.models.models import model_dict

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"
_TOKENIZED_BASE = Path(BASE_TOKENIZED)


def _resolve_cache(prefix: str) -> str:
    matches = sorted(_TOKENIZED_BASE.glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(f"No tokenized cache for prefix '{prefix}'.")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple matches for '{prefix}': {[m.name for m in matches]}")
    return str(matches[0])


# === Env params ===
MODEL_KEY     = os.environ.get("MODEL_KEY", "300m4k")
TEXT_SOURCE   = os.environ.get("TEXT_SOURCE", "dclm")
REPLAY_FRAC   = float(os.environ.get("REPLAY_FRAC", "0.30"))
TARGET_TOKENS = int(float(os.environ.get("TARGET_TOKENS", "6e9")))
RUN_TAG       = os.environ.get("RUN_TAG", f"smallscale_c5v2cont_{MODEL_KEY}")


# === Caches ===
SE_PYTHON_CACHE   = _resolve_cache("c5v2_stack_edu_python_clean")
SE_MARKDOWN_CACHE = _resolve_cache("c5v2_stack_edu_markdown_clean")
NEMOTRON_CC_CACHE = _resolve_cache("c5v2_nemotron_code_concepts")
NEMOTRON_UA_CACHE = _resolve_cache("c5v2_nemotron_unconditional_algorithmic")

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


def _collect_sp_nl_shards_with_rows() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for prefix in ("slimpajama_nl_en", "slimpajama_nl_chunk2_en"):
        try:
            root = Path(_resolve_cache(prefix))
        except FileNotFoundError:
            continue
        ledger = json.loads((root / "train" / "shard_ledger.json").read_text())
        for part_name, rows in sorted(ledger["shard_rows"].items()):
            part_path = root / "train" / part_name
            if not part_path.exists():
                continue
            out.append((str(part_path), int(rows)))
    return out


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
            raise RuntimeError(f"Multiple matches for {sub!r}")
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


# === Code mix proportions (same as C5-v3 phase 1 / C5-v2 stage 1) ===
CODE_RATIOS = {
    "se_python":   8.8 / 16.3,
    "nemotron_cc": 7.3 / 16.3,
    "nemotron_ua": 0.2 / 16.3,
}
MARKUP_RATIOS = {"se_markdown": 1.0}


# === Batch + step layout ===
NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
PER_DEVICE_PARALLELISM = 8
TRAIN_BATCH_SIZE = TOTAL_GPUS * PER_DEVICE_PARALLELISM
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP   # 22,886 for 300m4k @ 6e9

STAGE2_START_STEP = NUM_TRAIN_STEPS // 2             # 11,443
STAGE2_START_SEQ = STAGE2_START_STEP * TRAIN_BATCH_SIZE


# === Build components ===
_code_key_to_cache = {
    "se_python": SE_PYTHON_CACHE,
    "nemotron_cc": NEMOTRON_CC_CACHE,
    "nemotron_ua": NEMOTRON_UA_CACHE,
}
_markup_key_to_cache = {
    "se_markdown": SE_MARKDOWN_CACHE,
}

_code_components = {f"code_{k}": DatasetComponent(cache_dir=v) for k, v in _code_key_to_cache.items()}
_markup_components = {f"markup_{k}": DatasetComponent(cache_dir=v) for k, v in _markup_key_to_cache.items()}

if TEXT_SOURCE == "dclm":
    _text_components = {f"dclm_shard{i}": DatasetComponent(cache_dir=p) for i, p in enumerate(DCLM_SHARDS)}
    text_names = list(_text_components)
    _text_weights_pure = {n: 1.0 / len(text_names) for n in text_names}
elif TEXT_SOURCE == "sp_nl":
    sp_shards = _collect_sp_nl_shards_with_rows()
    sp_total = sum(r for _, r in sp_shards) or 1
    _text_components = {f"sp_nl_shard{i:03d}": DatasetComponent(cache_dir=p) for i, (p, _) in enumerate(sp_shards)}
    text_names = list(_text_components)
    _text_weights_pure = {f"sp_nl_shard{i:03d}": r / sp_total for i, (_, r) in enumerate(sp_shards)}
else:
    raise ValueError(f"unknown TEXT_SOURCE: {TEXT_SOURCE!r}")


def _stage1_weights() -> dict[str, float]:
    """100% code+markup. text + eval-only get weight 0."""
    w: dict[str, float] = {}
    for k, r in CODE_RATIOS.items():
        w[f"code_{k}"] = 0.80 * r
    for k, r in MARKUP_RATIOS.items():
        w[f"markup_{k}"] = 0.20 * r
    for n in text_names:
        w[n] = 0.0
    w["dclm_200m_val"] = 0.0
    for s in PALOMA_SUBSETS:
        w[f"paloma_{s}"] = 0.0
    return w


def _stage2_weights() -> dict[str, float]:
    """(1-REPLAY_FRAC) text + REPLAY_FRAC × (80% code + 20% markup)."""
    w: dict[str, float] = {}
    text_share = 1.0 - REPLAY_FRAC
    for n, p in _text_weights_pure.items():
        w[n] = text_share * p
    for k, r in CODE_RATIOS.items():
        w[f"code_{k}"] = REPLAY_FRAC * 0.80 * r
    for k, r in MARKUP_RATIOS.items():
        w[f"markup_{k}"] = REPLAY_FRAC * 0.20 * r
    w["dclm_200m_val"] = 0.0
    for s in PALOMA_SUBSETS:
        w[f"paloma_{s}"] = 0.0
    return w


data_config = LmDataConfig(
    components={
        **_text_components,
        **_code_components,
        **_markup_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights=[
        (0, _stage1_weights()),
        (STAGE2_START_SEQ, _stage2_weights()),
    ],
    num_validation_sequences={k: 0 for k in {**_text_components, **_code_components, **_markup_components}},
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=False,
    shuffle_before_trainval_split=False,
    stop_strategy="restart",
    enforce_eos=True,
)

model_config = model_dict[MODEL_KEY]

train_config = TrainLmConfig(
    data=data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency-smallscale",
            entity="dongwei_jiang",
            tags=["smallscale", MODEL_KEY, "c5v2-continuous-cosine", TEXT_SOURCE,
                  f"replay{int(REPLAY_FRAC*100)}", RUN_TAG, f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path=f"checkpoints/smallscale_{MODEL_KEY}_c5v2cont_{TEXT_SOURCE}_r{int(REPLAY_FRAC*100)}/",
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
    print(f"=== smallscale C5-v2 continuous cosine: {MODEL_KEY} on {TEXT_SOURCE} ({TARGET_TOKENS/1e9:.2f} B tokens) ===")
    print(f"  stage 2 starts at step {STAGE2_START_STEP} / seq {STAGE2_START_SEQ:,}")
    print(f"  stage 1 weights ({len(_stage1_weights())} keys, 100% code+markup):")
    for k, v in _stage1_weights().items():
        if v > 0:
            print(f"    {k:35s} {v:.4f}")
    print(f"  stage 2 weights ({1-REPLAY_FRAC:.0%} text + {REPLAY_FRAC:.0%} code+markup):")
    for k, v in _stage2_weights().items():
        if v > 0:
            print(f"    {k:35s} {v:.4f}")
    print(f"  LR=3e-4 cosine→0 (CONTINUOUS across both stages), warmup=1%, wd=0.1")
    print(f"  nodes={NUM_PROC} batch={TRAIN_BATCH_SIZE} steps={NUM_TRAIN_STEPS:,}")
    from levanter.main import train_lm
    train_lm.main(train_config)
