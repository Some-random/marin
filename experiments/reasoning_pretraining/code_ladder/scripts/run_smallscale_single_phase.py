# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Small-scale single-phase (A5 / A5-SP) at Chinchilla-optimal budget.
#
# Env vars (REQUIRED):
#   MODEL_KEY     — "300m4k" or "600m4k"
#   TEXT_SOURCE   — "dclm" or "sp_nl"
#   TARGET_TOKENS — total trained tokens (e.g. 6e9 for 300M, 12e9 for 600M)
#   RUN_TAG       — used for wandb + checkpoint dir
#
# Hparams identical to A5 / A5-SP otherwise: batch=TOTAL_GPUS*8 × seq 4096,
# LR=3e-4 cosine→0 (warmup 1%), wd=0.1, β=(0.9,0.95), max_grad_norm=1.0.

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
MODEL_KEY     = os.environ["MODEL_KEY"]              # "300m4k" / "600m4k"
TEXT_SOURCE   = os.environ["TEXT_SOURCE"]            # "dclm" / "sp_nl"
TARGET_TOKENS = int(float(os.environ["TARGET_TOKENS"]))
RUN_TAG       = os.environ["RUN_TAG"]


# === Text source caches ===
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


# === Paloma + dclm_200m_val for held-out tracking ===
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


NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
PER_DEVICE_PARALLELISM = 8
TRAIN_BATCH_SIZE = TOTAL_GPUS * PER_DEVICE_PARALLELISM
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP


# === Build train components + weights ===
if TEXT_SOURCE == "dclm":
    _dclm_components = {f"dclm_shard{i}": DatasetComponent(cache_dir=p) for i, p in enumerate(DCLM_SHARDS)}
    text_components = _dclm_components
    text_weights = {f"dclm_shard{i}": 1.0 / len(DCLM_SHARDS) for i in range(len(DCLM_SHARDS))}
elif TEXT_SOURCE == "sp_nl":
    sp_shards = _collect_sp_nl_shards_with_rows()
    sp_total = sum(r for _, r in sp_shards) or 1
    text_components = {f"sp_nl_shard{i:03d}": DatasetComponent(cache_dir=p) for i, (p, _) in enumerate(sp_shards)}
    text_weights = {f"sp_nl_shard{i:03d}": r / sp_total for i, (_, r) in enumerate(sp_shards)}
else:
    raise ValueError(f"unknown TEXT_SOURCE: {TEXT_SOURCE!r}")

train_weights = {
    **text_weights,
    "dclm_200m_val": 0.0,
    **{f"paloma_{s}": 0.0 for s in PALOMA_SUBSETS},
}

data_config = LmDataConfig(
    components={
        **text_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights=train_weights,
    num_validation_sequences={k: 0 for k in text_components},
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
            tags=["smallscale", MODEL_KEY, TEXT_SOURCE, "single-phase", RUN_TAG, f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path=f"checkpoints/smallscale_{MODEL_KEY}_{TEXT_SOURCE}/",
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
    print(f"=== smallscale single-phase: {MODEL_KEY} on {TEXT_SOURCE} ({TARGET_TOKENS/1e9:.2f} B tokens) ===")
    print(f"  nodes={NUM_PROC} batch={TRAIN_BATCH_SIZE} steps={NUM_TRAIN_STEPS:,}")
    print(f"  tokens trained: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f} B")
    from levanter.main import train_lm
    train_lm.main(train_config)
