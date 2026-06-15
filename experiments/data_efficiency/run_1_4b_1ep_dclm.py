# 1.4B model, 1 epoch over ~28B DCLM tokens (baseline for the 1-epoch
# comparison series). Reads multi-node env vars (JAX_DIST_*) if set by
# multi_node_launch.sh, otherwise runs single-node.
#
# Hyperparameter design rationale (locked in 2026-06-01 from open-source
# 1.3-7B-class references):
#   - WD=0.1: phi-1/phi-1.5/OLMo-2/Marin all use 0.1. Our prior WD=1.6 was a
#     repetition-overfit hack for 16-epoch DCLM; at 1 epoch it's unnecessary.
#   - LR=3e-4: matches OLMo 2 7B (1.4B is between phi-1.5's 2e-4 and OLMo 2's
#     3e-4; pick higher).
#   - cosine schedule to min_lr_ratio=0.0 (same as Konwoo / our baseline).
#   - β=(0.9, 0.95) AdamW (our baseline + phi).
#   - max_grad_norm=1.0.
#
# Data:
#   - Source: outputs/tokenized/dclm_baseline-0206f1 (canonical marin DCLM,
#     verified 35.5B tokens across 7 full + 1 partial shard, llama-3 vocab).
#   - 1 epoch over ~28B subset (first 6 shards = ~29.8B tokens).
#   - DCLM_VAL: same dclm_200m_val we've been using for in-domain held-out
#     overfit monitoring.

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
# Canonical marin DCLM tokenization. The top-level cache wasn't finalized
# (missing the merged ledger), but each per-shard subdir is a self-contained
# mini-cache (with a `train -> .` symlink). Wire all 7 full shards as separate
# data components. 7 × ~5B = 34.85B unique tokens available.
DCLM_SHARDS = [
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00006",  # 4.93B
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00020",  # 4.95B
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00026",  # 5.04B
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00035",  # 5.00B
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00042",  # 4.94B
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00047",  # 5.00B
    f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00071",  # 4.99B
]
DCLM_VAL = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_val-415aea"

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
    """Build DistributedConfig from JAX_DIST_* env vars set by multi_node_launch.sh."""
    num_proc = os.environ.get("JAX_DIST_NUM_PROCESSES")
    if num_proc is None:
        return DistributedConfig()
    return DistributedConfig(
        num_processes=int(num_proc),
        process_id=int(os.environ["JAX_DIST_PROCESS_ID"]),
        coordinator_address=os.environ["JAX_DIST_COORDINATOR"],
        local_device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    )


# Batch and step count must be divisible by total #GPUs. Total = num_processes × 8.
NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
# Aim for 28B trained tokens. seq_len=4096. tokens_per_step = batch * 4096.
# batch must be ≥ TOTAL_GPUS * per_device_parallelism (8). So min batch on
# 2-node = 128, 4-node = 256. Steps = 28e9 / (batch * 4096).
PER_DEVICE_PARALLELISM = 8
TRAIN_BATCH_SIZE = TOTAL_GPUS * PER_DEVICE_PARALLELISM  # 64 single-node, 128 2-node, 256 4-node
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096
# 30.77B target so that the matched code-mix variant uses 1 epoch over all 3
# code sources (5.4B aryabumi_synth + 1.35B aryabumi_web + 0.94B opc = 7.69B)
# at 25%, with 23.08B DCLM (1 epoch from the 35.5B available).
TARGET_TOKENS = 30_770_000_000
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP


_dclm_components = {f"dclm_shard{i}": DatasetComponent(cache_dir=p) for i, p in enumerate(DCLM_SHARDS)}
_dclm_weights = {k: 1.0 / len(DCLM_SHARDS) for k in _dclm_components}
# Limit per-component validation slice so eval is fast. dclm_200m_val and
# paloma_* are the actual held-out eval signal (small + correctly separate).
# The DCLM shards have train==validation (via symlink), so we slice to 0 to
# disable their per-component eval contribution.
_shard_val_sizes = {k: 0 for k in _dclm_components}

data_config = LmDataConfig(
    components={
        **_dclm_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights={
        **_dclm_weights,
        "dclm_200m_val": 0.0,
        **{k: 0.0 for k in paloma_components},
    },
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
            tags=["1.4b", "1ep", "baseline", "dclm-only", "wd-0.1", f"nodes-{NUM_PROC}"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,  # 8 evals over the run
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_1ep_dclm/",
            save_interval=timedelta(minutes=30),
            keep=[{"every": NUM_TRAIN_STEPS // 4}],  # keep ~3 mid-training checkpoints
        ),
        ray=RayConfig(auto_start_cluster=False),
        distributed=_distributed_from_env(),
        # Persist XLA compile result so re-runs/restarts skip the long compile
        # (4-node 1.4B compile takes ~10 min; 8-node likely longer).
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
    print(f"=== 1.4B 1-epoch DCLM baseline ===")
    print(f"  num_processes (nodes): {NUM_PROC}")
    print(f"  total GPUs: {TOTAL_GPUS}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}")
    print(f"  tokens_per_step: {TOKENS_PER_STEP:,}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}")
    print(f"  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  LR={3e-4}, WD=0.1, cosine to 0")
    from levanter.main import train_lm
    train_lm.main(train_config)
