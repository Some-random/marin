"""8B DCLM-only overnight run — 12h budget, ~3.5-4B tokens.

WHY 8B instead of the originally-planned 4B:
  - llama_3_5b (hidden_dim=2560) hit a JAX SPMD 'Involuntary full
    rematerialization' that caused multi-node first-step hangs (8-node
    smokes at b=64, b=128, b=256 all timed out at the JAX shutdown
    barrier before completing step 1). 1-node 3.5B ran at 133 s/it which
    is 60× slower than 1-node 8B's 2 s/it — sharding pathology.
  - llama_8b (hidden_dim=4096) shards cleanly: confirmed 8-node smoke
    ran 50 steps at 2.9 s/it steady state (90k tok/s on 8 nodes).
  - Compromise: '4B undertrained' becomes '8B more-undertrained'
    (TPP ~0.5 vs 0.9). Same qualitative comparison.

Layout: 8 × p4d-24xlarge nodes (64 × A100-40GB).
  - Pure FSDP via embed-axis sharding (model_axis_size=1, data=-1).
  - Same hyperparams as A5 1ep DCLM (wd=0.1, LR=3e-4 cosine to 0).
  - Token budget: TARGET_TOKENS=3.5B → ~12h wall at 90k tok/s.
  - WANDB offline (online flooded socket errors in earlier attempts).
"""

import os
# HARD-DISABLE wandb BEFORE any levanter/wandb import — multi-node runs were
# hitting BrokenPipe / barrier timeouts because rank-0 wandb-online couldn't
# connect cleanly. Setting these here, before imports, prevents wandb init.
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

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
from levanter.tracker import NoopConfig
from levanter.distributed import RayConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.llama import llama_3_5b

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"
# Same 7-shard set A5 used.
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

# Batch sizing.
# A5 used batch=256 × seq=4096 = ~1M tokens/step on 4 nodes.
# At 8 nodes with 64 GPUs and 4B model, batch=128 (per_device=2) = 524k
# tokens/step. We expect ~3-4 s/step → ~150k tok/s → ~6.5B tokens in 12h.
# If OOM, drop PER_DEVICE_PARALLELISM to 1 (batch=64) and accept ~4B tokens.
PER_DEVICE_PARALLELISM = 1
TRAIN_BATCH_SIZE = TOTAL_GPUS * PER_DEVICE_PARALLELISM    # 64 on 8 nodes
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096                 # 262 144 tokens/step
TARGET_TOKENS = 6_000_000_000
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP

paloma_components = _paloma_components()

_dclm_components = {f"dclm_shard{i}": DatasetComponent(cache_dir=p) for i, p in enumerate(DCLM_SHARDS)}
_dclm_weights = {k: 1.0 / len(DCLM_SHARDS) for k in _dclm_components}
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

model_config = llama_3_5b

train_config = TrainLmConfig(
    data=data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=NoopConfig(),  # wandb-disabled — was breaking multi-node coordination
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 6,  # 6 evals
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/8b_dclm_short/",
            save_interval=timedelta(hours=2),  # keep a few intermediate
            keep=[{"every": NUM_TRAIN_STEPS // 3}],
        ),
        ray=RayConfig(auto_start_cluster=False),
        distributed=_distributed_from_env(),
        # Same FSDP-via-embed-axis layout the smoke verified.
        mesh=MeshConfig(
            axes={"data": -1, "replica": 1, "model": 1},
            dcn_axes={"replica_dcn": -1},
        ),
        jax_compilation_cache_dir="/fsx/users/dongweij/marin/outputs/jax_compile_cache",
    ),
    model=model_config,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=3e-4,    # match A5
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
    print(f"=== 8B DCLM short run (overnight 12h) ===")
    print(f"  num_processes (nodes): {NUM_PROC}")
    print(f"  total GPUs: {TOTAL_GPUS}")
    print(f"  per_device_parallelism: {PER_DEVICE_PARALLELISM}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}")
    print(f"  tokens/step: {TOKENS_PER_STEP:,}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}")
    print(f"  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  LR={3e-4}, WD=0.1, cosine to 0 (matches A5)")
    from levanter.main import train_lm
    train_lm.main(train_config)
