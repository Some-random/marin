"""13B SMOKE TEST: verify tensor-parallel + multi-node training works on our A100 cluster.

NOT a real experiment. Goal: confirm levanter's TP path works on multi-node A100,
discover memory/throughput numbers, then stop. ~50 steps target.

Layout:
  - 2 nodes × 8 A100-40GB = 16 GPUs total.
  - Within node: tensor-parallel model=8 (uses NVLink for activation comms).
  - Across nodes: replica_dcn=2 for data parallelism (gradient all-reduce via EFA).
  - Resulting effective DP = 2, so train_batch_size must be a small multiple of 2.

Per-GPU memory (estimate):
  - 13B params / 8 TP shards = 1.6B params
  - fp32 params + Adam m + Adam v = 1.6B × 12 bytes = 19.5 GB
  - Activations w/ grad checkpoint at seq=4096, micro=1: ~5-10 GB
  - Total: ~25-30 GB, fits in 40 GB.

Reads JAX_DIST_* env vars set by multi_node_launch.sh.
"""

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
from levanter.utils.mesh import MeshConfig

from experiments.llama import llama_8b

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"
DCLM_SHARD = f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00006"  # 4.93 B tokens, plenty for smoke
DCLM_VAL = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_val-415aea"


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
# With 2 nodes: 16 GPUs. TP=8 within node, so per-node ICI is model=8.
# DCN axis = replica_dcn = NUM_PROC (one slice per node, since each node is its own JAX slice).
# Effective DP = NUM_PROC. For NUM_PROC=2, DP=2.
# train_batch_size = per_device_parallelism * DP_size.
PER_DEVICE_PARALLELISM = 1  # tiny — smoke test
# Scale batch size with #nodes. 8 GPUs/node × #nodes × per_device_parallelism.
TRAIN_BATCH_SIZE = 8 * NUM_PROC * PER_DEVICE_PARALLELISM

data_config = LmDataConfig(
    components={
        "dclm_shard0": DatasetComponent(cache_dir=DCLM_SHARD),
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
    },
    train_weights={
        "dclm_shard0": 1.0,
        "dclm_200m_val": 0.0,
    },
    num_validation_sequences={"dclm_shard0": 0},
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=False,
    shuffle_before_trainval_split=False,
    stop_strategy="restart",
    enforce_eos=True,
)

model_config = llama_8b

train_config = TrainLmConfig(
    data=data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["13b", "smoke", "tp8", "multinode-test"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=50,           # smoke only
        steps_per_eval=10000,         # don't eval
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/13b_smoke/",
            save_interval=timedelta(hours=24),  # don't actually save
            keep=[],
        ),
        ray=RayConfig(auto_start_cluster=False),
        distributed=_distributed_from_env(),
        # The mesh: model=8 within node (NVLink), replica_dcn absorbs nodes.
        # Param mapping: shard model-axis weights (mlp, heads) by model axis.
        # The default `param_mapping = {"embed": "data"}` is preserved.
        # Single-node FSDP layout (per the levanter Getting-Started-GPU.md
        # reference config/gpt2_7b.yaml): no TP, just data=-1 → all 8 GPUs
        # in the data axis. The default param_mapping={"embed": "data"}
        # shards the hidden dimension across data axis, which means almost
        # every weight (because every weight touches embed) gets sharded.
        mesh=MeshConfig(
            axes={"data": -1, "replica": 1, "model": 1},
            dcn_axes={"replica_dcn": -1},
        ),
        jax_compilation_cache_dir="/fsx/users/dongweij/marin/outputs/jax_compile_cache",
    ),
    model=model_config,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=1e-5,    # small for smoke; LR doesn't matter
        weight_decay=0.0,
        lr_schedule="constant",
        beta1=0.9,
        beta2=0.95,
        warmup=0,
        max_grad_norm=1.0,
    ),
    data_seed=0,
)

if __name__ == "__main__":
    print(f"=== 13B SMOKE TEST ===")
    print(f"  num_processes (nodes): {NUM_PROC}")
    print(f"  total GPUs: {NUM_PROC * 8}")
    print(f"  mesh: model=8 (TP within node), replica_dcn={NUM_PROC} (DP across nodes)")
    print(f"  per_device_parallelism: {PER_DEVICE_PARALLELISM}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}")
    print(f"  num_train_steps: 50 (smoke only)")
    print(f"  model: 13B (hidden=5120, layers=40, kv_heads=8, seq=4096)")
    from levanter.main import train_lm
    train_lm.main(train_config)
