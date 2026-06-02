# Smoke test for multi-node training setup.
#
# Reads JAX_DIST_NUM_PROCESSES / JAX_DIST_PROCESS_ID / JAX_DIST_COORDINATOR
# env vars (set by multi_node_launch.sh). Trains the 1.4B model for 30 steps
# on whatever tokenized DCLM slice is available, to verify the multi-node
# coordinator handshake + NCCL all-reduce works end-to-end on this cluster.
#
# Single-node usage (no env vars set): just runs normally on one node's 8 GPUs.
# Multi-node usage: launched via multi_node_launch.sh.

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
DCLM_TRAIN = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_train-d321eb"

# Read multi-node env vars (set by multi_node_launch.sh). All absent => single-node.
def _distributed_from_env() -> DistributedConfig:
    num_proc = os.environ.get("JAX_DIST_NUM_PROCESSES")
    proc_id = os.environ.get("JAX_DIST_PROCESS_ID")
    coord = os.environ.get("JAX_DIST_COORDINATOR")
    if num_proc is None:
        return DistributedConfig()
    return DistributedConfig(
        num_processes=int(num_proc),
        process_id=int(proc_id),
        coordinator_address=coord,
        local_device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    )


data_config = LmDataConfig(
    components={"dclm_200m": DatasetComponent(cache_dir=DCLM_TRAIN)},
    train_weights={"dclm_200m": 1.0},
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
            tags=["smoke-test", "multinode-validation"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        # batch_size = num_nodes * 8 GPUs * 8 per-device = num_nodes * 64.
        # Single-node = 64. 2-node = 128. 4-node = 256.
        train_batch_size=int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1")) * 64,
        num_train_steps=30,  # smoke test: 30 steps is enough to verify init + a few iterations
        steps_per_eval=100,  # don't eval during smoke test
        per_device_parallelism=8,
        per_device_eval_parallelism=8,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/smoke_multinode/",
            save_interval=timedelta(minutes=999),
            keep=[],
        ),
        ray=RayConfig(auto_start_cluster=False),
        distributed=_distributed_from_env(),
    ),
    model=model_config,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=1e-3,
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
    from levanter.main import train_lm
    train_lm.main(train_config)
