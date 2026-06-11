"""Quick 50-step smoke to verify 4B at per_device=4 (batch=256) doesn't OOM.
Same config as run_4b_dclm_short.py but only 50 steps."""

import os
from pathlib import Path

_secrets = Path(__file__).resolve().parents[2] / ".secrets"
if _secrets.exists():
    for line in _secrets.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

# wandb async-write socket errors floodied the previous 8-node logs and may
# have triggered the JAX barrier timeout on this 8-node config. Force offline.
os.environ["WANDB_MODE"] = "offline"

from datetime import timedelta
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

from experiments.llama import llama_3_5b

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"
DCLM_SHARD = f"{BASE_TOKENIZED}/dclm_baseline-0206f1/train/part-00006"
DCLM_VAL = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_val-415aea"


def _distributed_from_env():
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
PER_DEVICE_PARALLELISM = 1  # batch=64 — exactly matches the 8B smoke that ran
                            # cleanly; higher values caused hangs/barrier timeouts
TRAIN_BATCH_SIZE = NUM_PROC * 8 * PER_DEVICE_PARALLELISM  # 128 on 8 nodes

data_config = LmDataConfig(
    components={
        "dclm_shard0": DatasetComponent(cache_dir=DCLM_SHARD),
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
    },
    train_weights={"dclm_shard0": 1.0, "dclm_200m_val": 0.0},
    num_validation_sequences={"dclm_shard0": 0},
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=False,
    shuffle_before_trainval_split=False,
    stop_strategy="restart",
    enforce_eos=True,
)

train_config = TrainLmConfig(
    data=data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency", entity="dongwei_jiang",
            tags=["3.5b", "smoke", "batch256", "8node-fsdp"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=50,
        steps_per_eval=10000,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/3_5b_smoke_b256/",
            save_interval=timedelta(hours=24),
            keep=[],
        ),
        ray=RayConfig(auto_start_cluster=False),
        distributed=_distributed_from_env(),
        mesh=MeshConfig(
            axes={"data": -1, "replica": 1, "model": 1},
            dcn_axes={"replica_dcn": -1},
        ),
        jax_compilation_cache_dir="/fsx/users/dongweij/marin/outputs/jax_compile_cache",
    ),
    model=llama_3_5b,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=3e-4, weight_decay=0.1, lr_schedule="constant",
        beta1=0.9, beta2=0.95, warmup=0.01, max_grad_norm=1.0,
    ),
    data_seed=0,
)

if __name__ == "__main__":
    print(f"=== 3.5B smoke per_device={PER_DEVICE_PARALLELISM} batch={TRAIN_BATCH_SIZE} ===")
    from levanter.main import train_lm
    train_lm.main(train_config)
