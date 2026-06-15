# 1.4B model, 1 epoch over the matched code-mix variant:
#   - 75% DCLM (23.08B tokens, 1 epoch from the 35.5B available)
#   - 25% code (1 epoch over each code source):
#       * aryabumi_code_synth_full   5.40B  → 70.2% of the 25% slice
#       * aryabumi_code_web          1.35B  → 17.6%
#       * opc_algorithmic            0.94B  → 12.2%
#       * total code                 7.69B  = 25% of 30.77B total trained
#
# Compares apples-to-apples against run_1_4b_1ep_dclm.py (same total trained
# tokens, same hyperparams, same n_steps). Hypothesis: under matched-compute
# and 1-epoch (no repetition confound), Aryabumi-style 25% code mix either
# helps (Paloma + Closed-book NL up) or doesn't (matches baseline = May 26
# code-mix "win" was just a unique-tokens confound from v1's larger
# code corpus).
#
# Reads JAX_DIST_* env vars for multi-node like the baseline.

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
# See run_1_4b_1ep_dclm.py for why we use per-shard subdirs.
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
ARYABUMI_SYNTH = f"{BASE_TOKENIZED}/aryabumi_code_synth_full-0678c3"
ARYABUMI_WEB = f"{BASE_TOKENIZED}/aryabumi_code_web-591a44"
OPC_ALG = f"{BASE_TOKENIZED}/opc_algorithmic-ffc825"

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
PER_DEVICE_PARALLELISM = 8
TRAIN_BATCH_SIZE = TOTAL_GPUS * PER_DEVICE_PARALLELISM
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096
TARGET_TOKENS = 30_770_000_000  # same as run_1_4b_1ep_dclm.py
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP


# Mixture weights designed so that each component is sampled exactly 1 epoch
# at 30.77B total trained tokens:
#   - dclm  : 0.75   * 30.77B = 23.08B (1 epoch of subset)
#   - synth : 0.175  * 30.77B =  5.38B (1 epoch over 5.40B)
#   - web   : 0.044  * 30.77B =  1.35B (1 epoch over 1.35B)
#   - opc   : 0.031  * 30.77B =  0.95B (1 epoch over 0.94B)
_dclm_components = {f"dclm_shard{i}": DatasetComponent(cache_dir=p) for i, p in enumerate(DCLM_SHARDS)}
_dclm_weights = {k: 0.75 / len(DCLM_SHARDS) for k in _dclm_components}
# Limit per-component validation. DCLM shards + aryabumi caches have
# train==validation (via symlinks); slice to 0 to disable their per-component
# eval. dclm_200m_val + paloma_* remain the actual held-out signals.
_shard_val_sizes = {
    **{k: 0 for k in _dclm_components},
    "aryabumi_synth": 0,
    "aryabumi_web": 0,
    "opc_algorithmic": 0,
}

data_config = LmDataConfig(
    components={
        **_dclm_components,
        "aryabumi_synth": DatasetComponent(cache_dir=ARYABUMI_SYNTH),
        "aryabumi_web":   DatasetComponent(cache_dir=ARYABUMI_WEB),
        "opc_algorithmic": DatasetComponent(cache_dir=OPC_ALG),
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights={
        **_dclm_weights,
        "aryabumi_synth": 0.175,
        "aryabumi_web":   0.044,
        "opc_algorithmic": 0.031,
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
            tags=["1.4b", "1ep", "code-mix-25", "aryabumi-style", "wd-0.1", f"nodes-{NUM_PROC}"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_1ep_code25/",
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
    print(f"=== 1.4B 1-epoch CODE-MIX 25% (aryabumi-style) ===")
    print(f"  num_processes (nodes): {NUM_PROC}")
    print(f"  total GPUs: {TOTAL_GPUS}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}")
    print(f"  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  mix: dclm 75% / synth 17.5% / web 4.4% / opc 3.1% (each ~1 epoch)")
    print(f"  LR={3e-4}, WD=0.1, cosine to 0")
    from levanter.main import train_lm
    train_lm.main(train_config)
