# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Small-scale continued pretraining phase 2 (C5-v3 / C5-v4 / C5-v6 mirror) at Chinchilla ½ budget.
#
# Env vars:
#   MODEL_KEY       — "300m4k" / "600m4k"
#   TARGET_TOKENS   — phase 2 tokens (e.g. 3e9 = ½ of 300M Chinchilla budget)
#   TEXT_SOURCE     — "dclm" or "sp_nl"
#   REPLAY_FRAC     — code+markup fraction in phase 2 (e.g. 0.10 → C5-v3/v4 style, 0.30 → C5-v6 style)
#   PHASE1_CKPT     — checkpoint to init from (model weights only, fresh optimizer)
#   RUN_TAG         — wandb tag + checkpoint dir

import os
from datetime import timedelta
from pathlib import Path

_secrets = Path(__file__).resolve().parents[2] / ".secrets"
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

from experiments.data_efficiency.models import model_dict

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"
_TOKENIZED_BASE = Path(BASE_TOKENIZED)


def _resolve_cache(prefix: str) -> str:
    matches = sorted(_TOKENIZED_BASE.glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(f"No tokenized cache for prefix '{prefix}'.")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple matches for '{prefix}': {[m.name for m in matches]}")
    return str(matches[0])


MODEL_KEY     = os.environ["MODEL_KEY"]
TARGET_TOKENS = int(float(os.environ["TARGET_TOKENS"]))
TEXT_SOURCE   = os.environ["TEXT_SOURCE"]
REPLAY_FRAC   = float(os.environ["REPLAY_FRAC"])
PHASE1_CKPT   = os.environ["PHASE1_CKPT"]
RUN_TAG       = os.environ["RUN_TAG"]


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


CODE_RATIOS = {
    "se_python":   8.8 / 16.3,
    "nemotron_cc": 7.3 / 16.3,
    "nemotron_ua": 0.2 / 16.3,
}
MARKUP_RATIOS = {"se_markdown": 1.0}


NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
PER_DEVICE_PARALLELISM = 8
TRAIN_BATCH_SIZE = TOTAL_GPUS * PER_DEVICE_PARALLELISM
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP


# Build components
_code_components = {
    "code_se_python":   DatasetComponent(cache_dir=SE_PYTHON_CACHE),
    "code_nemotron_cc": DatasetComponent(cache_dir=NEMOTRON_CC_CACHE),
    "code_nemotron_ua": DatasetComponent(cache_dir=NEMOTRON_UA_CACHE),
}
_markup_components = {"markup_se_markdown": DatasetComponent(cache_dir=SE_MARKDOWN_CACHE)}

if TEXT_SOURCE == "dclm":
    _text_components = {f"dclm_shard{i}": DatasetComponent(cache_dir=p) for i, p in enumerate(DCLM_SHARDS)}
    _text_weights_pure = {f"dclm_shard{i}": 1.0 / len(DCLM_SHARDS) for i in range(len(DCLM_SHARDS))}
elif TEXT_SOURCE == "sp_nl":
    sp_shards = _collect_sp_nl_shards_with_rows()
    sp_total = sum(r for _, r in sp_shards) or 1
    _text_components = {f"sp_nl_shard{i:03d}": DatasetComponent(cache_dir=p) for i, (p, _) in enumerate(sp_shards)}
    _text_weights_pure = {f"sp_nl_shard{i:03d}": r / sp_total for i, (_, r) in enumerate(sp_shards)}
else:
    raise ValueError(f"unknown TEXT_SOURCE: {TEXT_SOURCE!r}")


_train_weights: dict[str, float] = {}
text_share = 1.0 - REPLAY_FRAC
for name, w in _text_weights_pure.items():
    _train_weights[name] = text_share * w
for k, r in CODE_RATIOS.items():
    _train_weights[f"code_{k}"] = REPLAY_FRAC * 0.80 * r
for k, r in MARKUP_RATIOS.items():
    _train_weights[f"markup_{k}"] = REPLAY_FRAC * 0.20 * r
_train_weights["dclm_200m_val"] = 0.0
for s in PALOMA_SUBSETS:
    _train_weights[f"paloma_{s}"] = 0.0

data_config = LmDataConfig(
    components={
        # STRICT PREFIX REPLAY VARIANT (300M version of 1.4B c5v6_strict_step14671):
        # code + markup FIRST so the per-component Feistel shuffle keys match phase 1's
        # order (code-p1's components dict also has code+markup first). Phase 2's reads of
        # the same caches = strict prefix of phase 1's shuffled stream. The 1.4B result
        # showed strict LOSES to same-cache-different-shuffle (Mean Aggregate -0.009);
        # this script tests whether the same holds at 300M.
        **_code_components,
        **_markup_components,
        **_text_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights=_train_weights,
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
    initialize_from_checkpoint_path=PHASE1_CKPT,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency-smallscale",
            entity="dongwei_jiang",
            tags=["smallscale", MODEL_KEY, "phase2", TEXT_SOURCE, f"replay{int(REPLAY_FRAC*100)}", RUN_TAG, f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path=f"checkpoints/smallscale_{MODEL_KEY}_p2_strict_{TEXT_SOURCE}_r{int(REPLAY_FRAC*100)}/",
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
    print(f"=== smallscale phase 2: {MODEL_KEY} on {TEXT_SOURCE} @ replay={REPLAY_FRAC:.0%} ({TARGET_TOKENS/1e9:.2f} B tokens) ===")
    print(f"  init_from: {PHASE1_CKPT}")
    print(f"  nodes={NUM_PROC} batch={TRAIN_BATCH_SIZE} steps={NUM_TRAIN_STEPS:,}")
    from levanter.main import train_lm
    train_lm.main(train_config)
