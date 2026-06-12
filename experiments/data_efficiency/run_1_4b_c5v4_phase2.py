# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B C5-v4 PHASE 2: continued text pretraining on SlimPajama-NL.
#
# Hypothesis: the NL deficit we see in C5/C5-v2/C5-v3 (Open-book and
# Closed-book NL trailing A5 1ep DCLM-only by 5–11 pp) is partly a *data*
# axis effect, not just the LR-schedule recipe. Aryabumi et al §2.1
# trained on SlimPajama-with-GitHub-and-StackExchange-removed (CommonCrawl
# + C4 + Books + ArXiv + Wikipedia, 503B tokens); we have only ever used
# DCLM-baseline (CommonCrawl-only). C5-v4 swaps the phase-2 text source
# from DCLM → SlimPajama-NL, holding everything else from C5-v3 phase 2
# constant.
#
# Initialization: SAME as C5-v3 — fresh cosine 3e-4 → 0 from the C5-v3
# phase-1 step-14671 code-only checkpoint, fresh optimizer state.
# Code mix proportions inside the 10% code+markup slot: SAME as C5-v3.
# Only the 90% TEXT source changes.

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


# === Phase 1 final checkpoint to initialize from (UNCHANGED from C5-v3) ===
PHASE1_INIT_FROM = "checkpoints/1_4b_c5v3_phase1/8dtdcear/step-14671"


def _resolve_cache(prefix: str) -> str:
    matches = sorted(_TOKENIZED_BASE.glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(
            f"No tokenized cache for prefix '{prefix}'. "
            f"Run `MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python "
            f"-m experiments.data_efficiency.code_data_c5v2` first."
        )
    return str(matches[-1])


try:
    SE_PYTHON_CACHE = _resolve_cache("c5v2_stack_edu_python_clean")
    SE_MARKDOWN_CACHE = _resolve_cache("c5v2_stack_edu_markdown_clean")
    NEMOTRON_CC_CACHE = _resolve_cache("c5v2_nemotron_code_concepts")
    NEMOTRON_UA_CACHE = _resolve_cache("c5v2_nemotron_unconditional_algorithmic")
except FileNotFoundError as _e:
    SE_PYTHON_CACHE = ""
    SE_MARKDOWN_CACHE = ""
    NEMOTRON_CC_CACHE = ""
    NEMOTRON_UA_CACHE = ""
    print(f"[c5v4-p2] WARN: {_e}")


# === TEXT: SlimPajama-NL (replaces DCLM) ===
# Tokenized cache at outputs/tokenized/slimpajama_nl-<hash>/train/part-<N>-of-128/
# After filtering Github + StackExchange: 13.23 B Llama-3.1 tokens.
# Per-source share (vs Aryabumi target): CC 58.0% (57.0%), C4 30.7% (29.2%),
# ArXiv 3.7% (5.0%), Books 4.3% (4.6%), Wikipedia 3.3% (4.2%).
# (Add chunk_2 shards here after the second tokenize finishes if you want
# >13.23 B unique-text budget — current setting epochs the cache ~1.05× to
# hit phase 2's 13.85 B text target.)
def _collect_sp_nl_shards() -> list[str]:
    """Collect SlimPajama-NL English-only shards across all available caches."""
    shards: list[str] = []
    for prefix in ("slimpajama_nl_en", "slimpajama_nl_chunk2_en"):
        try:
            root = Path(_resolve_cache(prefix))
        except FileNotFoundError:
            continue
        shards.extend(sorted(str(p) for p in (root / "train").glob("part-*-of-*")))
    return shards


SP_NL_SHARDS = _collect_sp_nl_shards()


# === DCLM_200M_VAL — kept for in-domain held-out tracking ===
DCLM_VAL = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_val-415aea"


# === Code mix proportions (UNCHANGED from C5-v3 phase 2) ===
CODE_RATIOS = {
    "se_python": 8.8 / 16.3,
    "nemotron_cc": 7.3 / 16.3,
    "nemotron_ua": 0.2 / 16.3,
}
MARKUP_RATIOS = {"se_markdown": 1.0}


# === Paloma eval components (UNCHANGED) ===
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
        found[f"paloma_{sub}"] = DatasetComponent(cache_dir=str(matches[-1]))
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
NUM_TRAIN_STEPS = 14_672  # UNCHANGED: same phase-2 budget as C5-v3 = 15.39 B tokens.


_code_key_to_cache = {
    "se_python": SE_PYTHON_CACHE,
    "nemotron_cc": NEMOTRON_CC_CACHE,
    "nemotron_ua": NEMOTRON_UA_CACHE,
}
_markup_key_to_cache = {
    "se_markdown": SE_MARKDOWN_CACHE,
}


def _phase2_weights() -> dict[str, float]:
    """90% SlimPajama-NL (equal across shards) + 10% (80% code + 20% markup) — matches Aryabumi §3.1 footnote 5."""
    w: dict[str, float] = {}
    if SP_NL_SHARDS:
        sp_share = 0.90 / len(SP_NL_SHARDS)
        for i, _ in enumerate(SP_NL_SHARDS):
            w[f"sp_nl_shard{i:03d}"] = sp_share
    for k, r in CODE_RATIOS.items():
        w[f"code_{k}"] = 0.10 * 0.80 * r
    for k, r in MARKUP_RATIOS.items():
        w[f"markup_{k}"] = 0.10 * 0.20 * r
    for k in ["dclm_200m_val", *[f"paloma_{s}" for s in PALOMA_SUBSETS]]:
        w[k] = 0.0
    return w


_sp_nl_components = {
    f"sp_nl_shard{i:03d}": DatasetComponent(cache_dir=p)
    for i, p in enumerate(SP_NL_SHARDS)
}
_code_components = {f"code_{k}": DatasetComponent(cache_dir=v) for k, v in _code_key_to_cache.items() if v}
_markup_components = {f"markup_{k}": DatasetComponent(cache_dir=v) for k, v in _markup_key_to_cache.items() if v}

_shard_val_sizes = {k: 0 for k in {**_sp_nl_components, **_code_components, **_markup_components}}

data_config = LmDataConfig(
    components={
        **_sp_nl_components,
        **_code_components,
        **_markup_components,
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
    initialize_from_checkpoint_path=os.environ.get("C5V3_PHASE1_CKPT", PHASE1_INIT_FROM),
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["1.4b", "c5v4", "phase2-text-continued", "slimpajama-nl", "wd-0.1", f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_c5v4_phase2/",
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
    print("=== 1.4B C5-v4 PHASE 2 (continued text from code-LM, SlimPajama-NL) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  initialize_from_checkpoint_path: {train_config.initialize_from_checkpoint_path}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  LR=3e-4 (FRESH), WD=0.1, cosine to 0 (warmup 1% ≈ 147 steps)")
    print(f"  data: 90% SlimPajama-NL ({len(SP_NL_SHARDS)} shards) + 10% (80% code + 20% markup)")
    if not SE_PYTHON_CACHE or not SE_MARKDOWN_CACHE or not NEMOTRON_CC_CACHE or not NEMOTRON_UA_CACHE:
        print("\nERROR: code caches missing.")
        raise SystemExit(1)
    if not SP_NL_SHARDS:
        print("\nERROR: SlimPajama-NL shards missing.")
        raise SystemExit(1)
    from levanter.main import train_lm
    train_lm.main(train_config)
