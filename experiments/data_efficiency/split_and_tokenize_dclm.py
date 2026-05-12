"""
Split the downloaded 1.25M-doc DCLM JSONL into disjoint subsets for H1 experiment,
then tokenize each subset into its own Levanter cache.

Splits:
  - phase1: first ~325k docs (~400M tokens) — used by control phase 1
  - phase2: next ~650k docs (~800M tokens) — used by both arms in phase 2
  - remaining: spare

Usage:
  .venv/bin/python experiments/data_efficiency/split_and_tokenize_dclm.py
"""

import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RAW_JSONL = "/fsx/users/dongweij/marin/outputs/raw/dclm_1500m.jsonl"
OUTPUT_BASE = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency"
TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

SPLITS = {
    "dclm_h1_phase1": 360_000,   # ~440M tokens for control phase 1 (matching OWM+Code budget)
    "dclm_h1_phase2": 660_000,   # ~810M tokens for phase 2
}


def split_jsonl():
    split_files = {}
    for name in SPLITS:
        path = os.path.join(OUTPUT_BASE, f"{name}.jsonl")
        if os.path.exists(path):
            n = sum(1 for _ in open(path))
            logger.info(f"{name}.jsonl already exists ({n:,} docs), skipping split.")
            split_files[name] = path
            continue
        split_files[name] = path

    if all(os.path.exists(p) for p in split_files.values()):
        return split_files

    logger.info(f"Splitting {RAW_JSONL} into {len(SPLITS)} subsets...")
    handles = {}
    for name in SPLITS:
        handles[name] = open(split_files[name], "w")

    offset = 0
    split_order = list(SPLITS.keys())
    current_split_idx = 0
    current_count = 0
    target = SPLITS[split_order[0]]

    with open(RAW_JSONL) as f:
        for line in f:
            if current_split_idx >= len(split_order):
                break
            name = split_order[current_split_idx]
            handles[name].write(line)
            current_count += 1
            if current_count >= target:
                logger.info(f"  {name}: {current_count:,} docs")
                current_split_idx += 1
                current_count = 0
                if current_split_idx < len(split_order):
                    target = SPLITS[split_order[current_split_idx]]

    for h in handles.values():
        h.close()

    return split_files


def tokenize_split(name, jsonl_path):
    cache_dir = os.path.join(OUTPUT_BASE, name, "train")
    ledger_path = os.path.join(cache_dir, "shard_ledger.json")

    if os.path.exists(ledger_path):
        meta = json.load(open(ledger_path))
        if meta.get("is_finished"):
            logger.info(f"{name} cache already built ({meta['total_num_rows']:,} docs)")
            return

    from levanter.data.sharded_datasource import datasource_from_jsonl
    from levanter.data.text.cache import build_lm_dataset_cache
    from levanter.data.text.formats import TextLmDatasetFormat
    from levanter.store.cache import CacheOptions
    from transformers import AutoTokenizer

    logger.info(f"Tokenizing {name} from {jsonl_path} -> {cache_dir} ...")
    source = datasource_from_jsonl([jsonl_path])
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    build_lm_dataset_cache(
        cache_dir=cache_dir,
        source=source,
        format=TextLmDatasetFormat(text_key="text"),
        tokenizer=tokenizer,
        options=CacheOptions(batch_size=256),
        enforce_eos=True,
    )
    logger.info(f"{name} tokenization complete!")


def count_tokens(name):
    import tensorstore as ts
    import asyncio

    cache_dir = os.path.join(OUTPUT_BASE, name, "train")
    ledger = json.load(open(os.path.join(cache_dir, "shard_ledger.json")))
    total_docs = ledger['total_num_rows']

    async def _count():
        store = await ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': os.path.join(cache_dir, "input_ids/offsets")},
        })
        total_tokens = int(await store[total_docs].read())
        logger.info(f"{name}: {total_docs:,} docs, {total_tokens:,} tokens ({total_tokens/1e6:.0f}M)")

    asyncio.run(_count())


if __name__ == "__main__":
    split_files = split_jsonl()
    for name, path in split_files.items():
        tokenize_split(name, path)
        count_tokens(name)
