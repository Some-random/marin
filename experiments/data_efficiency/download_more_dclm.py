"""
Download and tokenize ~1.5B tokens from DCLM baseline for H1 experiment.

Step 1: Stream ~500k docs from mlfoundations/dclm-baseline-1.0, save as JSONL
Step 2: Tokenize JSONL into Levanter cache format

Usage:
  .venv/bin/python experiments/data_efficiency/download_more_dclm.py
"""

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RAW_JSONL = "/fsx/users/dongweij/marin/outputs/raw/dclm_1500m.jsonl"
CACHE_DIR = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_1500m_train"
TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
TARGET_DOCS = 1_250_000


def download():
    if os.path.exists(RAW_JSONL):
        n = sum(1 for _ in open(RAW_JSONL))
        logger.info(f"JSONL already exists with {n:,} docs, skipping download.")
        return

    from datasets import load_dataset

    logger.info(f"Streaming {TARGET_DOCS:,} docs from DCLM baseline...")
    ds = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True)

    os.makedirs(os.path.dirname(RAW_JSONL), exist_ok=True)
    count = 0
    with open(RAW_JSONL, "w") as f:
        for doc in ds:
            f.write(json.dumps({"text": doc["text"]}) + "\n")
            count += 1
            if count % 10_000 == 0:
                logger.info(f"  Downloaded {count:,} / {TARGET_DOCS:,} docs")
            if count >= TARGET_DOCS:
                break

    logger.info(f"Downloaded {count:,} docs to {RAW_JSONL}")


def tokenize():
    ledger_path = os.path.join(CACHE_DIR, "train", "shard_ledger.json")
    if os.path.exists(ledger_path):
        meta = json.load(open(ledger_path))
        if meta.get("is_finished"):
            logger.info(f"Cache already built at {CACHE_DIR}/train ({meta['total_num_rows']:,} docs)")
            return

    from levanter.data.sharded_datasource import datasource_from_jsonl
    from levanter.data.text.cache import build_lm_dataset_cache
    from levanter.data.text.formats import TextLmDatasetFormat
    from levanter.store.cache import CacheOptions
    from transformers import AutoTokenizer

    logger.info(f"Tokenizing {RAW_JSONL} -> {CACHE_DIR}/train ...")
    source = datasource_from_jsonl([RAW_JSONL])
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    cache = build_lm_dataset_cache(
        cache_dir=os.path.join(CACHE_DIR, "train"),
        source=source,
        format=TextLmDatasetFormat(text_key="text"),
        tokenizer=tokenizer,
        options=CacheOptions(batch_size=256),
        enforce_eos=True,
    )
    logger.info("Tokenization complete!")


def count_tokens():
    import tensorstore as ts
    import asyncio

    ledger = json.load(open(os.path.join(CACHE_DIR, "train", "shard_ledger.json")))
    total_docs = ledger['total_num_rows']

    async def _count():
        store = await ts.open({
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': os.path.join(CACHE_DIR, "train/input_ids/offsets")},
        })
        total_tokens = int(await store[total_docs].read())
        logger.info(f"Total docs: {total_docs:,}")
        logger.info(f"Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")

    asyncio.run(_count())


if __name__ == "__main__":
    download()
    tokenize()
    count_tokens()
