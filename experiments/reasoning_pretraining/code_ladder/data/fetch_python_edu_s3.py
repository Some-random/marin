# Fetch python-edu file contents from Software Heritage S3 (public bucket).
# The smollm-corpus/python-edu config provides only blob_id metadata; the
# actual gzipped source code lives at s3://softwareheritage/content/{blob_id}.
#
# Writes JSONL.gz shards of ~100k records each with fields:
#   {blob_id, repo_name, path, length_bytes, score, int_score, text}
#
# Usage:
#   .venv/bin/python -m experiments.reasoning_pretraining.code_ladder.data.fetch_python_edu_s3 \
#     --metadata-dir /fsx/users/dongweij/marin/outputs/raw/phi_1_5_mix/python-edu \
#     --out-dir /fsx/users/dongweij/marin/outputs/raw/phi_1_5_mix/python_edu_content \
#     --threads 128

import argparse
import gzip
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BUCKET = "softwareheritage"
KEY_PREFIX = "content/"
RECORDS_PER_SHARD = 100_000


def make_s3_client():
    # Public bucket, no creds needed; UNSIGNED disables sigv4 signing.
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=2048,
        ),
    )


# Thread-local S3 client so each worker thread reuses one connection pool.
_thread_local = __import__("threading").local()


def fetch_one(meta_row):
    """Fetch one blob; return dict with text or None on miss/failure."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = make_s3_client()
    client = _thread_local.client
    blob_id = meta_row["blob_id"]
    try:
        obj = client.get_object(Bucket=BUCKET, Key=f"{KEY_PREFIX}{blob_id}")
        raw = obj["Body"].read()
        text = gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("utf-8", errors="replace")
        return {**meta_row, "text": text}
    except Exception as e:  # noqa: BLE001 — log and continue on missing/failed blobs
        return {**meta_row, "text": None, "_error": f"{type(e).__name__}: {e}"}


def iter_metadata(metadata_dir: Path, rank: int = 0, world_size: int = 1):
    """Yield dicts with python-edu metadata fields, across all parquet shards.

    When world_size > 1, only yields rows where global_index % world_size == rank.
    """
    paths = sorted(metadata_dir.glob("train-*.parquet"))
    global_idx = 0
    for p in paths:
        t = pq.read_table(p)
        cols = t.column_names
        for batch in t.to_batches(max_chunksize=10_000):
            d = batch.to_pydict()
            n = len(d[cols[0]])
            for i in range(n):
                if global_idx % world_size == rank:
                    yield {c: d[c][i] for c in cols}
                global_idx += 1


def write_shard(out_dir: Path, shard_idx: int, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"python_edu_{shard_idx:05d}.jsonl.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log.info(f"wrote {out_path.name} ({len(records):,} records)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="cap total records (for testing)")
    ap.add_argument("--rank", type=int, default=0, help="this worker's rank (0..world_size-1)")
    ap.add_argument("--world-size", type=int, default=1, help="total number of workers")
    args = ap.parse_args()
    assert 0 <= args.rank < args.world_size, "rank must be < world_size"

    # Per-rank output subdir, so multiple workers don't race on shard file names.
    rank_out_dir = args.out_dir / f"rank_{args.rank:02d}" if args.world_size > 1 else args.out_dir
    rank_out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip records already written. We count completed shards (each shard
    # is the next RECORDS_PER_SHARD slice of THIS rank's metadata slice).
    existing = sorted(rank_out_dir.glob("python_edu_*.jsonl.gz"))
    next_shard_idx = len(existing)
    skip_records = next_shard_idx * RECORDS_PER_SHARD
    if skip_records:
        log.info(f"[rank {args.rank}] resuming: skipping {skip_records:,} records ({next_shard_idx} shards already present)")

    total = 0
    successes = 0
    failures = 0
    t0 = time.time()
    buffer: list[dict] = []
    shard_idx = next_shard_idx

    metadata_iter = iter_metadata(args.metadata_dir, args.rank, args.world_size)
    for _ in range(skip_records):
        next(metadata_iter, None)

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        # Stream metadata into the pool in batches; collect results in arrival order
        BATCH = RECORDS_PER_SHARD
        batch_meta = []
        for row in metadata_iter:
            batch_meta.append(row)
            if args.limit is not None and total + len(batch_meta) > args.limit:
                batch_meta = batch_meta[: args.limit - total]

            if len(batch_meta) >= BATCH or (args.limit is not None and total + len(batch_meta) >= args.limit):
                # Fan out this batch
                futures = [pool.submit(fetch_one, r) for r in batch_meta]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res.get("text") is not None:
                        successes += 1
                    else:
                        failures += 1
                    buffer.append(res)
                total += len(batch_meta)
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed else 0.0
                log.info(
                    f"fetched {total:,} (succ={successes:,} fail={failures:,}) "
                    f"in {elapsed:.0f}s — {rate:.0f}/s — buffer={len(buffer):,}"
                )
                # Flush
                while len(buffer) >= RECORDS_PER_SHARD:
                    write_shard(rank_out_dir, shard_idx, buffer[:RECORDS_PER_SHARD])
                    buffer = buffer[RECORDS_PER_SHARD:]
                    shard_idx += 1
                batch_meta = []
                if args.limit is not None and total >= args.limit:
                    break

        # Trailing partial batch
        if batch_meta:
            futures = [pool.submit(fetch_one, r) for r in batch_meta]
            for fut in as_completed(futures):
                res = fut.result()
                if res.get("text") is not None:
                    successes += 1
                else:
                    failures += 1
                buffer.append(res)
            total += len(batch_meta)

    if buffer:
        write_shard(rank_out_dir, shard_idx, buffer)
    elapsed = time.time() - t0
    log.info(
        f"[rank {args.rank}] DONE — total fetched={total:,}, successes={successes:,}, failures={failures:,}, "
        f"elapsed={elapsed:.0f}s, avg rate={total / elapsed:.0f}/s"
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
