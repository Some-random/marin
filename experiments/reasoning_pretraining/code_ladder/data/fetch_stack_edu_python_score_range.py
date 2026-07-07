# Fetch Stack-Edu Python source code from Software Heritage S3 for a SCORE RANGE.
#
# Used to extend the existing stack-edu-python-content cache (score >= 3.0) with
# DISJOINT lower-score blobs in [score_min, score_max). For C5-v6-NEW we use
# [2.7, 3.0) to get fresh Python docs of similar (but slightly lower) edu quality.
#
# Writes JSONL.gz shards of ~100k records each with fields:
#   {blob_id, repo_name, path, score, length_bytes, text}
#
# Source parquets: /fsx/users/dongweij/marin/outputs/raw/stack-edu/Python/train-*.parquet
# SWH bucket: s3://softwareheritage/content/{blob_id} (unsigned, public)
#
# Usage (8 parallel shards, ~26 min wall for ~6.9M docs in [2.7, 3.0)):
#   for r in 0 1 2 3 4 5 6 7; do
#     nohup .venv/bin/python -m experiments.reasoning_pretraining.code_ladder.data.fetch_stack_edu_python_score_range \
#       --metadata-dir /fsx/users/dongweij/marin/outputs/raw/stack-edu/Python \
#       --out-dir /fsx/users/dongweij/marin/outputs/raw/stack-edu-python-content-low \
#       --score-min 2.7 --score-max 3.0 \
#       --threads 512 --rank $r --world-size 8 \
#       > /fsx/users/dongweij/marin/logs/se_python_fetch_low_rank${r}.log 2>&1 &
#   done

import argparse
import gzip
import io
import json
import logging
import sys
import threading
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
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=2048,
        ),
    )


_thread_local = threading.local()


def fetch_one(meta_row):
    """Fetch one blob from SWH; return meta_row with `text` populated (or None on miss)."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = make_s3_client()
    client = _thread_local.client
    blob_id = meta_row["blob_id"]
    try:
        obj = client.get_object(Bucket=BUCKET, Key=f"{KEY_PREFIX}{blob_id}")
        raw = obj["Body"].read()
        text = gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("utf-8", errors="replace")
        return {**meta_row, "text": text}
    except Exception as e:  # noqa: BLE001
        return {**meta_row, "text": None, "_error": f"{type(e).__name__}: {e}"}


def iter_metadata(
    metadata_dir: Path,
    *,
    score_min: float,
    score_max: float,
    rank: int = 0,
    world_size: int = 1,
):
    """Yield Stack-Edu rows in [score_min, score_max), partitioned by rank.

    Keeps the projection minimal so we don't haul large `text`-less columns
    through memory. Score range is half-open: [score_min, score_max).
    """
    paths = sorted(metadata_dir.glob("train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet shards under {metadata_dir}")
    keep_cols = ["blob_id", "repo_name", "path", "score", "length_bytes"]
    global_idx = 0
    for p in paths:
        t = pq.read_table(p, columns=keep_cols)
        for batch in t.to_batches(max_chunksize=10_000):
            d = batch.to_pydict()
            n = len(d["blob_id"])
            for i in range(n):
                s = float(d["score"][i])
                if not (score_min <= s < score_max):
                    continue
                if global_idx % world_size == rank:
                    yield {c: d[c][i] for c in keep_cols}
                global_idx += 1


def write_shard(out_dir: Path, shard_idx: int, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"se_python_low_{shard_idx:05d}.jsonl.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log.info(f"wrote {out_path.name} ({len(records):,} records)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--score-min", type=float, required=True)
    ap.add_argument("--score-max", type=float, required=True)
    ap.add_argument("--threads", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="cap total records (for testing)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    args = ap.parse_args()
    assert 0 <= args.rank < args.world_size
    assert args.score_min < args.score_max

    rank_out_dir = args.out_dir / f"rank_{args.rank:02d}" if args.world_size > 1 else args.out_dir
    rank_out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip records whose shards already exist
    existing = sorted(rank_out_dir.glob("se_python_low_*.jsonl.gz"))
    next_shard_idx = len(existing)
    skip_records = next_shard_idx * RECORDS_PER_SHARD
    if skip_records:
        log.info(f"[rank {args.rank}] resuming: skipping {skip_records:,} records ({next_shard_idx} shards present)")

    total = 0
    successes = 0
    failures = 0
    t0 = time.time()
    buffer: list[dict] = []
    shard_idx = next_shard_idx

    metadata_iter = iter_metadata(
        args.metadata_dir,
        score_min=args.score_min,
        score_max=args.score_max,
        rank=args.rank,
        world_size=args.world_size,
    )
    for _ in range(skip_records):
        next(metadata_iter, None)

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        BATCH = RECORDS_PER_SHARD
        batch_meta = []
        for row in metadata_iter:
            batch_meta.append(row)
            if args.limit is not None and total + len(batch_meta) > args.limit:
                batch_meta = batch_meta[: args.limit - total]

            if len(batch_meta) >= BATCH or (args.limit is not None and total + len(batch_meta) >= args.limit):
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
                    f"[rank {args.rank}] fetched {total:,} (succ={successes:,} fail={failures:,}) "
                    f"in {elapsed:.0f}s — {rate:.0f}/s — buffer={len(buffer):,}"
                )
                while len(buffer) >= RECORDS_PER_SHARD:
                    write_shard(rank_out_dir, shard_idx, buffer[:RECORDS_PER_SHARD])
                    buffer = buffer[RECORDS_PER_SHARD:]
                    shard_idx += 1
                batch_meta = []
                if args.limit is not None and total >= args.limit:
                    break

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
        f"[rank {args.rank}] DONE — total={total:,}, succ={successes:,}, fail={failures:,}, "
        f"elapsed={elapsed:.0f}s, avg rate={total / elapsed:.0f}/s"
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
