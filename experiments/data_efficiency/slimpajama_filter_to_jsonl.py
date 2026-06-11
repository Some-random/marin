#!/usr/bin/env python3
"""Filter SlimPajama parquets → jsonl.gz with NL sources only.

For each parquet in data/slimpajama_raw_chunk1/data/, keep rows where
meta.redpajama_set_name ∈ NL_SOURCES (drop GitHub + StackExchange) and write
to outputs/raw/slimpajama_nl/part-XXXXX.jsonl.gz.

Use as input to marin's default_tokenize() → Levanter cache.

Also writes outputs/raw/slimpajama_nl/manifest.json with per-source row/byte counts
so we can verify the cache matches Aryabumi's per-source proportions.
"""

import ast
import gzip
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

NL_SOURCES = {
    "RedPajamaCommonCrawl", "RedPajamaC4",
    "RedPajamaBook", "RedPajamaArXiv", "RedPajamaWikipedia",
}
DROP_SOURCES = {"RedPajamaGithub", "RedPajamaStackExchange"}

REPO = Path("/fsx/users/dongweij/marin")
SRC = REPO / "data" / "slimpajama_raw_chunk1" / "data"
DST = REPO / "outputs" / "raw" / "slimpajama_nl"
DST.mkdir(parents=True, exist_ok=True)


def process_one(parquet_path: Path) -> dict:
    """Read parquet, filter, write a single jsonl.gz, return per-source counts."""
    out_path = DST / parquet_path.name.replace(".parquet", ".jsonl.gz")
    stats = {"src_rows": 0, "kept_rows": 0, "by_source": Counter(), "by_source_bytes": Counter()}
    table = pq.read_table(parquet_path)
    texts = table.column("text").to_pylist()
    metas = table.column("meta").to_pylist()
    stats["src_rows"] = len(texts)

    # rokset3/slim_pajama_chunk1 stores meta as a Python-repr STRING like
    #   "{'redpajama_set_name': 'RedPajamaC4'}"
    # DKYoon stores it as a struct (already a dict). Handle both.
    def _src_of(m):
        if isinstance(m, dict):
            return m["redpajama_set_name"]
        return ast.literal_eval(m)["redpajama_set_name"]

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for txt, m in zip(texts, metas):
            src = _src_of(m)
            if src not in NL_SOURCES:
                continue
            stats["kept_rows"] += 1
            stats["by_source"][src] += 1
            stats["by_source_bytes"][src] += len(txt.encode("utf-8"))
            f.write(json.dumps({"text": txt, "source": src}) + "\n")
    return stats


def _process_one_or_skip(parquet_path: Path) -> dict:
    """Skip if the corresponding jsonl.gz already exists. Else run process_one."""
    out_path = DST / parquet_path.name.replace(".parquet", ".jsonl.gz")
    if out_path.exists() and out_path.stat().st_size > 0:
        # already done — but we still need to recount for the manifest.
        # Read it back to recover stats.
        stats = {"src_rows": 0, "kept_rows": 0, "by_source": Counter(), "by_source_bytes": Counter()}
        with gzip.open(out_path, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                stats["kept_rows"] += 1
                stats["by_source"][d["source"]] += 1
                stats["by_source_bytes"][d["source"]] += len(d["text"].encode("utf-8"))
        stats["src_rows"] = stats["kept_rows"]  # under-count; OK for monitoring purposes
        return stats
    return process_one(parquet_path)


def main():
    import multiprocessing as mp
    parquets = sorted(SRC.glob("*.parquet"))
    if not parquets:
        print(f"ERROR: no parquets under {SRC}")
        return

    n_workers = min(16, mp.cpu_count())
    print(f"processing {len(parquets)} parquets → {DST} ({n_workers} workers)", flush=True)
    agg = {"by_source": Counter(), "by_source_bytes": Counter(), "src_rows": 0, "kept_rows": 0}
    with mp.Pool(n_workers) as pool:
        for i, s in enumerate(pool.imap_unordered(_process_one_or_skip, parquets), 1):
            agg["src_rows"] += s["src_rows"]
            agg["kept_rows"] += s["kept_rows"]
            agg["by_source"] += s["by_source"]
            agg["by_source_bytes"] += s["by_source_bytes"]
            if i % 5 == 0 or i == len(parquets):
                print(f"  [{i}/{len(parquets)}] cumulative: kept={agg['kept_rows']:,} ", flush=True)

    total_kept = agg["kept_rows"]
    print(f"\nfinal: {agg['src_rows']:,} src rows → {total_kept:,} kept ({100*total_kept/agg['src_rows']:.1f}%)")
    print("per-source kept rows / bytes:")
    for src in sorted(agg["by_source"].keys()):
        rows = agg["by_source"][src]
        bytes_ = agg["by_source_bytes"][src]
        print(f"  {src:25} {rows:>12,} rows {bytes_/1e9:>8.2f} GB ({100*bytes_/sum(agg['by_source_bytes'].values()):.1f}% of NL bytes)")

    manifest_path = DST / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "kept_rows": total_kept,
            "src_rows": agg["src_rows"],
            "by_source_rows": dict(agg["by_source"]),
            "by_source_bytes": dict(agg["by_source_bytes"]),
            "nl_sources": list(NL_SOURCES),
            "drop_sources": list(DROP_SOURCES),
        }, f, indent=2)
    print(f"\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
