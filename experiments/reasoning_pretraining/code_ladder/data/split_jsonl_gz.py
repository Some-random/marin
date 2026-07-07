#!/usr/bin/env python3
"""Split each .jsonl.gz under SRC into N_SPLITS sub-files under DST.

Lines are distributed round-robin across the N_SPLITS output files so each
sub-file has roughly equal row count and the source distribution stays
balanced.

Usage:
  SRC=<dir with foo.jsonl.gz, bar.jsonl.gz> \\
  DST=<output dir> \\
  N_SPLITS=10 \\
  N_WORKERS=10 \\
    .venv/bin/python experiments/reasoning_pretraining/code_ladder/data/split_jsonl_gz.py
"""

import gzip
import multiprocessing as mp
import os
from pathlib import Path

SRC = Path(os.environ["SRC"])
DST = Path(os.environ["DST"])
N_SPLITS = int(os.environ.get("N_SPLITS", "10"))
N_WORKERS = int(os.environ.get("N_WORKERS", "10"))
DST.mkdir(parents=True, exist_ok=True)


def split_one(src_path: Path) -> tuple[str, int]:
    """Split one .jsonl.gz into N_SPLITS files under DST, suffixed -split-X."""
    base = src_path.name.replace(".jsonl.gz", "")
    out_paths = [DST / f"{base}-split-{i:02d}.jsonl.gz" for i in range(N_SPLITS)]
    if all(p.exists() and p.stat().st_size > 0 for p in out_paths):
        return src_path.name, -1  # already split
    handles = [gzip.open(p, "wt", encoding="utf-8") for p in out_paths]
    n_rows = 0
    try:
        with gzip.open(src_path, "rt", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                handles[i % N_SPLITS].write(line)
                n_rows += 1
    finally:
        for h in handles:
            h.close()
    return src_path.name, n_rows


def main():
    srcs = sorted(SRC.glob("*.jsonl.gz"))
    if not srcs:
        print(f"ERROR: no jsonl.gz under {SRC}")
        return
    print(f"splitting {len(srcs)} files → {DST} ({N_SPLITS} splits each, {N_WORKERS} workers)", flush=True)
    with mp.Pool(N_WORKERS) as pool:
        for name, n in pool.imap_unordered(split_one, srcs):
            print(f"  {name}: {n:,} rows split", flush=True)
    print(f"done → {DST}", flush=True)


if __name__ == "__main__":
    main()
