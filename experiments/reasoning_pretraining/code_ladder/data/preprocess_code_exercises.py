# Combine the {problem, solution} fields in jinaai/code_exercises into a single
# `text` field so the standard tokenizer can ingest it. Output: a JSONL.gz
# sibling directory we can point default_tokenize at.

import gzip
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SRC = Path("/fsx/users/dongweij/marin/outputs/raw/phi_1_5_mix/code_exercises/data")
DST = Path("/fsx/users/dongweij/marin/outputs/raw/phi_1_5_mix/code_exercises_text")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    paths = sorted(SRC.glob("train-*.parquet"))
    log.info(f"merging {len(paths)} parquet files from {SRC}")
    total = 0
    for i, p in enumerate(paths):
        t = pq.read_table(p, columns=["problem", "solution"])
        problems = t["problem"].to_pylist()
        solutions = t["solution"].to_pylist()
        out_path = DST / f"code_exercises_{i:05d}.jsonl.gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            for prob, sol in zip(problems, solutions, strict=True):
                # Match phi-1-style: docstring/problem followed by the solution body.
                text = f"{prob}\n{sol}\n"
                f.write(json.dumps({"text": text}) + "\n")
        n = len(problems)
        total += n
        log.info(f"wrote {out_path.name} ({n:,} records, total so far {total:,})")
    log.info(f"DONE: {total:,} records across {len(paths)} files -> {DST}")


if __name__ == "__main__":
    main()
