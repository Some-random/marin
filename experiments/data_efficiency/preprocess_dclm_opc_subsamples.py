# Cap DCLM-164k-docs and opc_algorithmic to exact-epoch-matched sizes for
# the corrected 75% DCLM / 25% opc mix experiment.
#
# Targets:
#   DCLM: ~150 M tokens  →  ~121,500 docs (164,459 total × 73.9%)
#   opc:  ~50 M tokens   →  ~282,500 docs (5,322,920 total × 5.3%)
#
# With those slices + 75/25 weights + 12,800 steps × 64 × 4096 budget:
#   - DCLM seen 16.78 epochs
#   - opc seen 16.78 epochs
#   - Matched epochs ✓; same total training tokens as baseline (3.36 B).

import gzip
import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OUT = Path("/fsx/users/dongweij/marin/outputs/raw")

DCLM_DOCS = 121_500
OPC_DOCS = 282_500


def slice_dclm() -> None:
    out_path = OUT / "dclm_150m.jsonl.gz"
    if out_path.exists():
        log.info(f"DCLM slice already exists at {out_path}; skipping")
        return
    log.info(f"loading konwoo/dclm-164k-docs-train (full 164k docs)")
    ds = load_dataset("konwoo/dclm-164k-docs-train", split="train")
    log.info(f"slicing to first {DCLM_DOCS:,} docs (target ~150 M tokens)")
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= DCLM_DOCS:
                break
            f.write(json.dumps({"text": ex["text"]}) + "\n")
    log.info(f"wrote {out_path} ({DCLM_DOCS:,} docs)")


def slice_opc() -> None:
    out_path = OUT / "opc_algorithmic_50m.jsonl.gz"
    src_path = OUT / "opc_algorithmic.jsonl.gz"
    if out_path.exists():
        log.info(f"opc slice already exists at {out_path}; skipping")
        return
    log.info(f"reading {src_path}")
    written = 0
    with gzip.open(src_path, "rt", encoding="utf-8") as fin, gzip.open(out_path, "wt", encoding="utf-8") as fout:
        for line in fin:
            if written >= OPC_DOCS:
                break
            fout.write(line)
            written += 1
    log.info(f"wrote {out_path} ({written:,} docs)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    slice_dclm()
    slice_opc()


if __name__ == "__main__":
    main()
