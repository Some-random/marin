#!/usr/bin/env python3
"""Sample raw text documents from SlimPajama-NL (per source) and DCLM-baseline,
write to outputs/eval_results/sp_vs_dclm_samples.md for visual comparison.

For each source: pick 3 random docs, truncate at 1500 chars, label by source.
Goal: let a human eyeball whether SlimPajama-NL is actually 'better text' than DCLM.
"""

import gzip
import json
import random
from pathlib import Path

REPO = Path("/fsx/users/dongweij/marin")
SP_NL_DIR = REPO / "outputs" / "raw" / "slimpajama_nl"
DCLM_RAW = REPO / "outputs" / "raw" / "dclm_baseline"  # may not exist; fall back to jsonl shards
OUT = REPO / "outputs" / "eval_results" / "sp_vs_dclm_samples.md"

random.seed(42)

CHAR_LIMIT = 1500
N_PER_SOURCE = 3


def find_dclm_jsonl() -> list[Path]:
    """DCLM raw data may live at various paths. Probe a few likely ones."""
    candidates = [
        REPO / "outputs" / "raw" / "dclm_1500m.jsonl",
        REPO / "outputs" / "raw" / "dclm_150m.jsonl.gz",
        REPO / "outputs" / "raw" / "dclm_baseline",
    ]
    files: list[Path] = []
    for c in candidates:
        if c.is_file():
            files.append(c)
        elif c.is_dir():
            files.extend(c.glob("**/*.jsonl*"))
    return files


def sample_sp_by_source(n_per_source: int = N_PER_SOURCE) -> dict[str, list[str]]:
    """Pick docs from a few jsonl.gz shards, group by source."""
    shards = sorted(SP_NL_DIR.glob("*.jsonl.gz"))
    if not shards:
        return {}
    random.shuffle(shards)
    out: dict[str, list[str]] = {
        "RedPajamaCommonCrawl": [],
        "RedPajamaC4": [],
        "RedPajamaBook": [],
        "RedPajamaArXiv": [],
        "RedPajamaWikipedia": [],
    }
    for shard in shards:
        if all(len(v) >= n_per_source for v in out.values()):
            break
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                s = d["source"]
                if s in out and len(out[s]) < n_per_source:
                    out[s].append(d["text"][:CHAR_LIMIT])
                if all(len(v) >= n_per_source for v in out.values()):
                    break
    return out


def sample_dclm(n: int = 5) -> list[str]:
    """Pick N docs from DCLM jsonl(.gz). Returns text strings."""
    files = find_dclm_jsonl()
    if not files:
        return []
    random.shuffle(files)
    samples: list[str] = []
    for fp in files:
        opener = gzip.open if fp.suffix in (".gz", ".gzip") else open
        try:
            with opener(fp, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    text = d.get("text") or d.get("content")
                    if text:
                        samples.append(text[:CHAR_LIMIT])
                        if len(samples) >= n:
                            return samples
        except Exception:
            continue
    return samples


def write_md(sp_samples: dict[str, list[str]], dclm_samples: list[str]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# SlimPajama-NL vs DCLM-baseline — raw doc samples\n\n")
        f.write("Goal: eyeball whether SlimPajama-NL is actually 'better text' than DCLM-baseline.\n")
        f.write("Each doc truncated to 1500 chars. 3 docs per SlimPajama source.\n\n")
        f.write("---\n\n")
        f.write("## DCLM-baseline (CommonCrawl, filtered)\n\n")
        if dclm_samples:
            for i, t in enumerate(dclm_samples, 1):
                f.write(f"### DCLM doc {i}\n\n```\n{t}\n```\n\n")
        else:
            f.write("_(could not locate raw DCLM jsonl on disk; only the tokenized cache is present)_\n\n")
        f.write("---\n\n")
        f.write("## SlimPajama-NL (after dropping GitHub + StackExchange)\n\n")
        for src, docs in sp_samples.items():
            short = src.replace("RedPajama", "")
            f.write(f"### {short}\n\n")
            for i, t in enumerate(docs, 1):
                f.write(f"#### {short} doc {i}\n\n```\n{t}\n```\n\n")
    print(f"wrote: {OUT}")


def main():
    sp = sample_sp_by_source()
    dclm = sample_dclm(n=5)
    write_md(sp, dclm)
    print(f"SlimPajama-NL: {sum(len(v) for v in sp.values())} docs across {len(sp)} sources")
    print(f"DCLM: {len(dclm)} docs")


if __name__ == "__main__":
    main()
