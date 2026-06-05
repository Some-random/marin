"""Download StarCoderData per-language slices at Aryabumi-Table-3/4 ratios.

Aryabumi (arxiv 2408.10914) uses Stack v1 filtered by Starcoder. StarCoderData
is the Starcoder-published artifact of that filtering, so it's the closest
public mirror of Aryabumi's exact code source.

Per-language token targets (15B Stack + 4B markup = 19B unique tokens for C5):
    Stack (multi-language, Aryabumi Table 3, top-10 re-normalized):
        java        2.685B,  javascript 2.642B,  php   2.153B,
        python      1.659B,  c-sharp    1.434B,  typescript 1.368B,
        c           1.146B,  cpp        0.849B,  go    0.603B,  ruby  0.465B
    Markup (Aryabumi Table 4, top-5 re-normalized):
        markdown    2.400B,  yaml   0.477B,  json  0.441B,  html  0.379B,  css  0.303B

We download enough parquet files per language to cover the target, with a
~20% over-download margin. Each parquet has a `content` field that we convert
to a jsonl.gz where each row is {"text": <content>}. Tokenization (Llama-3.1)
runs separately via `code_data_aryabumi.py`.

Run via:
    cd /fsx/users/dongweij/marin
    .venv/bin/python -m experiments.data_efficiency.download_starcoderdata
"""

import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "bigcode/starcoderdata"
RAW_DIR = Path("/fsx/users/dongweij/marin/outputs/raw/starcoderdata")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Per-language target token counts (rounded up ~20% for safety).
# Char-per-token heuristic: ~4 (English) / ~3 (code) — we'll over-download
# slightly and stop when the jsonl total characters / 3 >= target.
TARGETS = {
    # Stack web code (Aryabumi Table 3, top-10 re-normalized to sum to 100%)
    "java":        3_200_000_000,
    "javascript":  3_200_000_000,
    "php":         2_600_000_000,
    "python":      2_000_000_000,
    "c-sharp":     1_700_000_000,
    "typescript":  1_700_000_000,
    "c":           1_400_000_000,
    "cpp":         1_000_000_000,
    "go":            720_000_000,
    "ruby":          560_000_000,
    # Markup (Aryabumi Table 4, top-5 re-normalized to sum to 100%)
    "markdown":    2_900_000_000,
    "yaml":          575_000_000,
    "json":          530_000_000,
    "html":          455_000_000,
    "css":           365_000_000,
}
CHAR_PER_TOKEN = 3.0  # conservative for code; will over-download not under


def list_files(language: str) -> list[str]:
    """List parquet files in a language directory of starcoderdata."""
    from huggingface_hub import HfApi
    api = HfApi()
    all_files = api.list_repo_files(REPO, repo_type="dataset")
    return sorted([f for f in all_files if f.startswith(f"{language}/") and f.endswith(".parquet")])


def download_one(language: str, parquet_path: str) -> Path:
    """Download a single parquet to raw/starcoderdata/<language>/<file>."""
    local_path = RAW_DIR / language / Path(parquet_path).name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        return local_path
    hf_hub_download(
        REPO,
        parquet_path,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
        local_dir_use_symlinks=False,
    )
    return local_path


def convert_parquet_to_jsonl(parquet_path: Path, out_handle, char_budget_remaining: int) -> tuple[int, int]:
    """Append rows of parquet (as {"text": content}) to jsonl handle until
    char budget is hit or parquet exhausted. Returns (rows_written, chars_written).
    """
    table = pq.read_table(parquet_path, columns=["content"])
    content_arr = table.column("content")
    rows = 0
    chars = 0
    for c in content_arr:
        s = c.as_py()
        if s is None:
            continue
        out_handle.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
        rows += 1
        chars += len(s)
        if chars >= char_budget_remaining:
            break
    return rows, chars


def fetch_language(language: str, target_tokens: int) -> None:
    """Download + convert one language until target_tokens of text is collected."""
    out_jsonl = RAW_DIR / f"{language}.jsonl.gz"
    if out_jsonl.exists():
        # rough check: if file already exists and is roughly the target size, skip
        size_bytes = out_jsonl.stat().st_size
        # compressed gz is ~3x smaller; uncompressed ≈ size_bytes * 3
        est_chars = size_bytes * 3
        est_tokens = est_chars / CHAR_PER_TOKEN
        if est_tokens >= target_tokens * 0.9:
            print(f"[{language}] already have ~{est_tokens/1e9:.2f}B tokens, skipping")
            return
    print(f"[{language}] target: {target_tokens/1e9:.2f}B tokens")
    files = list_files(language)
    if not files:
        print(f"[{language}] NO PARQUET FILES — skipping")
        return
    char_target = int(target_tokens * CHAR_PER_TOKEN)
    chars_so_far = 0
    rows_so_far = 0
    t0 = time.time()
    with gzip.open(out_jsonl, "wt", encoding="utf-8") as out:
        for i, pf in enumerate(files):
            try:
                local = download_one(language, pf)
            except Exception as e:
                print(f"[{language}] download {pf} FAILED: {e}")
                continue
            remaining = char_target - chars_so_far
            rows, chars = convert_parquet_to_jsonl(local, out, remaining)
            chars_so_far += chars
            rows_so_far += rows
            elapsed = time.time() - t0
            print(f"[{language}] file {i+1}/{len(files)}: +{rows:,} rows, "
                  f"+{chars/1e9:.2f}B chars; total {rows_so_far:,} rows / "
                  f"{chars_so_far/CHAR_PER_TOKEN/1e9:.2f}B tokens / {elapsed:.0f}s")
            # delete the parquet after consumption to save disk
            try:
                local.unlink()
            except Exception:
                pass
            if chars_so_far >= char_target:
                print(f"[{language}] target reached, stopping")
                break
    print(f"[{language}] DONE: {rows_so_far:,} rows, ~{chars_so_far/CHAR_PER_TOKEN/1e9:.2f}B tokens")


def main():
    # Run all languages in parallel (5 at a time to avoid HF rate limits + disk pressure).
    with ProcessPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_language, lang, tgt): lang for lang, tgt in TARGETS.items()}
        for f in as_completed(futures):
            lang = futures[f]
            try:
                f.result()
            except Exception as e:
                print(f"[{lang}] failed: {e}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
