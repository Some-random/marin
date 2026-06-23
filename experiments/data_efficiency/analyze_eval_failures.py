#!/usr/bin/env python
"""Triage failed lm-eval tasks so failures get fixed, not blindly retried.

Scans an eval results dir for task subdirs that produced NO results JSON, extracts the
first real error from each task's log (skipping the misleading torchrun
`<NO_OTHER_FAILURES>` summary), classifies the root cause against a taxonomy of failures
we've actually hit, writes `FAILURES.md` into the results dir, and prints a summary table.

Usage:
    python analyze_eval_failures.py <results_dir> [--expected t1,t2,...] [--quiet]

Exit code is 1 if any task failed (0 otherwise), so a runner can gate on it.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from enum import StrEnum

MAX_EXCERPT_LINES = 45
MAX_LOG_BYTES = 8 * 1024 * 1024  # cap NCCL_DEBUG=ALL logs so we don't read GBs


class FailureClass(StrEnum):
    NCCL_GATHER_OOM = "NCCL_GATHER_OOM"
    CUDA_OOM = "CUDA_OOM"
    CODE_EVAL_CACHE = "CODE_EVAL_CACHE"
    MAX_GEN_LEN = "MAX_GEN_LEN"
    OFFLINE_MODE = "OFFLINE_MODE"
    HUB_CONN = "HUB_CONN"
    DATASET_MISSING = "DATASET_MISSING"
    SIGBUS = "SIGBUS"
    KILLED = "KILLED"
    NCCL_OTHER = "NCCL_OTHER"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"
    NO_LOG = "NO_LOG"


@dataclass(frozen=True)
class Rule:
    cls: FailureClass
    transient: bool
    fix: str
    all_of: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()

    def matches(self, text: str) -> bool:
        if not all(re.search(p, text) for p in self.all_of):
            return False
        if self.any_of and not any(re.search(p, text) for p in self.any_of):
            return False
        return True


# Ordered most-specific first; first match wins. Signatures are taken verbatim from real
# failure logs (see OPS.md). NCCL_GATHER_OOM must precede CUDA_OOM because the gather OOM
# log also contains an out-of-memory string.
RULES: tuple[Rule, ...] = (
    Rule(
        FailureClass.NCCL_GATHER_OOM, transient=False,
        fix="Confirm `export NCCL_P2P_DISABLE=1` runs before `accelerate launch` (it is in all 5 eval "
            "runners — grep to verify it wasn't dropped). This is the P2P/CUMEM gather OOM, see OPS.md.",
        all_of=(r"gather_object",),
        any_of=(r"Cuda failure 2 'out of memory'", r"unhandled cuda error"),
    ),
    Rule(
        FailureClass.CUDA_OOM, transient=False,
        fix="Real GPU-memory OOM (not the gather bug). Lower the batch size — e.g. BATCH_SIZE=4 for 4B models.",
        any_of=(r"torch\.OutOfMemoryError", r"CUDA out of memory", r"Tried to allocate \d"),
    ),
    Rule(
        FailureClass.CODE_EVAL_CACHE, transient=False,
        fix="mbpp/humaneval HF `code_eval` cache collides under torchrun (per-rank metrics file race). Run via "
            "convert_and_eval_v2.sh (per-rank HF_METRICS_CACHE) or with --num_processes 1.",
        all_of=(r"code_eval",),
        any_of=(r"FileExistsError", r"already exists", r"HF_ALLOW_CODE_EVAL",
                r"FileNotFoundError", r"default_experiment.*\.arrow"),
    ),
    Rule(
        FailureClass.MAX_GEN_LEN, transient=False,
        fix="Generation-length config: requested max_gen_toks >= the model's max_length. Lower gen_kwargs "
            "max_gen_toks (or raise the model max_length) for this task — seen on mmlu_pro/bbh.",
        any_of=(r"must be less than model's maximum sequence length",
                r"Invalid configuration: requested max tokens to generate"),
    ),
    Rule(
        FailureClass.OFFLINE_MODE, transient=False,
        fix="Set HF_DATASETS_OFFLINE=0 and HF_HUB_OFFLINE=0 — paloma/gsm fetch builder scripts from the Hub.",
        any_of=(r"OfflineModeIsEnabled", r"Couldn't reach .* on the Hub"),
    ),
    Rule(
        FailureClass.HUB_CONN, transient=True,
        fix="Hub/network flakiness — safe to retry once.",
        any_of=(r"ConnectionError", r"Max retries exceeded", r"ReadTimeout", r"\b504\b", r"HTTPError",
                r"Connection reset"),
    ),
    Rule(
        FailureClass.DATASET_MISSING, transient=False,
        fix="Dataset/path missing — check the task YAML / dataset name. Not a retry candidate.",
        any_of=(r"Couldn't find a dataset script", r"doesn't exist on the Hub",
                r"FileNotFoundError.*\.(json|parquet|yaml)"),
    ),
    Rule(
        FailureClass.SIGBUS, transient=True,
        fix="SIGBUS / signal 7 — usually a Lustre/mmap fault (often in the HF metrics-cache finalize for "
            "mbpp/humaneval). Re-run on a clean node; if it recurs on code tasks use per-rank HF_METRICS_CACHE "
            "(convert_and_eval_v2.sh).",
        any_of=(r"SIGBUS", r"[Ss]ignal 7\b", r"Bus error", r"exitcode\s*:?\s*-7"),
    ),
    Rule(
        FailureClass.KILLED, transient=True,
        fix="Process killed (OOM-killer / SIGKILL) — likely node memory pressure. Re-run on a clean node "
            "(nvidia-smi shows ~0 MiB used).",
        any_of=(r"\bKilled\b", r"signal 9 \(SIGKILL\)", r"received signal 9", r"out of memory; killing"),
    ),
    Rule(
        FailureClass.NCCL_OTHER, transient=True,
        fix="NCCL error not matching the known gather bug — capture NCCL_DEBUG=INFO on a re-run; may be a "
            "transient fabric issue.",
        any_of=(r"NCCL error", r"NCCL Error", r"ncclInternalError", r"ncclSystemError", r"ncclUnhandledCudaError"),
    ),
    Rule(
        FailureClass.TIMEOUT, transient=True,
        fix="Timed out — re-run; if it recurs the task is too slow for the batch/node.",
        any_of=(r"\bTimeoutError\b", r"timed out", r"Watchdog caught collective"),
    ),
)


@dataclass(frozen=True)
class Verdict:
    task: str
    cls: FailureClass
    transient: bool
    fix: str
    excerpt: str
    log_path: str


def _has_result(task_dir: str) -> bool:
    """A task passed if it wrote a results JSON (or, for bigcode, a non-empty metrics.json)."""
    if glob.glob(os.path.join(task_dir, "**", "results_*.json"), recursive=True):
        return True
    if "bigcode" in os.path.basename(task_dir):
        metrics = glob.glob(os.path.join(task_dir, "**", "metrics.json"), recursive=True)
        return any(os.path.getsize(m) > 0 for m in metrics)
    return False


def find_failed_tasks(results_dir: str) -> list[str]:
    """Immediate subdirs of results_dir that produced no result. Sorted for stable output."""
    failed = []
    for entry in sorted(os.listdir(results_dir)):
        task_dir = os.path.join(results_dir, entry)
        if os.path.isdir(task_dir) and not _has_result(task_dir):
            failed.append(task_dir)
    return failed


def find_log(task_dir: str) -> str | None:
    """Locate the most informative log for a task.

    Runners write the per-task log either as a sibling file (`<task_dir>.log`) or inside the
    dir (`refill.log`, `eval.log`, `run.log`). Pick the largest non-empty candidate.
    """
    candidates = [task_dir + ".log"]
    candidates += glob.glob(os.path.join(task_dir, "*.log"))
    candidates += glob.glob(os.path.join(task_dir, "**", "*.log"), recursive=True)
    present = [c for c in dict.fromkeys(candidates) if os.path.isfile(c) and os.path.getsize(c) > 0]
    if not present:
        return None
    return max(present, key=os.path.getsize)


def read_log(log_path: str) -> str:
    """Read up to MAX_LOG_BYTES; for oversized logs keep the tail (errors land at the end)."""
    size = os.path.getsize(log_path)
    with open(log_path, errors="replace") as f:
        if size > MAX_LOG_BYTES:
            f.seek(size - MAX_LOG_BYTES)
        return f.read()


def extract_first_error(text: str) -> str:
    """Return a short excerpt around the FIRST real Python traceback, not the torchrun summary."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\[rank\d+\]: Traceback", line) or line.startswith("Traceback (most recent call last)"):
            start = i
            break
    if start is None:
        for i, line in enumerate(lines):
            if re.search(r"Cuda failure|OutOfMemoryError|NCCL Error|OfflineModeIsEnabled|Error:", line):
                start = max(0, i - 2)
                break
    if start is None:
        return "\n".join(lines[-MAX_EXCERPT_LINES:]).strip()
    return "\n".join(lines[start:start + MAX_EXCERPT_LINES]).strip()


def classify(text: str) -> tuple[FailureClass, bool, str]:
    for rule in RULES:
        if rule.matches(text):
            return rule.cls, rule.transient, rule.fix
    return (
        FailureClass.UNKNOWN, False,
        "No known signature matched — read the excerpt and the full log; do NOT auto-retry an unknown failure.",
    )


def build_verdict(task_dir: str) -> Verdict:
    task = os.path.basename(task_dir)
    log_path = find_log(task_dir)
    if log_path is None:
        return Verdict(
            task, FailureClass.NO_LOG, False,
            "No log captured — the task likely never launched (node busy, ssh failed, or it crashed before "
            "redirecting output). Check the parent runner log and node availability.",
            "(no log file found)", "",
        )
    text = read_log(log_path)
    cls, transient, fix = classify(text)
    return Verdict(task, cls, transient, fix, extract_first_error(text), log_path)


def render_report(results_dir: str, verdicts: list[Verdict], now: str) -> str:
    n_transient = sum(v.transient for v in verdicts)
    n_permanent = len(verdicts) - n_transient
    out = [
        f"# Eval failure triage — {os.path.basename(results_dir.rstrip('/'))}",
        f"Generated {now}",
        "",
        f"**{len(verdicts)} task(s) failed** (no results JSON): "
        f"{n_transient} transient (safe to retry) · {n_permanent} permanent (fix before retry).",
        "",
    ]
    for v in verdicts:
        tag = "transient — retry OK" if v.transient else "permanent — FIX FIRST"
        out += [
            f"## {v.task}  —  `{v.cls}`  ({tag})",
            f"**Fix:** {v.fix}",
            f"**Log:** `{v.log_path or 'n/a'}`",
            "",
            "```",
            v.excerpt,
            "```",
            "",
        ]
    return "\n".join(out)


def print_summary(verdicts: list[Verdict]) -> None:
    print(f"\n=== eval failure triage: {len(verdicts)} failed task(s) ===")
    width = max((len(v.task) for v in verdicts), default=4)
    for v in verdicts:
        flag = "RETRY-OK " if v.transient else "FIX-FIRST"
        print(f"  {v.task:<{width}}  {v.cls:<16}  [{flag}]")
    n_transient = sum(v.transient for v in verdicts)
    print(f"  -> {n_transient} transient, {len(verdicts) - n_transient} permanent. "
          f"See FAILURES.md for the diagnosis + fix per task.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--expected", default="", help="comma-separated task names that should exist")
    ap.add_argument("--now", default="", help="timestamp string (caller supplies; avoids tz guesswork)")
    ap.add_argument("--quiet", action="store_true", help="write FAILURES.md but skip the stdout table")
    ap.add_argument("--no-write", action="store_true", help="classify + print only; don't write FAILURES.md")
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"results_dir not found: {args.results_dir}")
        return 2

    failed_dirs = find_failed_tasks(args.results_dir)
    verdicts = [build_verdict(d) for d in failed_dirs]

    # Expected tasks that never even produced a directory.
    if args.expected:
        present = {os.path.basename(d) for d in failed_dirs}
        present |= {
            os.path.basename(p) for p in glob.glob(os.path.join(args.results_dir, "*")) if os.path.isdir(p)
        }
        for task in (t.strip() for t in args.expected.split(",") if t.strip()):
            if not any(task == d or d.endswith("__" + task) or task in d for d in present):
                verdicts.append(Verdict(
                    task, FailureClass.NO_LOG, False,
                    "Expected task produced no output directory at all — its launch did not start "
                    "(node busy / ssh failed / wrong task name).",
                    "(no output directory)", "",
                ))

    if not verdicts:
        if not args.quiet:
            print("=== eval failure triage: no failed tasks (all produced results) ===")
        return 0

    report = render_report(args.results_dir, verdicts, args.now or "(timestamp not supplied)")
    report_path = os.path.join(args.results_dir, "FAILURES.md")
    if not args.no_write:
        with open(report_path, "w") as f:
            f.write(report)
    if not args.quiet:
        print_summary(verdicts)
        print(f"  {'(--no-write) would write' if args.no_write else 'wrote'} {report_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
