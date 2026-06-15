#!/usr/bin/env python3
"""eval_section3.py — one-shot tool for filling §3 of EVALUATION.md.

Goals (after 13 rounds of patching):
  - For one or more model checkpoints, run the canonical v2 suite (in parallel
    if multiple free nodes are available).
  - Extract every §3 task with a metric-fallback list that handles all the
    lm-eval version variations (bbh, mmlu_pro, lambada acc-not-ppl, mmlu mean,
    bigcode HE, math metric prefs).
  - Validate the §3 table structure (pipe count, alignment, category headers
    with footnote markers, Mean row placement) before writing.
  - Recompute the 5 Mean rows in place using the canonical category map.
  - Self-test on a sample row before committing.

Usage:
  # Evaluate one model end-to-end (run suite + extract + update §3 + validate)
  python eval_section3.py run <LABEL> <LEVANTER_DIR_OR_HF_DIR> [--node NODE]

  # Evaluate several models in parallel across free nodes
  python eval_section3.py run-many spec.json

  # Just extract + update §3 from an existing results dir (no re-run)
  python eval_section3.py fill-from-results <RESULTS_DIR> <COL_LABEL>

  # Validate the table after manual edits
  python eval_section3.py validate

The skill .agents/skills/eval-for-section3/SKILL.md invokes this tool.
"""

import argparse
import glob
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path("/fsx/users/dongweij/marin")
EVAL_DIR = REPO / "outputs" / "eval_results"
CKPT_DIR = REPO / "checkpoints"
LOG_DIR = REPO / "logs"
MD = REPO / "experiments" / "data_efficiency" / "EVALUATION.md"
EVAL_SCRIPT = REPO / "experiments" / "data_efficiency" / "convert_and_eval_v2.sh"
RUN_EVAL_V2 = REPO / "experiments" / "data_efficiency" / "run_eval_v2.sh"
RUN_PALOMA = REPO / "experiments" / "data_efficiency" / "run_paloma_for_model.sh"
RUN_GSM = REPO / "experiments" / "data_efficiency" / "run_gsm_for_model.sh"

DEFAULT_NODE_POOL = [
    "gpu-st-p4d24xlarge-1", "gpu-st-p4d24xlarge-2",
    "gpu-st-p4d24xlarge-3", "gpu-st-p4d24xlarge-4",
    "gpu-dy-p4d24xlarge-1", "gpu-dy-p4d24xlarge-2",
    "gpu-dy-p4d24xlarge-3", "gpu-dy-p4d24xlarge-4",
    "gpu-dy-p4d24xlarge-5", "gpu-dy-p4d24xlarge-8",
    "gpu-dy-p4d24xlarge-9",
]


# ====================================================================
# Canonical §3 row config — single source of truth for every task we
# care about. `metric_prefer` handles lm-eval version variations.
# ====================================================================
@dataclass
class TaskRow:
    label: str            # exact §3 row label (left column)
    lm_eval_task: str     # task name as lm-eval reports it
    metric_prefer: tuple  # ordered list of metric keys to try
    runs_in_v2_suite: bool = True  # if False, requires a separate runner


TASKS: list[TaskRow] = [
    TaskRow("sciq[0]", "sciq", ("acc,none",)),
    TaskRow("boolq[0]", "boolq", ("acc,none",)),
    TaskRow("piqa[0]", "piqa", ("acc,none",)),
    # acc_norm is the canonical reporting metric for uneven-length multi-choice
    # tasks (HF open-llm-leaderboard convention). Prefer acc_norm,none; fall back
    # to acc,none for older result JSONs that didn't log acc_norm.
    TaskRow("openbookqa_fact[0]", "openbookqa_fact", ("acc_norm,none", "acc,none")),
    TaskRow("arc_easy[25]", "arc_easy", ("acc,none",)),
    TaskRow("arc_challenge[25]", "arc_challenge", ("acc_norm,none", "acc,none")),
    TaskRow("hellaswag[10]", "hellaswag", ("acc_norm,none", "acc,none")),
    TaskRow("winogrande[5]", "winogrande", ("acc,none",)),
    TaskRow("mmlu[5]", "_mmlu_mean", ()),                         # special: average mmlu_<subject>
    TaskRow("commonsense_qa[0]", "commonsense_qa", ("acc,none",)),
    TaskRow("social_iqa[0]", "social_iqa", ("acc,none",)),
    TaskRow("logiqa[0]", "logiqa", ("acc_norm,none", "acc,none")),
    TaskRow("lambada_openai[0]", "lambada_openai", ("acc,none",)),  # NOT perplexity
    TaskRow("copa[0]", "copa", ("acc,none",)),
    TaskRow("wsc[0]", "wsc", ("acc,none",)),
    TaskRow("storycloze_2018_local[0]", "storycloze_2018_local", ("acc,none",)),
    TaskRow("cb[0]", "cb", ("acc,none",)),
    TaskRow("quac_first_turn[0]", "quac_first_turn", ("f1,none",)),
    TaskRow("agieval_lsat_ar[0]", "agieval_lsat_ar", ("acc_norm,none", "acc,none")),
    TaskRow("gpqa_diamond[0]", "gpqa_diamond_zeroshot", ("acc,none",)),
    TaskRow("bbh[3] (limit=0.1)", "bbh", (
        "exact_match,get-answer",       # current lm-eval (June 2026)
        "exact_match,strict-match",     # older lm-eval
        "acc,none",                     # fallback
    )),
    TaskRow("mmlu_pro[5] (limit=0.1)", "mmlu_pro", (
        "exact_match,custom-extract",   # current
        "exact_match,strict-match",     # older
        "acc,none",                     # fallback
    )),
    TaskRow("gsm8k[5]", "gsm8k", ("exact_match,strict-match",)),
    TaskRow("gsm8k_cot[8]", "gsm8k_cot", ("exact_match,strict-match",)),
    TaskRow("minerva_math[4]", "minerva_math", (
        "exact_match,none",             # matches existing §3 column convention
        "math_verify,none",             # newer lm-eval default; more permissive
    )),
    TaskRow("humaneval[0] (lm-eval)", "humaneval", ("pass@1,create_test",)),
    TaskRow("humaneval[0] (bigcode) ‡‡", "_bigcode_he", ()),  # special: read bigcode metrics.json
    TaskRow("mbpp[3]", "mbpp", ("pass_at_1,none",)),
    # Perplexity rows are NOT run by run_eval_v2.sh — they come from
    # Levanter in-training eval. Listed here for completeness only.
    TaskRow("gsm_symbolic_main[8]", "gsm_symbolic_main", ("exact_match,strict-match",), runs_in_v2_suite=False),
    TaskRow("gsm_noop[8]", "gsm_noop", ("exact_match,strict-match",), runs_in_v2_suite=False),
    TaskRow("dclm_200m_val (nats)", "_dclm_in_training", (), runs_in_v2_suite=False),
    TaskRow("paloma_macro (bpb)", "_paloma_macro", (), runs_in_v2_suite=False),
]


# ====================================================================
# Mean rows config — single source of truth for which tasks aggregate
# into which Mean. Each Mean row is inserted right after its category's
# last task row.
# ====================================================================
MEAN_ROWS = [
    ("Mean Open-book",
     "Open-book",
     ["sciq[0]", "boolq[0]", "piqa[0]", "openbookqa_fact[0]"]),
    ("Mean Closed-book NL",
     "Closed-book NL",
     ["arc_easy[25]", "arc_challenge[25]", "hellaswag[10]", "winogrande[5]", "mmlu[5]",
      "commonsense_qa[0]", "social_iqa[0]", "logiqa[0]", "lambada_openai[0]", "copa[0]", "wsc[0]",
      "storycloze_2018_local[0]", "cb[0]", "quac_first_turn[0]"]),
    ("Mean Aggregate",
     "Aggregate",
     ["agieval_lsat_ar[0]", "gpqa_diamond[0]", "bbh[3] (limit=0.1)", "mmlu_pro[5] (limit=0.1)"]),
    ("Mean Math (standard)",
     "Math (standard)",
     ["gsm8k[5]", "gsm8k_cot[8]", "minerva_math[4]"]),
    ("Mean Math (perturbation-robust)",
     "Math (perturbation-robust)",
     ["gsm_symbolic_main[8]", "gsm_noop[8]"]),
    ("Mean Code",
     "Code",
     ["humaneval[0] (lm-eval)", "humaneval[0] (bigcode) ‡‡", "mbpp[3]"]),
]


# ====================================================================
# Helpers
# ====================================================================
def find_v2_score(results_dir: Path, task: TaskRow):
    """Special-case handlers + generic lookup."""
    if task.lm_eval_task == "_mmlu_mean":
        vals = []
        for jp in results_dir.glob("**/*results*.json"):
            try:
                d = json.load(open(jp))
            except Exception:
                continue
            for tname, m in d.get("results", {}).items():
                if (tname.startswith("mmlu_") and
                    tname not in ("mmlu_humanities", "mmlu_social_sciences", "mmlu_stem", "mmlu_other") and
                    "acc,none" in m):
                    vals.append(m["acc,none"])
        return sum(vals) / len(vals) if vals else None

    if task.lm_eval_task == "_bigcode_he":
        bc = results_dir / "bigcode_humaneval" / "metrics.json"
        if bc.exists():
            try:
                d = json.load(open(bc))
                return d.get("humaneval", {}).get("pass@1")
            except Exception:
                return None
        return None

    if not task.runs_in_v2_suite:
        return None

    for jp in results_dir.glob("**/*results*.json"):
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        for tname, m in d.get("results", {}).items():
            if tname != task.lm_eval_task:
                continue
            for metric in task.metric_prefer:
                if metric in m and isinstance(m[metric], float):
                    return m[metric]
    return None


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if v >= 1.0:
        return f"{v:.3f}"
    if v < 0.01 and v != 0:
        return f"{v:.3f}"
    return f"{v:.3f}"


def extract_cell_float(cell: str):
    s = cell.strip().replace("**", "").replace(" ¶", "").replace("*", "").strip()
    if s in ("", "—") or "n/a" in s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_row_label(cells):
    if len(cells) < 2:
        return None
    s = cells[1].strip()
    # Strip leading/trailing ** but preserve trailing footnote markers like ° ‡‡
    s = re.sub(r"^\*\*|\*\*$", "", s)
    return s


def is_category_header_label(label: str) -> bool:
    """Category headers are labels like '**Open-book**' or '**Math (perturbation-robust)** °'.
    They may have a trailing footnote marker after the closing **."""
    if not label:
        return False
    # After parse_row_label strips outer **, label is like 'Open-book' or 'Math (perturbation-robust) °'
    # Differentiate from task rows like 'sciq[0]' or 'humaneval[0] (bigcode) ‡‡' by:
    # - category labels don't contain '[' or numeric metric markers
    if "[" in label:
        return False
    return True


def find_section3_table(lines):
    """Returns (header_idx, table_end_idx_exclusive)."""
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("| Task | base (x16)"):
            header_idx = i
            break
    if header_idx is None:
        return None, None
    table_end = None
    for i in range(header_idx + 1, len(lines)):
        if not lines[i].startswith("|"):
            table_end = i
            break
    if table_end is None:
        table_end = len(lines)
    return header_idx, table_end


def validate_table(lines, strict: bool = False):
    """Returns list of validation errors; empty if OK.

    When strict=True, also asserts that EVERY (task, column) pair has a real
    value unless one of these is true:
      - the task is in EXPECTED_BLANKS (runs_in_v2_suite=False — gsm_symbolic,
        gsm_noop, dclm_200m_val, paloma_macro), AND
      - the cell is `—` or an explicit n/a marker.
    Mean rows are excluded (their values are computed, not measured).
    Category header rows are excluded.
    Any other empty / `—` cell is reported as an error so the user sees
    exactly which (model, task) pair is missing a number.
    """
    errors = []
    header_idx, table_end = find_section3_table(lines)
    if header_idx is None:
        errors.append("§3 header line not found")
        return errors
    expected_pipe_count = lines[header_idx].count("|")
    for i in range(header_idx, table_end):
        ln = lines[i]
        if not ln.startswith("|") or ln.startswith("|---|"):
            continue
        pc = ln.count("|")
        if pc != expected_pipe_count:
            errors.append(f"line {i}: pipe count {pc} != header {expected_pipe_count}: {ln[:80]}")

    # Mean row placement: each Mean row must be immediately after its category's last task row.
    for mean_label, category_label, member_tasks in MEAN_ROWS:
        mean_row_idx = None
        for i in range(header_idx + 1, table_end):
            cells = lines[i].split("|")
            if parse_row_label(cells) == mean_label:
                mean_row_idx = i
                break
        if mean_row_idx is None:
            errors.append(f"missing Mean row: '{mean_label}'")
            continue
        last_member_idx = None
        for member in member_tasks:
            for i in range(header_idx + 1, table_end):
                cells = lines[i].split("|")
                if parse_row_label(cells) == member:
                    last_member_idx = i
        if last_member_idx is not None and mean_row_idx != last_member_idx + 1:
            errors.append(
                f"Mean row '{mean_label}' is at line {mean_row_idx}, expected line {last_member_idx + 1}"
            )

    if strict:
        # Build per-task expected-blank set.
        expected_blank_labels = {t.label for t in TASKS if not t.runs_in_v2_suite}
        all_task_labels = {t.label for t in TASKS}

        # Get column headers (model labels) and their cell indices.
        header_cells = lines[header_idx].split("|")
        # Cells layout: [0]="", [1]=" Task ", [2..N-2]=model columns, [N-1]=""
        model_cols = []  # list of (cell_idx, header_label_for_display)
        for ci, c in enumerate(header_cells):
            if ci in (0, 1, len(header_cells) - 1):
                continue
            label = c.strip().replace("**", "").strip()
            # Strip trailing footnote markers like ◊ § ‖ † ª ¥ ¤
            label = re.sub(r"\s+[◊§‖†ªª¥¤]+(\s+[◊§‖†ªª¥¤]+)*\s*$", "", label).strip()
            model_cols.append((ci, label))

        # Walk every task row and check each model cell.
        mean_labels = {m[0] for m in MEAN_ROWS}
        for i in range(header_idx + 1, table_end):
            ln = lines[i]
            if not ln.startswith("|") or ln.startswith("|---|"):
                continue
            cells = ln.split("|")
            label = parse_row_label(cells)
            if not label:
                continue
            if label in mean_labels:
                continue
            if label not in all_task_labels:
                # Category header row or unknown — skip
                continue
            task_can_be_blank = label in expected_blank_labels
            for cell_idx, model_label in model_cols:
                if cell_idx >= len(cells):
                    continue
                raw = cells[cell_idx].strip()
                stripped = raw.replace("**", "").replace("*", "").replace("¶", "").strip()
                # Acceptable values:
                #   a number (parses as float)
                #   `—` IF the task is expected to be blank
                #   `n/a (ctx) ™` — explicit "not applicable" marker (e.g. phi-1 mmlu_pro context limit)
                #   `0.000` etc — counts as a number
                is_na_marker = stripped.startswith("n/a")
                is_number = False
                try:
                    float(stripped)
                    is_number = True
                except ValueError:
                    pass
                if is_number or is_na_marker:
                    continue
                # Treat `—` plus any trailing footnote marker (‡, ¶, †, ™, ‖, °, etc) as a blank.
                em_dash_only = re.sub(r"^—[\s‡¶†™‖°◊§ª¥¤‡‡‡]*$", "", stripped)
                if stripped == "" or em_dash_only == "" or stripped == "—":
                    if task_can_be_blank:
                        continue
                    errors.append(
                        f"strict: ({model_label!r}, {label!r}) at line {i} is `{stripped or 'empty'}` — should have a value"
                    )
                else:
                    errors.append(
                        f"strict: ({model_label!r}, {label!r}) at line {i} is unparseable: {raw[:40]!r}"
                    )
    return errors


def find_col_idx_in_header(header_line: str, col_label_substr: str) -> int | None:
    """Returns 0-indexed cell index (in header_line.split('|')) for the column whose header contains col_label_substr."""
    cells = header_line.split("|")
    for i, c in enumerate(cells):
        if col_label_substr in c:
            return i
    return None


def recompute_means(lines):
    """Recompute all 5 Mean rows in place. Returns count of updated."""
    header_idx, table_end = find_section3_table(lines)
    # Build row map
    row_map = {}
    for i in range(header_idx + 1, table_end):
        cells = lines[i].split("|")
        label = parse_row_label(cells)
        if label:
            row_map[label] = (i, cells)
    n_cells = len(lines[header_idx].split("|"))  # e.g. 20
    n_numeric = n_cells - 3  # subtract leading empty, Task cell, trailing empty
    updated = 0
    for mean_label, _, member_tasks in MEAN_ROWS:
        if mean_label not in row_map:
            continue
        mean_idx, _ = row_map[mean_label]
        cells = lines[mean_idx].split("|")
        for col_off in range(n_numeric):
            cell_idx = col_off + 2  # cells[2..18] are numeric for 17-col table
            vals = []
            for task in member_tasks:
                if task in row_map:
                    _, t_cells = row_map[task]
                    v = extract_cell_float(t_cells[cell_idx])
                    if v is not None:
                        vals.append(v)
            if vals:
                avg = sum(vals) / len(vals)
                cells[cell_idx] = f" *{avg:.3f}* "
            else:
                cells[cell_idx] = " — "
        lines[mean_idx] = "|".join(cells)
        updated += 1
    return updated


# ====================================================================
# Subcommands
# ====================================================================
def cmd_validate(args):
    lines = MD.read_text().split("\n")
    errors = validate_table(lines, strict=args.strict)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)
    msg = "§3 table validation OK (strict: every (model, task) cell has a value)." if args.strict else "§3 table validation OK (structure)."
    print(msg)


def cmd_fill_from_results(args):
    """Fill a §3 column from a single results dir. Column is identified by header substring match."""
    results_dir = Path(args.results_dir)
    col_label_substr = args.col_label

    lines = MD.read_text().split("\n")
    header_idx, table_end = find_section3_table(lines)
    col_idx = find_col_idx_in_header(lines[header_idx], col_label_substr)
    if col_idx is None:
        print(f"ERROR: column with label substring '{col_label_substr}' not found in header")
        sys.exit(1)
    print(f"target column index in cells split: {col_idx}")

    # Collect scores
    scores = {}
    missing = []
    for task in TASKS:
        if not task.runs_in_v2_suite:
            continue
        v = find_v2_score(results_dir, task)
        if v is not None:
            scores[task.label] = v
        else:
            missing.append(task.label)
    print(f"extracted {len(scores)} scores, {len(missing)} missing")
    if missing:
        print("missing:", missing)

    # Update cells
    updated = 0
    for i in range(header_idx + 1, table_end):
        cells = lines[i].split("|")
        label = parse_row_label(cells)
        if label in scores:
            cells[col_idx] = f" {fmt(scores[label])} "
            lines[i] = "|".join(cells)
            updated += 1
    print(f"updated {updated} cells")

    # Recompute Means
    n_means = recompute_means(lines)
    print(f"recomputed {n_means} Mean rows")

    # Validate
    errors = validate_table(lines)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    MD.write_text("\n".join(lines))
    print("Done. §3 validated.")


def cmd_fill_cell(args):
    """Fill a single (row, column) cell with a value. Validates after."""
    lines = MD.read_text().split("\n")
    header_idx, table_end = find_section3_table(lines)
    if header_idx is None:
        print("ERROR: §3 header not found")
        sys.exit(1)
    col_idx = find_col_idx_in_header(lines[header_idx], args.col)
    if col_idx is None:
        print(f"ERROR: column with substring '{args.col}' not found")
        sys.exit(1)
    row_idx = None
    for i in range(header_idx + 1, table_end):
        cells = lines[i].split("|")
        if parse_row_label(cells) == args.row:
            row_idx = i
            break
    if row_idx is None:
        print(f"ERROR: row labeled '{args.row}' not found")
        sys.exit(1)

    try:
        v = float(args.value)
    except ValueError:
        print(f"ERROR: value '{args.value}' is not a float")
        sys.exit(1)

    cells = lines[row_idx].split("|")
    cells[col_idx] = f" {fmt(v)} "
    lines[row_idx] = "|".join(cells)

    # Recompute Means in case the change affects any aggregate
    recompute_means(lines)

    # Validate
    errors = validate_table(lines)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    MD.write_text("\n".join(lines))
    print(f"updated ({args.row!r}, {args.col!r}) = {fmt(v)}")


def cmd_run(args):
    """End-to-end: run v2 suite on a node, then fill §3."""
    label = args.label
    src = args.src
    node = args.node
    hf_dst = CKPT_DIR / f"{label}_hf"

    # Smoke check node is free
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", node,
         "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc"],
        capture_output=True, text=True, timeout=10,
    )
    try:
        mem = int(out.stdout.strip())
    except Exception:
        print(f"ERROR: couldn't read GPU mem on {node}: {out.stderr}")
        sys.exit(1)
    if mem > 1000:
        print(f"ERROR: {node} has {mem} MiB GPU mem in use — not safe to launch eval")
        sys.exit(1)
    print(f"{node} is free (mem={mem} MiB)")

    # Launch eval
    ts = subprocess.check_output(["date", "+%Y%m%d_%H%M%S"], env={"TZ": "America/Los_Angeles"}).decode().strip()
    log_path = REPO / "logs" / f"v2_{label}_{ts}.log"
    print(f"launching eval; log: {log_path}")
    proc = subprocess.Popen(
        ["nohup", "bash", str(EVAL_SCRIPT),
         "--label", label, "--src", str(src), "--hf-dst", str(hf_dst), "--node", node],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"launched (pid {proc.pid}). Wait for 'ALL DONE' in {log_path}, then run:")
    print(f"  python {Path(__file__).name} fill-from-results <RESULTS_DIR> <COL_LABEL>")


def pick_free_nodes(n: int, pool: list[str]) -> list[str]:
    """Return up to n free nodes from the pool (GPU mem < 1 GiB total)."""
    free = []
    for node in pool:
        try:
            out = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", node,
                 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc"],
                capture_output=True, text=True, timeout=15,
            )
            mem = int(out.stdout.strip())
            if mem < 1000:
                free.append(node)
                if len(free) >= n:
                    return free
        except Exception:
            continue
    return free


def extract_dclm_from_train_log(log_path: Path) -> float | None:
    """Grep a Levanter training stdout log for the last `eval/dclm_200m_val/loss` value (nats)."""
    if not log_path.exists():
        return None
    last = None
    # Match: 'eval/dclm_200m_val/loss': 3.992  OR  dclm_200m_val/loss=3.992
    pat = re.compile(r"dclm_200m_val[^=:]*[=:'\"]+\s*([\d.]+)")
    with open(log_path, errors="ignore") as f:
        for line in f:
            for m in pat.finditer(line):
                try:
                    last = float(m.group(1))
                except ValueError:
                    pass
    return last


def extract_paloma_macro(paloma_root: Path) -> float | None:
    """Compute paloma macro bpb (mean across the 16 subsets) from a paloma results root.

    Each subset's results.json has `results.paloma_<subset>.bits_per_byte,none`.
    """
    bpbs = []
    for subdir in paloma_root.iterdir():
        if not subdir.is_dir() or not subdir.name.startswith("paloma_"):
            continue
        for jpath in subdir.glob("**/*results*.json"):
            try:
                d = json.load(open(jpath))
            except Exception:
                continue
            for task, metrics in d.get("results", {}).items():
                if not task.startswith("paloma_"):
                    continue
                v = metrics.get("bits_per_byte,none")
                if isinstance(v, float):
                    bpbs.append(v)
                    break
    if not bpbs:
        return None
    return sum(bpbs) / len(bpbs)


def extract_gsm_scores(gsm_root: Path) -> dict[str, float]:
    """Return {'gsm_symbolic_main': 0.012, 'gsm_noop': 0.000} from a gsm runner results root."""
    out = {}
    for jpath in gsm_root.glob("**/*results*.json"):
        try:
            d = json.load(open(jpath))
        except Exception:
            continue
        for task, metrics in d.get("results", {}).items():
            if task not in ("gsm_symbolic_main", "gsm_noop"):
                continue
            for k in ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none"):
                if k in metrics and isinstance(metrics[k], float):
                    out[task] = metrics[k]
                    break
    return out


def add_column_to_section3(lines, col_label_md: str, before_substr: str) -> int:
    """Insert a new column into the §3 table. Returns the new column's cell index.

    Updates header, underline row, and every data row (placeholder ` — `).
    Idempotent: if a column whose header contains the label already exists, returns its index without inserting.
    """
    header_idx, table_end = find_section3_table(lines)
    if header_idx is None:
        raise ValueError("§3 table not found")

    header = lines[header_idx]
    header_cells = header.split("|")
    label_core = col_label_md.replace("**", "").strip()
    for i, c in enumerate(header_cells):
        if label_core and label_core in c:
            return i  # already present
    insert_at = None
    for i, c in enumerate(header_cells):
        if before_substr in c:
            insert_at = i
            break
    if insert_at is None:
        raise ValueError(f"Anchor column substring {before_substr!r} not found in §3 header")

    header_cells.insert(insert_at, f" {col_label_md} ")
    lines[header_idx] = "|".join(header_cells)

    for i in range(header_idx + 1, table_end):
        ln = lines[i]
        row_cells = ln.split("|")
        if ln.startswith("|---|") or set(c.strip() for c in row_cells if c.strip()).issubset({"---"}):
            row_cells.insert(insert_at, "---")
        else:
            row_label = parse_row_label(row_cells)
            # Category-header rows (e.g. **Open-book**, **Math (standard)**) have empty cells
            # in every column — match that. Task rows get a real placeholder for later filling.
            if row_label and is_category_header_label(row_label):
                row_cells.insert(insert_at, " ")
            else:
                row_cells.insert(insert_at, " — ")
        lines[i] = "|".join(row_cells)

    return insert_at


def add_row_to_section2(lines, label: str, hf_dir_name: str, marker: str = "",
                         params: str = "TBD", tokens: str = "TBD", flops: str = "TBD",
                         unique: str = "TBD", notes: str = "TBD") -> None:
    """Insert a model row in §2 right before the phi-1 row. Idempotent (no-op if label exists)."""
    for ln in lines:
        if ln.startswith("|") and f"**{label}**" in ln and hf_dir_name in ln:
            return
    insert_at = None
    for i, ln in enumerate(lines):
        if ln.startswith("| phi-1 |"):
            insert_at = i
            break
    if insert_at is None:
        raise ValueError("phi-1 §2 row not found as insertion anchor")
    marker_str = f" {marker}" if marker else ""
    row = (f"| **{label}**{marker_str} | `{hf_dir_name}` | {params} | {tokens} | "
           f"{flops} | {unique} | {notes} |")
    lines.insert(insert_at, row)


def cmd_add_model(args):
    """End-to-end: add a model column to §3 + fill every fillable cell.

    Flow:
      1. Resolve HF dir (auto-convert from Levanter if needed via convert_and_eval_v2.sh).
      2. Insert column into §3 + row into §2.
      3. Pick 3 free nodes; launch v2-suite + paloma + gsm in parallel (skip suites via flags).
      4. If --train-log given, extract dclm_200m_val and fill that cell.
      5. Block on log completion; fill v2 cells, paloma_macro, gsm cells.
      6. Recompute Means + strict-validate.
    """
    label = args.label
    src = Path(args.src)
    hf_dir = src if src.name.endswith("_hf") else CKPT_DIR / f"{label}_hf"

    # 1. Resolve nodes
    pool = args.node_pool.split(",") if args.node_pool else DEFAULT_NODE_POOL
    n_wanted = (0 if args.no_v2 else 1) + (0 if args.no_paloma else 1) + (0 if args.no_gsm else 1)
    free = pick_free_nodes(n_wanted, pool)
    if len(free) < n_wanted:
        print(f"ERROR: need {n_wanted} free nodes, found {len(free)}: {free}")
        sys.exit(1)
    print(f"using nodes: {free}")
    nodes_iter = iter(free)
    v2_node = next(nodes_iter) if not args.no_v2 else None
    paloma_node = next(nodes_iter) if not args.no_paloma else None
    gsm_node = next(nodes_iter) if not args.no_gsm else None
    print(f"v2_node={v2_node} paloma_node={paloma_node} gsm_node={gsm_node}")

    # 2. Add column + §2 row (idempotent)
    lines = MD.read_text().split("\n")
    header_idx, _ = find_section3_table(lines)
    col_core_label = label
    if find_col_idx_in_header(lines[header_idx], col_core_label) is None:
        col_md = f"**{label}**" + (f" {args.footnote_marker}" if args.footnote_marker else "")
        col_idx = add_column_to_section3(lines, col_md, before_substr=args.insert_before)
        add_row_to_section2(
            lines, label=label, hf_dir_name=hf_dir.name, marker=args.footnote_marker or "",
            params=args.params, tokens=args.tokens, flops=args.flops,
            unique=args.unique, notes=args.notes,
        )
        errs = validate_table(lines)
        if errs:
            print("ERROR: validate failed after column insert:")
            for e in errs: print(f"  {e}")
            sys.exit(1)
        MD.write_text("\n".join(lines))
        print(f"inserted column at cell index {col_idx}, added §2 row")
    else:
        print(f"column {col_core_label!r} already present; will only fill cells")

    # 3. Launch sub-evals in parallel
    ts = subprocess.check_output(["date", "+%Y%m%d_%H%M%S"], env={"TZ": "America/Los_Angeles"}).decode().strip()
    launched = {}  # name -> (log_path, expected_done_marker, results_root)

    if v2_node:
        v2_log = LOG_DIR / f"v2_{label}_{ts}.log"
        cmd = ["nohup", "bash", str(EVAL_SCRIPT),
               "--label", label, "--src", str(src), "--hf-dst", str(hf_dir), "--node", v2_node]
        subprocess.Popen(cmd, stdout=open(v2_log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        launched["v2"] = (v2_log, "ALL DONE", None)
        print(f"v2-suite launched on {v2_node}; log {v2_log}")

    if paloma_node:
        p_log = LOG_DIR / f"paloma_{label}_{ts}.log"
        cmd = ["nohup", "ssh", paloma_node,
               f"bash {RUN_PALOMA} {label} {hf_dir}"]
        subprocess.Popen(cmd, stdout=open(p_log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        launched["paloma"] = (p_log, "paloma ALL DONE", None)
        print(f"paloma launched on {paloma_node}; log {p_log}")

    if gsm_node:
        g_log = LOG_DIR / f"gsm_{label}_{ts}.log"
        cmd = ["nohup", "ssh", gsm_node,
               f"bash {RUN_GSM} {label} {hf_dir}"]
        subprocess.Popen(cmd, stdout=open(g_log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        launched["gsm"] = (g_log, "gsm ALL DONE", None)
        print(f"gsm launched on {gsm_node}; log {g_log}")

    # 4. dclm_200m_val from training log (instant, do now)
    if args.train_log:
        v = extract_dclm_from_train_log(Path(args.train_log))
        if v is not None:
            cmd_fill_cell(argparse.Namespace(row="dclm_200m_val (nats)", col=col_core_label, value=str(v)))
            print(f"dclm_200m_val filled: {v:.3f} nats")
        else:
            print(f"WARN: could not find dclm_200m_val in {args.train_log}")

    if args.background:
        print("--background: launched all; cells will not be filled. Re-run with `fill-from-results` after completion.")
        return

    # 5. Block on each log until done marker (poll every 30 s).
    print("waiting for completions...")
    import time
    pending = set(launched.keys())
    while pending:
        for name in list(pending):
            log, marker, _ = launched[name]
            if not log.exists():
                continue
            try:
                if marker in log.read_text(errors="ignore"):
                    print(f"  {name} done")
                    pending.remove(name)
            except Exception:
                pass
        if pending:
            time.sleep(30)

    # 6. Fill cells
    if "v2" in launched:
        v2_results_pattern = EVAL_DIR / f"v2_{label}_*"
        candidates = sorted(EVAL_DIR.glob(f"v2_{label}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            cmd_fill_from_results(argparse.Namespace(results_dir=str(candidates[0]), col_label=col_core_label))

    if "paloma" in launched:
        p_candidates = sorted(EVAL_DIR.glob(f"paloma_{label}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if p_candidates:
            macro = extract_paloma_macro(p_candidates[0])
            if macro is not None:
                cmd_fill_cell(argparse.Namespace(row="paloma_macro (bpb)", col=col_core_label, value=str(macro)))
                print(f"paloma_macro filled: {macro:.3f} bpb")

    if "gsm" in launched:
        g_candidates = sorted(EVAL_DIR.glob(f"gsm_{label}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if g_candidates:
            scores = extract_gsm_scores(g_candidates[0])
            if "gsm_symbolic_main" in scores:
                cmd_fill_cell(argparse.Namespace(row="gsm_symbolic_main[8]", col=col_core_label, value=str(scores["gsm_symbolic_main"])))
            if "gsm_noop" in scores:
                cmd_fill_cell(argparse.Namespace(row="gsm_noop[8]", col=col_core_label, value=str(scores["gsm_noop"])))

    # 7. Final strict validate
    lines = MD.read_text().split("\n")
    errs = validate_table(lines, strict=True)
    if errs:
        print("strict-validate flagged remaining blanks:")
        for e in errs: print(f"  {e}")
    else:
        print("add-model: complete. strict-validate clean.")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="Validate §3 table structure")
    pv.add_argument("--strict", action="store_true",
                    help="Also assert that every (model, task) cell has a real value (excluding tasks that documented-blanks like gsm_symbolic, dclm_200m_val, paloma_macro).")
    pv.set_defaults(func=cmd_validate)

    pf = sub.add_parser("fill-from-results", help="Fill a §3 column from an existing v2-suite results dir")
    pf.add_argument("results_dir")
    pf.add_argument("col_label", help="substring that uniquely identifies the column header (e.g. 'C5-v3-small final')")
    pf.set_defaults(func=cmd_fill_from_results)

    pc = sub.add_parser("fill-cell", help="Fill a single (row, column) cell with a value (e.g. dclm_200m_val from a training log)")
    pc.add_argument("--row", required=True, help="row label (e.g. 'dclm_200m_val (nats)')")
    pc.add_argument("--col", required=True, help="column header substring (e.g. 'C5-v3 final')")
    pc.add_argument("--value", required=True, help="numeric value")
    pc.set_defaults(func=cmd_fill_cell)

    pr = sub.add_parser("run", help="Run v2 suite for a single model")
    pr.add_argument("label")
    pr.add_argument("src", help="Levanter checkpoint dir OR existing HF dir")
    pr.add_argument("--node", default="gpu-st-p4d24xlarge-2")
    pr.set_defaults(func=cmd_run)

    pa = sub.add_parser("add-model", help="End-to-end: insert column + launch v2/paloma/gsm + fill cells + strict-validate. One command for a new model.")
    pa.add_argument("--label", required=True, help="Column label / cell tag (e.g. 'c5v3_p2_a6_step14671')")
    pa.add_argument("--src", required=True, help="Levanter ckpt dir (will auto-convert) OR existing HF dir (path ends in _hf)")
    pa.add_argument("--train-log", help="Levanter training stdout log to grep for dclm_200m_val/loss")
    pa.add_argument("--footnote-marker", default="", help="Footnote marker for the column (e.g. '◊' or '◊§')")
    pa.add_argument("--insert-before", default="4B final", help="Substring of an existing §3 column; new column inserted to its LEFT")
    pa.add_argument("--node-pool", help="Comma-separated node names to consider; default = DEFAULT_NODE_POOL")
    pa.add_argument("--no-v2", action="store_true", help="Skip v2-suite (e.g. already filled)")
    pa.add_argument("--no-paloma", action="store_true", help="Skip paloma per-subset run")
    pa.add_argument("--no-gsm", action="store_true", help="Skip gsm_symbolic_main + gsm_noop run")
    pa.add_argument("--background", action="store_true", help="Launch all evals then exit (don't wait for completion)")
    # §2 row metadata (passed through to the inserted row in §2)
    pa.add_argument("--params", default="TBD")
    pa.add_argument("--tokens", default="TBD")
    pa.add_argument("--flops", default="TBD")
    pa.add_argument("--unique", default="TBD")
    pa.add_argument("--notes", default="TBD")
    pa.set_defaults(func=cmd_add_model)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
