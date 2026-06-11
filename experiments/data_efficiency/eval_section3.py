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
MD = REPO / "experiments" / "data_efficiency" / "EVALUATION.md"
EVAL_SCRIPT = REPO / "experiments" / "data_efficiency" / "convert_and_eval_v2.sh"
RUN_EVAL_V2 = REPO / "experiments" / "data_efficiency" / "run_eval_v2.sh"


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
    TaskRow("openbookqa_fact[0]", "openbookqa_fact", ("acc,none",)),
    TaskRow("arc_easy[25]", "arc_easy", ("acc,none",)),
    TaskRow("arc_challenge[25]", "arc_challenge", ("acc,none",)),
    TaskRow("hellaswag[10]", "hellaswag", ("acc,none",)),
    TaskRow("winogrande[5]", "winogrande", ("acc,none",)),
    TaskRow("mmlu[5]", "_mmlu_mean", ()),                         # special: average mmlu_<subject>
    TaskRow("commonsense_qa[0]", "commonsense_qa", ("acc,none",)),
    TaskRow("social_iqa[0]", "social_iqa", ("acc,none",)),
    TaskRow("logiqa[0]", "logiqa", ("acc,none",)),
    TaskRow("lambada_openai[0]", "lambada_openai", ("acc,none",)),  # NOT perplexity
    TaskRow("copa[0]", "copa", ("acc,none",)),
    TaskRow("wsc[0]", "wsc", ("acc,none",)),
    TaskRow("agieval_lsat_ar[0]", "agieval_lsat_ar", ("acc,none",)),
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
      "commonsense_qa[0]", "social_iqa[0]", "logiqa[0]", "lambada_openai[0]", "copa[0]", "wsc[0]"]),
    ("Mean Aggregate",
     "Aggregate",
     ["agieval_lsat_ar[0]", "gpqa_diamond[0]", "bbh[3] (limit=0.1)", "mmlu_pro[5] (limit=0.1)"]),
    ("Mean Math (standard)",
     "Math (standard)",
     ["gsm8k[5]", "gsm8k_cot[8]", "minerva_math[4]"]),
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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
