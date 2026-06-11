#!/usr/bin/env python3
"""Fill §3 cells for all 15 internal models (gsm_symbolic_main + gsm_noop)
and 4 C5-v3 columns (paloma_macro) once their respective runs complete.

Idempotent: only writes cells where a result file exists; logs how many
filled vs how many still pending.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_section3 import (
    extract_gsm_scores, extract_paloma_macro, find_col_idx_in_header,
    find_section3_table, parse_row_label, fmt, recompute_means, validate_table, MD,
)

EVAL_DIR = Path("/fsx/users/dongweij/marin/outputs/eval_results")

# Map gsm <runner LABEL> → §3 column substring used in header
# Note: ' (x16)' suffixes can clash; pick uniquely-identifying substrings.
GSM_TO_COL_SUBSTR = {
    "base": "base (x16)",
    "code25_v2": "code25 v2 (x16)",
    "c5v2_small_stage1": "C5-v2 small stage-1",
    "c5v2_small_final": "C5-v2 small ‖§",
    "a5_final": "A5 1ep final",
    "b4_final": "B4 1ep final",
    "c5_stage1": "C5 stage-1",
    "c5v2_stage1": "C5-v2 stage-1",
    "c5_final": "C5 final †",
    "c5v2_final": "C5-v2 final ‖",
    "c5v3_phase1": "C5-v3 phase 1",
    "c5v3_final": "C5-v3 final",
    "c5v3_small_phase1": "C5-v3-small phase 1",
    "c5v3_small_final": "C5-v3-small final",
    "4b_final": "4B final",
}

PALOMA_TO_COL_SUBSTR = {
    "c5v3_phase1_step14671": "C5-v3 phase 1",
    "c5v3_p2_a6_step14671": "C5-v3 final",
    "c5v3_small_phase1_step6399": "C5-v3-small phase 1",
    "c5v3_small_phase2_step6399": "C5-v3-small final",
}


def fill_cell_in_lines(lines, row_label, col_substr, value):
    header_idx, table_end = find_section3_table(lines)
    col_idx = find_col_idx_in_header(lines[header_idx], col_substr)
    if col_idx is None:
        return f"col '{col_substr}' not found"
    for i in range(header_idx + 1, table_end):
        cells = lines[i].split("|")
        if parse_row_label(cells) == row_label:
            cells[col_idx] = f" {fmt(value)} "
            lines[i] = "|".join(cells)
            return None
    return f"row '{row_label}' not found"


def main():
    lines = MD.read_text().split("\n")
    filled = 0
    pending = []
    errors = []

    # GSM (30 cells = 15 models × 2 tasks)
    for label, col_substr in GSM_TO_COL_SUBSTR.items():
        results_dirs = sorted(EVAL_DIR.glob(f"gsm_{label}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not results_dirs:
            pending.append(f"gsm_{label}: no results dir")
            continue
        scores = extract_gsm_scores(results_dirs[0])
        if "gsm_symbolic_main" in scores:
            err = fill_cell_in_lines(lines, "gsm_symbolic_main[8]", col_substr, scores["gsm_symbolic_main"])
            if err: errors.append(f"gsm_{label} symbolic_main: {err}")
            else: filled += 1
        else:
            pending.append(f"gsm_{label}: symbolic_main missing")
        if "gsm_noop" in scores:
            err = fill_cell_in_lines(lines, "gsm_noop[8]", col_substr, scores["gsm_noop"])
            if err: errors.append(f"gsm_{label} noop: {err}")
            else: filled += 1
        else:
            pending.append(f"gsm_{label}: noop missing")

    # PALOMA (4 cells)
    for label, col_substr in PALOMA_TO_COL_SUBSTR.items():
        results_dirs = sorted(EVAL_DIR.glob(f"paloma_{label}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not results_dirs:
            pending.append(f"paloma_{label}: no results dir")
            continue
        macro = extract_paloma_macro(results_dirs[0])
        if macro is None:
            pending.append(f"paloma_{label}: bpb not extractable yet")
            continue
        err = fill_cell_in_lines(lines, "paloma_macro (bpb)", col_substr, macro)
        if err: errors.append(f"paloma_{label}: {err}")
        else: filled += 1

    print(f"filled: {filled} cells")
    print(f"pending: {len(pending)}")
    for p in pending: print(f"  - {p}")
    if errors:
        print("ERRORS:")
        for e in errors: print(f"  - {e}")

    if filled > 0:
        recompute_means(lines)
        errs = validate_table(lines)
        if errs:
            print("VALIDATE FAILED (NOT WRITING):")
            for e in errs: print(f"  {e}")
            sys.exit(1)
        MD.write_text("\n".join(lines))
        print(f"§3 updated. {filled} cells filled, Means recomputed, structure validated.")

    # Final strict check
    print("--- strict-validate ---")
    errs = validate_table(MD.read_text().split("\n"), strict=True)
    if errs:
        print(f"{len(errs)} cells still blank:")
        for e in errs[:20]: print(f"  {e}")
        if len(errs) > 20:
            print(f"  ... and {len(errs)-20} more")
    else:
        print("§3 strict-validate: clean. Every fillable cell has a value.")


if __name__ == "__main__":
    main()
