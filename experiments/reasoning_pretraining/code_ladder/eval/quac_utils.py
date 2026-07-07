"""QUAC task utilities. First-turn only: 1 prompt per conversation = 1000 evaluations.

This avoids the multi-turn conversation-history complication and gives a tractable
single-shot QA evaluation that's roughly comparable to Aryabumi's "Acc." on QUAC.
"""

import transformers.data.metrics.squad_metrics as squad_metrics


def doc_to_text(doc):
    """First question only. Background + section title + context, then Q: A:"""
    bg = (doc.get("background") or "").strip()
    sec = (doc.get("section_title") or "").strip()
    ctx = (doc.get("context") or "").strip()
    q0 = doc["questions"][0]
    parts = []
    if bg:
        parts.append(bg)
    if sec:
        parts.append(sec)
    parts.append(ctx)
    return "\n\n".join(parts) + f"\n\nQ: {q0}\n\nA:"


def doc_to_target(doc):
    """Gold answer(s) for question 0. Dedup and keep up to all 4 annotators."""
    if not doc["answers"]["texts"] or not doc["answers"]["texts"][0]:
        return ["CANNOTANSWER"]
    answers = []
    for a in doc["answers"]["texts"][0]:
        if a and a.lower() not in (x.lower() for x in answers):
            answers.append(a)
    return answers or ["CANNOTANSWER"]


def em(gold_list, pred):
    return max(squad_metrics.compute_exact(a, pred) for a in gold_list)


def f1(gold_list, pred):
    return max(squad_metrics.compute_f1(a, pred) for a in gold_list)


def process_results(doc, results):
    gold_list = doc_to_target(doc)
    pred = results[0].strip()
    # Strip trailing "Q:" tail in case the model started a follow-up.
    if "\nQ:" in pred:
        pred = pred.split("\nQ:")[0].strip()
    return {
        "em": em(gold_list, pred),
        "f1": f1(gold_list, pred),
    }
