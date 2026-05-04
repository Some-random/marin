# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Run per-example eval on benchmarks, showing all options and scores.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/data_efficiency/generate_benchmark_examples.py \
    --checkpoint konwoo/300m4k-209Mx16-dclm-cos-lr0.0030-wd1.60-seed0 \
    --name "Paper 300M" \
    --output outputs/eval_results/paper_300m_examples.json \
    --num_examples 10
"""

import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def get_choice_logprobs(model, tokenizer, prompt, choices, device):
    scores = []
    for choice in choices:
        full = prompt + " " + choice
        inputs = tokenizer(full, return_tensors="pt").to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        prompt_len = len(prompt_ids)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        log_probs = F.log_softmax(logits, dim=-1)
        token_ids = inputs["input_ids"][0]
        score = sum(log_probs[j - 1, token_ids[j]].item() for j in range(prompt_len, len(token_ids)))
        scores.append(score)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", default="model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_examples", type=int, default=10)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    print(f"Loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB" if device != "cpu" else "CPU mode")

    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    piqa_ds = load_dataset("piqa", split="test", trust_remote_code=True)
    sciq_ds = load_dataset("allenai/sciq", split="test")

    results = {"model": args.name, "arc_easy": [], "piqa": [], "sciq": []}
    N = args.num_examples

    print(f"\n{'='*60}")
    print(f"ARC Easy ({N} examples)")
    print(f"{'='*60}")
    correct = 0
    for i in range(N):
        doc = arc[i]
        prompt = f"Question: {doc['question']}\nAnswer:"
        choices = doc["choices"]["text"]
        labels = doc["choices"]["label"]
        correct_idx = labels.index(doc["answerKey"])
        scores = get_choice_logprobs(model, tokenizer, prompt, choices, device)
        pred_idx = scores.index(max(scores))
        is_correct = pred_idx == correct_idx
        correct += is_correct

        print(f"\nQ: {doc['question']}")
        for j, (label, choice, score) in enumerate(zip(labels, choices, scores)):
            marker = "→" if j == pred_idx else " "
            star = "✓" if j == correct_idx else " "
            print(f"  {marker}{star} {label}. {choice}  [{score:.2f}]")
        print(f"  Result: {'✓' if is_correct else '✗'}")

        results["arc_easy"].append({
            "question": doc["question"],
            "options": {l: c for l, c in zip(labels, choices)},
            "scores": {l: s for l, s in zip(labels, scores)},
            "predicted": choices[pred_idx],
            "correct": choices[correct_idx],
            "is_correct": is_correct,
        })
    print(f"\nARC Easy: {correct}/{N}")

    print(f"\n{'='*60}")
    print(f"PIQA ({N} examples)")
    print(f"{'='*60}")
    correct = 0
    for i in range(N):
        doc = piqa_ds[i]
        prompt = f"Goal: {doc['goal']}\nSolution:"
        choices = [doc["sol1"], doc["sol2"]]
        correct_idx = doc["label"]
        scores = get_choice_logprobs(model, tokenizer, prompt, choices, device)
        pred_idx = scores.index(max(scores))
        is_correct = pred_idx == correct_idx
        correct += is_correct

        print(f"\nGoal: {doc['goal']}")
        for j, (choice, score) in enumerate(zip(choices, scores)):
            marker = "→" if j == pred_idx else " "
            star = "✓" if j == correct_idx else " "
            print(f"  {marker}{star} {j+1}. {choice[:150]}  [{score:.2f}]")
        print(f"  Result: {'✓' if is_correct else '✗'}")

        results["piqa"].append({
            "goal": doc["goal"],
            "options": {"1": doc["sol1"], "2": doc["sol2"]},
            "scores": {"1": scores[0], "2": scores[1]},
            "predicted": choices[pred_idx],
            "correct": choices[correct_idx],
            "is_correct": is_correct,
        })
    print(f"\nPIQA: {correct}/{N}")

    print(f"\n{'='*60}")
    print(f"SciQ ({N} examples)")
    print(f"{'='*60}")
    correct = 0
    for i in range(N):
        doc = sciq_ds[i]
        prompt = f"Question: {doc['question']}\nAnswer:"
        choices = [doc["correct_answer"], doc["distractor1"], doc["distractor2"], doc["distractor3"]]
        labels = ["A", "B", "C", "D"]
        scores = get_choice_logprobs(model, tokenizer, prompt, choices, device)
        pred_idx = scores.index(max(scores))
        is_correct = pred_idx == 0
        correct += is_correct

        print(f"\nQ: {doc['question']}")
        for j, (label, choice, score) in enumerate(zip(labels, choices, scores)):
            marker = "→" if j == pred_idx else " "
            star = "✓" if j == 0 else " "
            print(f"  {marker}{star} {label}. {choice}  [{score:.2f}]")
        print(f"  Result: {'✓' if is_correct else '✗'}")

        results["sciq"].append({
            "question": doc["question"],
            "options": {l: c for l, c in zip(labels, choices)},
            "scores": {l: s for l, s in zip(labels, scores)},
            "predicted": choices[pred_idx],
            "correct": choices[0],
            "is_correct": is_correct,
        })
    print(f"\nSciQ: {correct}/{N}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
