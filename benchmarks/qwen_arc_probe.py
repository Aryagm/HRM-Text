from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

import mlx.core as mx
from datasets import load_dataset
from transformers import AutoTokenizer

from mlx_hrm_text.qwen_hrm import QwenHrmForCausalLM


@dataclass(frozen=True)
class ArcCase:
    idx: int
    question: str
    choices: tuple[tuple[str, str], ...]
    answer: str


def normalize_label(label: str, idx: int) -> str:
    label = str(label).strip()
    if label.isdigit():
        return chr(ord("A") + int(label) - 1)
    if label:
        return label.upper()
    return chr(ord("A") + idx)


def load_arc_cases(split: str, limit: int | None) -> list[ArcCase]:
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    cases = []
    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        labels = [normalize_label(label, pos) for pos, label in enumerate(row["choices"]["label"])]
        texts = [str(text) for text in row["choices"]["text"]]
        answer = normalize_label(row["answerKey"], 0)
        if answer not in labels:
            continue
        cases.append(ArcCase(idx=idx, question=str(row["question"]), choices=tuple(zip(labels, texts)), answer=answer))
    return cases


def build_prompt(case: ArcCase) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in case.choices)
    return f"{case.question}\nOptions:\n{options}\nAnswer:"


def log_softmax_row(logits: mx.array) -> mx.array:
    logits = logits.astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def score_candidate(model: QwenHrmForCausalLM, tokenizer, prompt_ids: list[int], candidate: str) -> float:
    candidate_ids = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + candidate_ids
    logits = model(mx.array(input_ids, dtype=mx.int32)[None, :])[0]
    score = 0.0
    start = len(prompt_ids) - 1
    for idx, token_id in enumerate(candidate_ids):
        log_probs = log_softmax_row(logits[start + idx])
        mx.eval(log_probs)
        score += float(log_probs[token_id].item())
    return score / max(1, len(candidate_ids))


def configure_model(model: QwenHrmForCausalLM, mode: str, logit_fusion: str, logit_blend: float) -> None:
    if mode == "base":
        model.model.H_cycles = 0
        model.model.L_cycles = 1
        model.logit_blend = 0.0
        model.logit_fusion = "blend"
        return
    model.model.H_cycles = 1
    model.model.L_cycles = 1
    model.model.alpha_l = 0.03
    model.model.alpha_h = 0.05
    model.model.beta_l = 0.01
    model.model.beta_h = 0.01
    model.model.update_mix_l = 1.0
    model.model.update_mix_h = 1.0
    model.model.refined_delta_scale = 1.0
    model.logit_fusion = logit_fusion
    model.logit_blend = logit_blend
    model.fusion_threshold = 0.0


def evaluate(model: QwenHrmForCausalLM, tokenizer, cases: list[ArcCase]) -> tuple[int, list[dict[str, object]]]:
    rows = []
    correct = 0
    for case in cases:
        prompt_ids = tokenizer(build_prompt(case), add_special_tokens=False)["input_ids"]
        scores = {label: score_candidate(model, tokenizer, prompt_ids, label) for label, _ in case.choices}
        prediction = max(scores.items(), key=lambda item: item[1])[0]
        is_correct = prediction == case.answer
        correct += int(is_correct)
        row = {
            "case_index": case.idx,
            "question": case.question,
            "answer": case.answer,
            "prediction": prediction,
            "is_correct": is_correct,
        }
        for label, score in scores.items():
            row[f"score_{label}"] = score
        rows.append(row)
    return correct, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-Challenge logprob probe for Qwen-HRM MLX checkpoints.")
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--mode", choices=("base", "hrm"), default="hrm")
    parser.add_argument(
        "--logit-fusion",
        choices=("blend", "delta", "prob_blend", "confidence_gate", "agreement_blend"),
        default="blend",
    )
    parser.add_argument("--logit-blend", type=float, default=0.2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = load_arc_cases(args.split, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = QwenHrmForCausalLM.from_pretrained(args.model)
    configure_model(model, args.mode, args.logit_fusion, args.logit_blend)
    mx.eval(model.parameters())

    start = time.perf_counter()
    correct, rows = evaluate(model, tokenizer, cases)
    elapsed = time.perf_counter() - start
    total = len(rows)

    fields = [
        "timestamp_utc",
        "model",
        "split",
        "limit",
        "mode",
        "logit_fusion",
        "logit_blend",
        "correct",
        "total",
        "accuracy",
        "elapsed_s",
        "case_index",
        "question",
        "answer",
        "prediction",
        "is_correct",
    ]
    extra_fields = sorted({key for row in rows for key in row if key.startswith("score_")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra_fields)
        writer.writeheader()
        timestamp = datetime.now(timezone.utc).isoformat()
        for row in rows:
            writer.writerow(
                {
                    "timestamp_utc": timestamp,
                    "model": args.model,
                    "split": args.split,
                    "limit": args.limit,
                    "mode": args.mode,
                    "logit_fusion": model.logit_fusion,
                    "logit_blend": model.logit_blend,
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / max(1, total),
                    "elapsed_s": elapsed,
                    **row,
                }
            )
    print(f"{args.model} {args.mode} ARC-Challenge {args.split}[:{args.limit}]: {correct}/{total}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
