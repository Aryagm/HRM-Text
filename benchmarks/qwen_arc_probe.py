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


def load_arc_cases(arc_config: str, split: str, limit: int | None) -> list[ArcCase]:
    dataset = load_dataset("allenai/ai2_arc", arc_config, split=split)
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


def build_calibration_prompt(case: ArcCase, calibration: str) -> str | None:
    if calibration == "none":
        return None
    options = "\n".join(f"{label}. {text}" for label, text in case.choices)
    if calibration == "answer_prior":
        return "Answer:"
    if calibration == "options_prior":
        return f"Options:\n{options}\nAnswer:"
    raise ValueError(f"Unsupported calibration {calibration!r}")


def candidate_continuation(label: str, text: str, answer_scoring: str) -> str:
    if answer_scoring == "label":
        return label
    if answer_scoring == "choice_text":
        return text
    if answer_scoring == "label_text":
        return f"{label}. {text}"
    raise ValueError(f"Unsupported answer_scoring {answer_scoring!r}")


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


def configure_model(
    model: QwenHrmForCausalLM,
    mode: str,
    logit_fusion: str,
    logit_blend: float,
    h_cycles: int,
    l_cycles: int,
    alpha_l: float,
    alpha_h: float,
    beta_l: float,
    beta_h: float,
    split_index: int | None,
) -> None:
    if mode == "base":
        model.model.H_cycles = 0
        model.model.L_cycles = 1
        model.logit_blend = 0.0
        model.logit_fusion = "blend"
        return
    model.model.H_cycles = h_cycles
    model.model.L_cycles = l_cycles
    model.model.alpha_l = alpha_l
    model.model.alpha_h = alpha_h
    model.model.beta_l = beta_l
    model.model.beta_h = beta_h
    model.model.update_mix_l = 1.0
    model.model.update_mix_h = 1.0
    model.model.refined_delta_scale = 1.0
    if split_index is not None:
        model.model.config.split_index = split_index
    model.logit_fusion = logit_fusion
    model.logit_blend = logit_blend
    model.fusion_threshold = 0.0


def evaluate(
    model: QwenHrmForCausalLM,
    tokenizer,
    cases: list[ArcCase],
    *,
    answer_scoring: str,
    calibration: str,
    calibration_weight: float,
) -> tuple[int, list[dict[str, object]]]:
    rows = []
    correct = 0
    for case in cases:
        prompt_ids = tokenizer(build_prompt(case), add_special_tokens=False)["input_ids"]
        calibration_prompt = build_calibration_prompt(case, calibration)
        calibration_ids = (
            tokenizer(calibration_prompt, add_special_tokens=False)["input_ids"] if calibration_prompt is not None else None
        )
        scores = {}
        calibration_scores = {}
        raw_scores = {}
        for label, text in case.choices:
            continuation = candidate_continuation(label, text, answer_scoring)
            raw_score = score_candidate(model, tokenizer, prompt_ids, continuation)
            calibration_score = (
                score_candidate(model, tokenizer, calibration_ids, continuation) if calibration_ids is not None else 0.0
            )
            raw_scores[label] = raw_score
            calibration_scores[label] = calibration_score
            scores[label] = raw_score - calibration_weight * calibration_score
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
            row[f"raw_score_{label}"] = raw_scores[label]
            row[f"calibration_score_{label}"] = calibration_scores[label]
        rows.append(row)
    return correct, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="AI2 ARC logprob probe for Qwen-HRM MLX checkpoints.")
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--arc-config", choices=("ARC-Challenge", "ARC-Easy"), default="ARC-Challenge")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--mode", choices=("base", "hrm"), default="hrm")
    parser.add_argument(
        "--answer-scoring",
        choices=("label", "choice_text", "label_text"),
        default="label",
        help="Candidate continuation to score after the ARC prompt.",
    )
    parser.add_argument(
        "--calibration",
        choices=("none", "answer_prior", "options_prior"),
        default="none",
        help="Subtract a no-training prior score from each candidate.",
    )
    parser.add_argument("--calibration-weight", type=float, default=1.0)
    parser.add_argument(
        "--logit-fusion",
        choices=("blend", "delta", "prob_blend", "confidence_gate", "agreement_blend"),
        default="blend",
    )
    parser.add_argument("--logit-blend", type=float, default=0.2)
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--alpha-l", type=float, default=0.03)
    parser.add_argument("--alpha-h", type=float, default=0.05)
    parser.add_argument("--beta-l", type=float, default=0.01)
    parser.add_argument("--beta-h", type=float, default=0.01)
    parser.add_argument("--split-index", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    effective_calibration_weight = 0.0 if args.calibration == "none" else args.calibration_weight

    cases = load_arc_cases(args.arc_config, args.split, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = QwenHrmForCausalLM.from_pretrained(args.model)
    configure_model(
        model,
        args.mode,
        args.logit_fusion,
        args.logit_blend,
        args.h_cycles,
        args.l_cycles,
        args.alpha_l,
        args.alpha_h,
        args.beta_l,
        args.beta_h,
        args.split_index,
    )
    mx.eval(model.parameters())

    start = time.perf_counter()
    correct, rows = evaluate(
        model,
        tokenizer,
        cases,
        answer_scoring=args.answer_scoring,
        calibration=args.calibration,
        calibration_weight=effective_calibration_weight,
    )
    elapsed = time.perf_counter() - start
    total = len(rows)

    fields = [
        "timestamp_utc",
        "model",
        "arc_config",
        "split",
        "limit",
        "mode",
        "answer_scoring",
        "calibration",
        "calibration_weight",
        "logit_fusion",
        "logit_blend",
        "h_cycles",
        "l_cycles",
        "alpha_l",
        "alpha_h",
        "beta_l",
        "beta_h",
        "split_index",
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
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("score_") or key.startswith("raw_score_") or key.startswith("calibration_score_")
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra_fields, lineterminator="\n")
        writer.writeheader()
        timestamp = datetime.now(timezone.utc).isoformat()
        for row in rows:
            writer.writerow(
                {
                    **{field: "NA" for field in extra_fields},
                    "timestamp_utc": timestamp,
                    "model": args.model,
                    "arc_config": args.arc_config,
                    "split": args.split,
                    "limit": args.limit,
                    "mode": args.mode,
                    "answer_scoring": args.answer_scoring,
                    "calibration": args.calibration,
                    "calibration_weight": effective_calibration_weight,
                    "logit_fusion": model.logit_fusion,
                    "logit_blend": model.logit_blend,
                    "h_cycles": model.model.H_cycles,
                    "l_cycles": model.model.L_cycles,
                    "alpha_l": model.model.alpha_l,
                    "alpha_h": model.model.alpha_h,
                    "beta_l": model.model.beta_l,
                    "beta_h": model.model.beta_h,
                    "split_index": model.model.config.split_index,
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / max(1, total),
                    "elapsed_s": elapsed,
                    **row,
                }
            )
    print(
        f"{args.model} {args.mode} {args.arc_config} {args.split}[:{args.limit}] "
        f"{args.answer_scoring} calibration={args.calibration}: {correct}/{total}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
