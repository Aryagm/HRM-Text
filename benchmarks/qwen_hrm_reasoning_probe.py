from __future__ import annotations

import argparse
from dataclasses import dataclass
import time

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_hrm_text.qwen_hrm import QwenHrmForCausalLM


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    options: dict[str, str]
    answer: str


CASES = [
    Case(
        "algebra",
        "Solve 7(x - 3) + 2x = 5x + 29.",
        {"A": "10", "B": "12.5", "C": "25", "D": "50"},
        "B",
    ),
    Case(
        "markup",
        "A jacket is marked up by 25%, then discounted by 20%. Final price is $96. What was the original price?",
        {"A": "$76.80", "B": "$80", "C": "$96", "D": "$120"},
        "C",
    ),
    Case(
        "boxes",
        "Three boxes are labeled Apples, Oranges, and Mixed. All labels are wrong. You may draw one fruit from one box. Which labeled box do you draw from?",
        {"A": "Apples", "B": "Oranges", "C": "Mixed", "D": "any box"},
        "C",
    ),
    Case(
        "inclusion",
        "How many integers from 1 to 200 inclusive are divisible by 3 or 5 but not by 15?",
        {"A": "67", "B": "80", "C": "93", "D": "106"},
        "B",
    ),
    Case(
        "modular",
        "What is the remainder when 7^103 is divided by 13?",
        {"A": "1", "B": "6", "C": "8", "D": "11"},
        "B",
    ),
    Case(
        "work_rate",
        "Alice can finish a job in 6 hours and Bob can finish it in 10 hours. Working together, how many hours do they need?",
        {"A": "3.75", "B": "4", "C": "8", "D": "16"},
        "A",
    ),
    Case(
        "probability",
        "Two fair dice are rolled. What is the probability that the sum is 9?",
        {"A": "1/12", "B": "1/9", "C": "1/6", "D": "1/4"},
        "B",
    ),
    Case(
        "sequence",
        "A sequence has a_1 = 2 and a_n = 3a_{n-1} + 1. What is a_4?",
        {"A": "20", "B": "40", "C": "67", "D": "80"},
        "C",
    ),
    Case(
        "linear_word",
        "A number is doubled and then increased by 9. The result is 31. What is the number?",
        {"A": "10", "B": "11", "C": "20", "D": "22"},
        "B",
    ),
    Case(
        "reverse_discount",
        "After a 10% discount, a book costs $72. What was the original price?",
        {"A": "$64.80", "B": "$79.20", "C": "$80", "D": "$90"},
        "C",
    ),
    Case(
        "multiples",
        "How many positive integers less than 100 are divisible by 4 but not by 8?",
        {"A": "10", "B": "12", "C": "24", "D": "25"},
        "B",
    ),
    Case(
        "rectangle",
        "A rectangle has perimeter 50 and length 15. What is its width?",
        {"A": "5", "B": "10", "C": "20", "D": "35"},
        "B",
    ),
    Case(
        "average",
        "The average of five numbers is 12. Four of the numbers are 10, 11, 13, and 14. What is the fifth number?",
        {"A": "10", "B": "11", "C": "12", "D": "14"},
        "C",
    ),
    Case(
        "combinations",
        "How many ways are there to choose 2 items from 6 distinct items?",
        {"A": "12", "B": "15", "C": "30", "D": "36"},
        "B",
    ),
    Case(
        "coins",
        "Three fair coins are flipped. What is the probability of getting exactly two heads?",
        {"A": "1/8", "B": "1/4", "C": "3/8", "D": "1/2"},
        "C",
    ),
    Case(
        "modular_small",
        "What is the remainder when 5^4 is divided by 7?",
        {"A": "1", "B": "2", "C": "3", "D": "4"},
        "B",
    ),
    Case(
        "syllogism",
        "All bloops are razzies. All razzies are lazzies. Must all bloops be lazzies?",
        {"A": "yes", "B": "no", "C": "only some", "D": "cannot tell"},
        "A",
    ),
]


def log_softmax_row(row: mx.array) -> mx.array:
    row = row.astype(mx.float32)
    return row - mx.logsumexp(row, axis=-1)


def build_prompt(case: Case) -> str:
    options = "\n".join(f"{key}. {value}" for key, value in case.options.items())
    return f"{case.question}\nOptions:\n{options}\nAnswer:"


def score_candidate(model: QwenHrmForCausalLM, tokenizer, prompt_ids: list[int], candidate: str) -> float:
    candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + candidate_ids
    logits = model(mx.array(input_ids, dtype=mx.int32)[None, :])[0]

    score = 0.0
    start = len(prompt_ids) - 1
    for idx, token_id in enumerate(candidate_ids):
        log_probs = log_softmax_row(logits[start + idx])
        mx.eval(log_probs)
        score += float(log_probs[token_id].item())
    return score


def evaluate_mode(
    model: QwenHrmForCausalLM,
    tokenizer,
    *,
    h_cycles: int,
    l_cycles: int,
    logit_blend: float,
    alpha_l: float | None,
    alpha_h: float | None,
    beta_l: float | None,
    beta_h: float | None,
    update_mix_l: float | None,
    update_mix_h: float | None,
    refined_delta_scale: float | None,
    split_index: int | None,
) -> tuple[int, list[tuple[Case, str, dict[str, float]]]]:
    model.model.H_cycles = h_cycles
    model.model.L_cycles = l_cycles
    model.logit_blend = logit_blend
    if split_index is not None:
        model.model.config.split_index = split_index
    if alpha_l is not None:
        model.model.alpha_l = alpha_l
    if alpha_h is not None:
        model.model.alpha_h = alpha_h
    if beta_l is not None:
        model.model.beta_l = beta_l
    if beta_h is not None:
        model.model.beta_h = beta_h
    if update_mix_l is not None:
        model.model.update_mix_l = update_mix_l
    if update_mix_h is not None:
        model.model.update_mix_h = update_mix_h
    if refined_delta_scale is not None:
        model.model.refined_delta_scale = refined_delta_scale

    correct = 0
    rows = []
    for case in CASES:
        prompt_ids = tokenizer(build_prompt(case), add_special_tokens=False)["input_ids"]
        scores = {key: score_candidate(model, tokenizer, prompt_ids, f" {key}") for key in case.options}
        prediction = max(scores, key=scores.get)
        correct += int(prediction == case.answer)
        rows.append((case, prediction, scores))
    return correct, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiple-choice logit probe for the Qwen3 HRM conversion.")
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--logit-blend", type=float, default=0.2)
    parser.add_argument("--alpha-l", type=float, default=None)
    parser.add_argument("--alpha-h", type=float, default=None)
    parser.add_argument("--beta-l", type=float, default=None)
    parser.add_argument("--beta-h", type=float, default=None)
    parser.add_argument("--update-mix-l", type=float, default=None)
    parser.add_argument("--update-mix-h", type=float, default=None)
    parser.add_argument("--refined-delta-scale", type=float, default=None)
    parser.add_argument("--split-index", type=int, default=None, help="Number of lower layers assigned to L.")
    parser.add_argument("--include-base", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = QwenHrmForCausalLM.from_pretrained(args.model)
    mx.eval(model.parameters())

    modes = []
    if args.include_base:
        modes.append(("base", 0, 1, 0.0))
    modes.append(("hrm", args.h_cycles, args.l_cycles, args.logit_blend))

    for label, h_cycles, l_cycles, blend in modes:
        start = time.perf_counter()
        correct, rows = evaluate_mode(
            model,
            tokenizer,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            logit_blend=blend,
            alpha_l=args.alpha_l,
            alpha_h=args.alpha_h,
            beta_l=args.beta_l,
            beta_h=args.beta_h,
            update_mix_l=args.update_mix_l,
            update_mix_h=args.update_mix_h,
            refined_delta_scale=args.refined_delta_scale,
            split_index=args.split_index,
        )
        print(f"== {label} ==")
        for case, prediction, scores in rows:
            compact_scores = " ".join(f"{key}:{scores[key]:.2f}" for key in sorted(scores))
            marker = "ok" if prediction == case.answer else "miss"
            print(f"{case.name:12s} expected={case.answer} pred={prediction} {marker} scores[{compact_scores}]")
        elapsed = time.perf_counter() - start
        print(f"accuracy={correct}/{len(CASES)} elapsed={elapsed:.2f}s\n")


if __name__ == "__main__":
    main()
