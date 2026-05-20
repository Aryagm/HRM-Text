from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

import mlx.core as mx
from transformers import AutoTokenizer

from benchmarks.exact_answer_probe import CASES, extract_answer, first_number, is_correct, normalize
from mlx_hrm_text.qwen_hrm import QwenHrmForCausalLM


@dataclass(frozen=True)
class Candidate:
    case: str
    answer: str
    source_file: str
    source_mode: str
    source_correct: str


def log_softmax_row(logits: mx.array) -> mx.array:
    logits = logits.astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def score_continuation(model: QwenHrmForCausalLM, tokenizer, prompt: str, continuation: str) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + continuation_ids
    logits = model(mx.array(input_ids, dtype=mx.int32)[None, :])[0]

    score = 0.0
    start = len(prompt_ids) - 1
    for idx, token_id in enumerate(continuation_ids):
        log_probs = log_softmax_row(logits[start + idx])
        mx.eval(log_probs)
        score += float(log_probs[token_id].item())
    return score / max(1, len(continuation_ids))


def configure_model(model: QwenHrmForCausalLM, verifier_mode: str, logit_fusion: str, logit_blend: float) -> None:
    if verifier_mode == "base":
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


def verifier_prompt(tokenizer, problem: str, answer: str, *, chat_template: bool) -> str:
    prompt = (
        f"Problem:\n{problem}\n\n"
        f"Proposed answer:\n{answer}\n\n"
        "Is the proposed answer correct? Answer only yes or no.\nAnswer:"
    )
    if not chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def answer_prompt(tokenizer, problem: str, *, chat_template: bool) -> str:
    prompt = f"Problem:\n{problem}\n\nFinal answer:"
    if not chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def all_numbers(text: str) -> list[str]:
    return [match.replace(" ", "") for match in re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?", text)]


def candidate_variants(output: str) -> list[str]:
    extracted = extract_answer(output)
    variants = [extracted]
    normalized = normalize(extracted)
    if normalized and normalized != extracted:
        variants.append(normalized)
    for number in all_numbers(normalized):
        variants.append(number)
    for number in all_numbers(normalize(output))[-6:]:
        variants.append(number)
    deduped = []
    seen = set()
    for item in variants:
        clean = item.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean[:240])
    return deduped


def read_candidates(files: list[Path]) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = {}
    seen: set[tuple[str, str]] = set()
    for file in files:
        with file.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for answer in candidate_variants(row["output"]):
                    key = (row["case"], normalize(answer))
                    if key in seen:
                        continue
                    seen.add(key)
                    grouped.setdefault(row["case"], []).append(
                        Candidate(
                            case=row["case"],
                            answer=answer,
                            source_file=file.name,
                            source_mode=row.get("mode", ""),
                            source_correct=row.get("is_correct", ""),
                        )
                    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank saved Qwen exact-answer candidates with yes/no verifier logits.")
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--candidate-csv", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verifier-mode", choices=("base", "hrm"), default="base")
    parser.add_argument(
        "--logit-fusion",
        choices=("blend", "delta", "prob_blend", "confidence_gate", "agreement_blend"),
        default="agreement_blend",
    )
    parser.add_argument("--logit-blend", type=float, default=0.2)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--scoring", choices=("yesno", "answer_likelihood"), default="yesno")
    args = parser.parse_args()

    cases = {case.name: case for case in CASES}
    candidates = read_candidates(args.candidate_csv)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = QwenHrmForCausalLM.from_pretrained(args.model)
    configure_model(model, args.verifier_mode, args.logit_fusion, args.logit_blend)
    mx.eval(model.parameters())

    rows = []
    for case_name, case in cases.items():
        scored = []
        for candidate in candidates.get(case_name, []):
            if args.scoring == "yesno":
                prompt = verifier_prompt(tokenizer, case.prompt, candidate.answer, chat_template=args.chat_template)
                yes_score = score_continuation(model, tokenizer, prompt, " yes")
                no_score = score_continuation(model, tokenizer, prompt, " no")
                score = yes_score - no_score
            else:
                prompt = answer_prompt(tokenizer, case.prompt, chat_template=args.chat_template)
                yes_score = score_continuation(model, tokenizer, prompt, " " + candidate.answer)
                no_score = 0.0
                score = yes_score
            scored.append((score, yes_score, no_score, candidate))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        margin, yes_score, no_score, best = scored[0]
        correct, extracted = is_correct(best.answer, case.answers)
        rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "verifier_mode": args.verifier_mode,
                "logit_fusion": args.logit_fusion,
                "logit_blend": args.logit_blend,
                "chat_template": args.chat_template,
                "scoring": args.scoring,
                "case": case_name,
                "expected": "|".join(case.answers),
                "selected_answer": best.answer,
                "selected_extracted": extracted,
                "is_correct": correct,
                "margin": margin,
                "yes_score": yes_score,
                "no_score": no_score,
                "candidate_count": len(scored),
                "source_file": best.source_file,
                "source_mode": best.source_mode,
                "source_correct": best.source_correct,
            }
        )

    total = len(rows)
    correct = sum(row["is_correct"] for row in rows)
    fields = [
        "timestamp_utc",
        "model",
        "verifier_mode",
        "logit_fusion",
        "logit_blend",
        "chat_template",
        "scoring",
        "correct",
        "total",
        "accuracy",
        "case",
        "expected",
        "selected_answer",
        "selected_extracted",
        "is_correct",
        "margin",
        "yes_score",
        "no_score",
        "candidate_count",
        "source_file",
        "source_mode",
        "source_correct",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"correct": correct, "total": total, "accuracy": correct / total, **row})
    print(f"{args.verifier_mode} verifier: {correct}/{total}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
