from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import re
from pathlib import Path
import time


@dataclass(frozen=True)
class ExactCase:
    name: str
    prompt: str
    answers: tuple[str, ...]


CASES = [
    ExactCase("algebra", "Solve 7(x - 3) + 2x = 5x + 29.", ("12.5", "25/2")),
    ExactCase("markup", "A jacket is marked up by 25%, then discounted by 20%. Final price is $96. What was the original price?", ("96", "$96")),
    ExactCase("boxes", "Three boxes are labeled Apples, Oranges, and Mixed. All labels are wrong. You may draw one fruit from one box. Which labeled box do you draw from?", ("mixed", "Mixed")),
    ExactCase("inclusion", "How many integers from 1 to 200 inclusive are divisible by 3 or 5 but not by 15?", ("80",)),
    ExactCase("modular", "What is the remainder when 7^103 is divided by 13?", ("6",)),
    ExactCase("work_rate", "Alice can finish a job in 6 hours and Bob can finish it in 10 hours. Working together, how many hours do they need?", ("3.75", "15/4")),
    ExactCase("probability", "Two fair dice are rolled. What is the probability that the sum is 9?", ("1/9",)),
    ExactCase("sequence", "A sequence has a_1 = 2 and a_n = 3a_{n-1} + 1. What is a_4?", ("67",)),
    ExactCase("linear_word", "A number is doubled and then increased by 9. The result is 31. What is the number?", ("11",)),
    ExactCase("reverse_discount", "After a 10% discount, a book costs $72. What was the original price?", ("80", "$80")),
    ExactCase("multiples", "How many positive integers less than 100 are divisible by 4 but not by 8?", ("12",)),
    ExactCase("rectangle", "A rectangle has perimeter 50 and length 15. What is its width?", ("10",)),
    ExactCase("average", "The average of five numbers is 12. Four of the numbers are 10, 11, 13, and 14. What is the fifth number?", ("12",)),
    ExactCase("combinations", "How many ways are there to choose 2 items from 6 distinct items?", ("15",)),
    ExactCase("coins", "Three fair coins are flipped. What is the probability of getting exactly two heads?", ("3/8",)),
    ExactCase("modular_small", "What is the remainder when 5^4 is divided by 7?", ("2",)),
    ExactCase("syllogism", "All bloops are razzies. All razzies are lazzies. Must all bloops be lazzies?", ("yes",)),
]


FINAL_PATTERNS = [
    re.compile(r"<answer>\s*([^<]+?)\s*</answer>", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*([^<\n]+?)\s*</answer>", re.IGNORECASE | re.DOTALL),
    re.compile(r"final answer\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"boxed\s*(?:braces)?\s*[:：]?\s*\{([^{}]+)\}", re.IGNORECASE),
]


def extract_latex_box(text: str) -> str | None:
    for marker in ("\\boxed", "\\fbox"):
        idx = text.rfind(marker)
        if idx < 0:
            continue
        left = text.find("{", idx)
        if left < 0:
            continue
        depth = 0
        for pos in range(left, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[left + 1 : pos].strip()
    return None


def simplify_latex(text: str) -> str:
    text = text.strip()
    frac_pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while True:
        updated = frac_pattern.sub(r"\1/\2", text)
        if updated == text:
            break
        text = updated
    text = text.replace("\\(", " ").replace("\\)", " ")
    text = text.replace("\\[", " ").replace("\\]", " ")
    return text


def normalize(text: str) -> str:
    text = simplify_latex(text)
    text = text.strip().lower()
    text = re.sub(r"^(final\s+)?answer\s*[:：]\s*", "", text)
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("\\", "")
    text = text.replace("boxed", "")
    text = re.sub(r"[\[\]{}()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(".;:")
    return text


def first_number(text: str) -> str | None:
    match = re.search(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?", text)
    if match:
        return match.group(0).replace(" ", "")
    return None


def numeric_equal(a: str, b: str) -> bool:
    try:
        return Fraction(a) == Fraction(b)
    except Exception:
        try:
            return abs(float(a) - float(b)) < 1e-9
        except Exception:
            return False


def extract_answer(text: str) -> str:
    boxed = extract_latex_box(text)
    if boxed is not None:
        return boxed

    candidates = []
    for pattern in FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            if pattern.pattern.startswith("<answer>") and len({normalize(match) for match in matches}) > 1:
                return "AMBIGUOUS: " + " | ".join(match.strip() for match in matches[:5])
            candidates.append(matches[-1])
    if candidates:
        answer = candidates[-1]
    else:
        stripped = text.strip()
        if not stripped:
            answer = ""
        else:
            lines = [line.strip() for line in stripped.splitlines() if line.strip()]
            answer = lines[-1] if lines else stripped
    answer = answer.split("\n")[0]
    answer = answer.split("<|")[0]
    return answer.strip()


def is_correct(raw_output: str, expected: tuple[str, ...]) -> tuple[bool, str]:
    extracted = extract_answer(raw_output)
    norm = normalize(extracted)
    expected_norms = [normalize(item) for item in expected]
    if norm in expected_norms:
        return True, extracted

    number = first_number(norm)
    if number is not None:
        for exp in expected_norms:
            exp_num = first_number(exp)
            if exp_num is not None and numeric_equal(number, exp_num):
                return True, extracted
    for exp in expected_norms:
        if first_number(exp) is None and re.search(rf"\b{re.escape(exp)}\b", norm):
            return True, extracted
    return False, extracted


def qwen_prompt(prompt: str, prompt_style: str) -> str:
    if prompt_style == "boxed":
        return f"{prompt}\nReason step by step, and put your final answer within \\boxed{{}}."
    if prompt_style == "final":
        return f"{prompt}\nReason briefly. End with exactly one line in this format: Final answer: your answer"
    raise ValueError(f"Unsupported prompt_style {prompt_style!r}. Use final or boxed.")


def apply_qwen_chat_template(tokenizer, prompt: str, *, enabled: bool, enable_thinking: bool) -> str:
    if not enabled:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def hrm_prompt(prompt: str) -> str:
    inner = f"{prompt} Please reason step by step, and put your final answer within \\boxed{{}}."
    return f"<|im_start|><|quad_end|><|object_ref_end|>{inner}<|im_end|>"


def configure_qwen(
    model: QwenHrmForCausalLM,
    mode: str,
    logit_blend: float | None = None,
    logit_fusion: str | None = None,
    fusion_threshold: float | None = None,
) -> None:
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
    model.logit_blend = 0.2 if logit_blend is None else logit_blend
    model.logit_fusion = "blend" if logit_fusion is None else logit_fusion
    model.fusion_threshold = 0.0 if fusion_threshold is None else fusion_threshold


def run_qwen(
    model_id: str,
    mode: str,
    max_tokens: int,
    prompt_style: str,
    chat_template: bool,
    enable_thinking: bool,
    logit_blend: float | None,
    logit_fusion: str | None,
    fusion_threshold: float | None,
) -> list[dict[str, str | int | float | bool]]:
    import mlx.core as mx
    from transformers import AutoTokenizer

    from mlx_hrm_text.generate_qwen_hrm import generate as generate_qwen
    from mlx_hrm_text.qwen_hrm import QwenHrmForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = QwenHrmForCausalLM.from_pretrained(model_id)
    configure_qwen(model, mode, logit_blend=logit_blend, logit_fusion=logit_fusion, fusion_threshold=fusion_threshold)
    mx.eval(model.parameters())

    rows = []
    for case in CASES:
        prompt = apply_qwen_chat_template(
            tokenizer,
            qwen_prompt(case.prompt, prompt_style),
            enabled=chat_template,
            enable_thinking=enable_thinking,
        )
        start = time.perf_counter()
        output = generate_qwen(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - start
        correct, extracted = is_correct(output, case.answers)
        rows.append(
            {
                "case": case.name,
                "expected": "|".join(case.answers),
                "extracted": extracted,
                "is_correct": correct,
                "elapsed_s": elapsed,
                "prompt_style": prompt_style,
                "chat_template": chat_template,
                "enable_thinking": enable_thinking,
                "max_tokens": max_tokens,
                "logit_blend": model.logit_blend,
                "logit_fusion": model.logit_fusion,
                "fusion_threshold": model.fusion_threshold,
                "output": output,
            }
        )
    return rows


def run_hrm_text(model_dir: str, max_tokens: int, dtype: str) -> list[dict[str, str | int | float | bool]]:
    import mlx.core as mx
    from transformers import AutoTokenizer

    from mlx_hrm_text.generate import generate as generate_hrm_text
    from mlx_hrm_text.model import HrmTextForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    model = HrmTextForCausalLM.from_pretrained(model_dir, dtype=dtype)
    mx.eval(model.parameters())

    rows = []
    for case in CASES:
        start = time.perf_counter()
        output = generate_hrm_text(
            model,
            tokenizer,
            hrm_prompt(case.prompt),
            max_tokens=max_tokens,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - start
        correct, extracted = is_correct(output, case.answers)
        rows.append(
            {
                "case": case.name,
                "expected": "|".join(case.answers),
                "extracted": extracted,
                "is_correct": correct,
                "elapsed_s": elapsed,
                "prompt_style": "boxed",
                "chat_template": False,
                "enable_thinking": False,
                "max_tokens": max_tokens,
                "logit_blend": "",
                "logit_fusion": "",
                "fusion_threshold": "",
                "output": output,
            }
        )
    return rows


def write_results(out: Path, model: str, mode: str, rows: list[dict[str, str | int | float | bool]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    correct = sum(1 for row in rows if row["is_correct"])
    total = len(rows)
    fields = [
        "timestamp_utc",
        "model",
        "mode",
        "correct",
        "total",
        "accuracy",
        "case",
        "expected",
        "extracted",
        "is_correct",
        "elapsed_s",
        "prompt_style",
        "chat_template",
        "enable_thinking",
        "max_tokens",
        "logit_blend",
        "logit_fusion",
        "fusion_threshold",
        "output",
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        timestamp = datetime.now(timezone.utc).isoformat()
        for row in rows:
            writer.writerow(
                {
                    "timestamp_utc": timestamp,
                    "model": model,
                    "mode": mode,
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total,
                    **row,
                }
            )
    print(f"{model} {mode}: {correct}/{total}")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact-answer generative probe for Qwen-HRM and HRM-Text.")
    parser.add_argument("--engine", choices=("qwen", "hrm-text"), required=True)
    parser.add_argument("--model", required=True, help="Qwen HF/MLX repo id or HRM-Text local model dir.")
    parser.add_argument("--mode", choices=("base", "hrm"), default="hrm")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--qwen-prompt-style", choices=("final", "boxed"), default="final")
    parser.add_argument("--qwen-chat-template", action="store_true")
    parser.add_argument("--qwen-enable-thinking", action="store_true")
    parser.add_argument("--qwen-logit-blend", type=float, default=None)
    parser.add_argument(
        "--qwen-logit-fusion",
        choices=("blend", "delta", "prob_blend", "confidence_gate", "agreement_blend"),
        default=None,
    )
    parser.add_argument("--qwen-fusion-threshold", type=float, default=None)
    args = parser.parse_args()

    if args.engine == "qwen":
        rows = run_qwen(
            args.model,
            args.mode,
            args.max_tokens,
            args.qwen_prompt_style,
            args.qwen_chat_template,
            args.qwen_enable_thinking,
            args.qwen_logit_blend,
            args.qwen_logit_fusion,
            args.qwen_fusion_threshold,
        )
        write_results(args.out, args.model, args.mode, rows)
    else:
        rows = run_hrm_text(args.model, args.max_tokens, args.dtype)
        write_results(args.out, args.model, "hrm-text", rows)


if __name__ == "__main__":
    main()
