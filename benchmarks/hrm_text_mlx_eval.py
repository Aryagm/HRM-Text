from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluation.benchmarks import ARC, BoolQ, DROP, GSM8k, HellaSwag, MATH, MMLU, Winogrande
from mlx_hrm_text.generate import sample_next
from mlx_hrm_text.model import HrmTextForCausalLM, set_metal_swiglu


CONDITION_TOKENS = {
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
}

DEFAULT_GENERATION = {
    "batch_size": 33,
    "max_context": 3072,
    "temperature": 0.0,
    "condition": "synth,cot",
}

BENCHMARK_CONFIGS: dict[str, dict[str, Any]] = {
    "GSM8k": {"class": GSM8k},
    "MATH": {"class": MATH},
    "DROP": {"class": DROP, "generation_config": {"condition": "direct"}},
    "MMLU": {
        "class": MMLU,
        "kwargs": {"special_shots": {"high_school_european_history": 3}},
        "generation_config": {"condition": "direct", "max_context": 4096, "batch_size": 1},
    },
    "ARC": {
        "class": ARC,
        "generation_config": {"condition": "direct", "max_context": 4096, "batch_size": 1},
    },
    "HellaSwag": {
        "class": HellaSwag,
        "generation_config": {"condition": "direct", "max_context": 4096, "batch_size": 1},
    },
    "Winogrande": {
        "class": Winogrande,
        "generation_config": {"condition": "direct", "max_context": 4096, "batch_size": 1},
    },
    "BoolQ": {
        "class": BoolQ,
        "generation_config": {"condition": "direct", "max_context": 4096, "batch_size": 1},
    },
}


@dataclass
class RunConfig:
    model_dir: Path
    dtype: str
    max_context: int
    max_tokens: int | None
    temperature: float
    condition: str
    static_cache: bool


def format_hrm_prompt(condition: str, prompt: str) -> str:
    condition_prefix = "".join(CONDITION_TOKENS[item] for item in condition.split(","))
    return f"<|im_start|>{condition_prefix}{prompt.strip()}<|im_end|>"


def encode_prompt(tokenizer, condition: str, prompt: str, max_context: int) -> list[int] | None:
    formatted = format_hrm_prompt(condition, prompt)
    encoded = tokenizer(formatted, return_tensors="np", return_attention_mask=False, add_special_tokens=False)
    token_ids = encoded["input_ids"][0].tolist()
    if len(token_ids) >= max_context:
        return None
    return token_ids


def generate_one(
    model: HrmTextForCausalLM,
    tokenizer,
    prompt: str,
    config: RunConfig,
) -> str:
    prompt_ids = encode_prompt(tokenizer, config.condition, prompt, config.max_context)
    if prompt_ids is None:
        return ""

    generation_budget = config.max_tokens
    if generation_budget is None:
        generation_budget = config.max_context - len(prompt_ids)
    generation_budget = min(generation_budget, config.max_context - len(prompt_ids))
    if generation_budget <= 0:
        return ""

    cache = model.make_cache(max_length=config.max_context if config.static_cache else None)
    input_ids = mx.array(prompt_ids, dtype=mx.int32)
    logits = model.prefill(input_ids, cache)
    next_id = sample_next(logits[0], config.temperature)

    generated: list[int] = []
    position = len(prompt_ids)
    stop_id = tokenizer.convert_tokens_to_ids("<|box_end|>")
    for _ in range(generation_budget):
        if next_id == stop_id:
            break
        generated.append(next_id)
        logits = model.decode_one(mx.array([next_id], dtype=mx.int32), position, cache)
        next_id = sample_next(logits[0], config.temperature)
        position += 1

    return tokenizer.decode(generated, skip_special_tokens=False)


def load_existing(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    existing: dict[int, str] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            existing[int(row["index"])] = row["generation"]
    return existing


def instantiate_benchmark(name: str):
    try:
        config = BENCHMARK_CONFIGS[name]
    except KeyError as exc:
        known = ", ".join(BENCHMARK_CONFIGS)
        raise ValueError(f"Unknown benchmark {name!r}. Known benchmarks: {known}") from exc

    kwargs = config.get("kwargs", {})
    benchmark = config["class"](**kwargs)
    return benchmark, config


def slice_benchmark(benchmark, start: int, limit: int | None) -> tuple[list[str], list[Any]]:
    end = None if limit is None else start + limit
    prompts = benchmark.prompts[start:end]
    ground_truths = benchmark.ground_truths[start:end]
    benchmark.prompts = prompts
    benchmark.ground_truths = ground_truths
    return prompts, ground_truths


def run_benchmark(
    model: HrmTextForCausalLM,
    tokenizer,
    name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    benchmark, benchmark_config = instantiate_benchmark(name)
    prompts, _ = slice_benchmark(benchmark, args.start, args.limit)

    gen_config = DEFAULT_GENERATION | benchmark.generation_overrides | benchmark_config.get("generation_config", {})
    max_tokens = args.max_tokens if args.max_tokens is not None else gen_config.get("max_tokens")
    run_config = RunConfig(
        model_dir=args.model_dir,
        dtype=args.dtype,
        max_context=int(args.max_context or gen_config["max_context"]),
        max_tokens=None if max_tokens is None else int(max_tokens),
        temperature=float(args.temperature if args.temperature is not None else gen_config["temperature"]),
        condition=str(args.condition or gen_config["condition"]),
        static_cache=bool(args.static_cache),
    )

    prediction_path = args.out_dir / f"{args.run_name}_{name.lower()}_predictions.jsonl"
    summary_path = args.out_dir / f"{args.run_name}_{name.lower()}_summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing(prediction_path) if args.resume else {}
    generations_by_index = dict(existing)

    started_at = time.perf_counter()
    with prediction_path.open("a") as out:
        progress = tqdm(range(len(prompts)), desc=name)
        for local_index in progress:
            if local_index in generations_by_index:
                continue

            generation = generate_one(model, tokenizer, prompts[local_index], run_config)
            generations_by_index[local_index] = generation
            out.write(json.dumps({"index": local_index, "generation": generation}, ensure_ascii=False) + "\n")
            out.flush()

    missing = [i for i in range(len(prompts)) if i not in generations_by_index]
    if missing:
        raise RuntimeError(f"{name} is missing {len(missing)} generations after run.")

    generations = [generations_by_index[i] for i in range(len(prompts))]
    metrics = benchmark.compute_metrics(generations)
    elapsed = time.perf_counter() - started_at
    summary = {
        "benchmark": name,
        "model_dir": str(args.model_dir),
        "run_name": args.run_name,
        "start": args.start,
        "limit": args.limit,
        "generation_config": {
            "condition": run_config.condition,
            "max_context": run_config.max_context,
            "max_tokens": run_config.max_tokens,
            "temperature": run_config.temperature,
            "static_cache": run_config.static_cache,
        },
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "seconds_per_example": elapsed / max(1, len(prompts)),
        "prediction_path": str(prediction_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def parse_benchmarks(values: list[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        names.extend(item.strip() for item in value.split(",") if item.strip())
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official HRM-Text benchmarks through the MLX checkpoint.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True, help="Prefix used for prediction and summary files.")
    parser.add_argument(
        "--benchmark",
        "--benchmarks",
        action="append",
        required=True,
        help="Benchmark name or comma-separated names. Supported: " + ", ".join(BENCHMARK_CONFIGS),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/hrm_text_mlx_eval"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-context", type=int, default=None, help="Override benchmark max_context.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override benchmark max_tokens.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Run only a slice for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing prediction JSONL rows.")
    parser.add_argument("--static-cache", action="store_true")
    parser.add_argument("--metal-swiglu", action="store_true")
    parser.add_argument("--h-cycles", type=int, default=None)
    parser.add_argument("--l-cycles", type=int, default=None)
    args = parser.parse_args()

    set_metal_swiglu(args.metal_swiglu)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    model = HrmTextForCausalLM.from_pretrained(args.model_dir, dtype=args.dtype)
    if args.h_cycles is not None:
        model.model.H_cycles = args.h_cycles
    if args.l_cycles is not None:
        model.model.L_cycles = args.l_cycles
    mx.eval(model.parameters())

    summaries = []
    for benchmark in parse_benchmarks(args.benchmark):
        summaries.append(run_benchmark(model, tokenizer, benchmark, args))

    if len(summaries) > 1:
        combined_path = args.out_dir / f"{args.run_name}_summary.json"
        combined_path.write_text(json.dumps({"runs": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
