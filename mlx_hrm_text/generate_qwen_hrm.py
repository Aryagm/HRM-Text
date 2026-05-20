from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_hrm_text.qwen_hrm import QwenHrmForCausalLM


def sample_next(logits: mx.array, temperature: float) -> int:
    logits = logits.astype(mx.float32)
    if temperature <= 1e-6:
        token = mx.argmax(logits, axis=-1)
    else:
        token = mx.random.categorical(logits / temperature)
    mx.eval(token)
    return int(token.item())


def format_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def generate(
    model: QwenHrmForCausalLM,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    eos_token_id: int | None,
) -> str:
    encoded = tokenizer(prompt, return_tensors="np", return_attention_mask=False, add_special_tokens=False)
    prompt_ids = encoded["input_ids"][0].tolist()
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens.")

    max_length = len(prompt_ids) + max_tokens if model.use_static_cache else None
    cache = model.make_cache(max_length=max_length)
    logits = model.prefill(mx.array(prompt_ids, dtype=mx.int32), cache)
    next_id = sample_next(logits[0], temperature)

    generated: list[int] = []
    position = len(prompt_ids)
    for _ in range(max_tokens):
        if eos_token_id is not None and next_id == eos_token_id:
            break
        generated.append(next_id)
        logits = model.decode_one(mx.array([next_id], dtype=mx.int32), position, cache)
        next_id = sample_next(logits[0], temperature)
        position += 1

    return tokenizer.decode(generated, skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a no-training Qwen3 HRM conversion on MLX.")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-1.7B-4bit",
        help="Local path or Hugging Face repo id for an MLX Qwen3 checkpoint.",
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--static-cache", action="store_true", help="Preallocate KV caches.")
    parser.add_argument("--chat-template", action="store_true", help="Use the Qwen chat template with thinking enabled.")
    parser.add_argument("--h-cycles", type=int, default=None)
    parser.add_argument("--l-cycles", type=int, default=None)
    parser.add_argument("--alpha-l", type=float, default=None)
    parser.add_argument("--alpha-h", type=float, default=None)
    parser.add_argument("--beta-l", type=float, default=None)
    parser.add_argument("--beta-h", type=float, default=None)
    parser.add_argument("--update-mix-l", type=float, default=None)
    parser.add_argument("--update-mix-h", type=float, default=None)
    parser.add_argument("--refined-delta-scale", type=float, default=None)
    parser.add_argument("--split-index", type=int, default=None, help="Number of lower layers assigned to L.")
    parser.add_argument("--logit-blend", type=float, default=None, help="0.0 is base model only, 1.0 is HRM logits only.")
    parser.add_argument(
        "--logit-fusion",
        choices=("blend", "delta", "prob_blend", "confidence_gate", "agreement_blend"),
        default=None,
    )
    parser.add_argument("--fusion-threshold", type=float, default=None)
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    tokenizer_source = args.model if Path(args.model).exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=True)
    prompt = format_prompt(tokenizer, args.prompt, args.chat_template)

    model = QwenHrmForCausalLM.from_pretrained(args.model, dtype=args.dtype, revision=args.revision)
    if args.h_cycles is not None:
        model.model.H_cycles = args.h_cycles
    if args.l_cycles is not None:
        model.model.L_cycles = args.l_cycles
    if args.alpha_l is not None:
        model.model.alpha_l = args.alpha_l
    if args.alpha_h is not None:
        model.model.alpha_h = args.alpha_h
    if args.beta_l is not None:
        model.model.beta_l = args.beta_l
    if args.beta_h is not None:
        model.model.beta_h = args.beta_h
    if args.update_mix_l is not None:
        model.model.update_mix_l = args.update_mix_l
    if args.update_mix_h is not None:
        model.model.update_mix_h = args.update_mix_h
    if args.refined_delta_scale is not None:
        model.model.refined_delta_scale = args.refined_delta_scale
    if args.split_index is not None:
        model.model.config.split_index = args.split_index
    if args.logit_blend is not None:
        model.logit_blend = args.logit_blend
    if args.logit_fusion is not None:
        model.logit_fusion = args.logit_fusion
    if args.fusion_threshold is not None:
        model.fusion_threshold = args.fusion_threshold
    model.use_static_cache = args.static_cache
    mx.eval(model.parameters())

    print(
        generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
