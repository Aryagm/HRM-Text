# HRM-Text on Apple MLX

This adds an Apple Silicon inference path for HRM-Text. It does not replace the
CUDA/H100 training stack; upstream training still depends on PyTorch FSDP2 and
FlashAttention 3. The MLX path is intended for local generation from an exported
checkpoint.

## Install

```bash
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip install -r requirements-mlx.txt
```

## Convert

First export a trained HRM-Text checkpoint to the repository's HF-style format:

```bash
python -m conversion.convert_to_hf \
  --ckpt_path checkpoints/... \
  --out_dir exports/hrm-text-hf
```

Then convert/cast that export for MLX:

```bash
python -m conversion.convert_to_mlx \
  --hf-dir exports/hrm-text-hf \
  --out-dir exports/hrm-text-mlx \
  --dtype bfloat16
```

`bfloat16` is the default because the PyTorch inference path also uses BF16.
Use `float16` if it benchmarks better on your machine.

## Generate

```bash
python -m mlx_hrm_text.generate \
  --model-dir exports/hrm-text-mlx \
  --prompt "Explain why recurrent latent reasoning can improve sample efficiency." \
  --max-tokens 128 \
  --temperature 0
```

## Benchmark

```bash
python -m mlx_hrm_text.benchmark \
  --model-dir exports/hrm-text-mlx \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --dtype bfloat16
```

The benchmark reports prefill tokens/sec and autoregressive decode tokens/sec.

## Persist 4-bit Weights

```bash
python -m conversion.quantize_mlx \
  --model-dir exports/hrm-text-1b-hf \
  --out-dir exports/hrm-text-1b-mlx-4bit-g128 \
  --bits 4 \
  --group-size 128 \
  --mode affine
```

The generated directory can be used directly with `mlx_hrm_text.generate` and
`mlx_hrm_text.benchmark`; the loader reads `quantization.json` and builds the
quantized module layout before loading weights.

On an Apple M4 Max, the persisted 4-bit group-128 checkpoint is about 745 MB
and reaches roughly 56 decode tokens/sec with:

```bash
python -m mlx_hrm_text.benchmark \
  --model-dir exports/hrm-text-1b-mlx-4bit-g128 \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --dtype bfloat16 \
  --metal-swiglu
```

`--mode mxfp4 --group-size 32` is similar in speed and size.

## Implementation Notes

- Attention uses `mx.fast.scaled_dot_product_attention`, so the core prefill and
  decode attention work runs through MLX's optimized Metal path.
- Decode keeps per-recurrence KV caches for HRM's H and L recurrent modules,
  matching the original `create_cache` structure.
- PrefixLM exports use unmasked prompt attention during prefill and reuse the
  cache for one-token decode.
- The current CLI is single-prompt, single-batch inference. Batched scheduling
  similar to `simple_inference_engine.py` can be layered on top of the same
  cache structure.
