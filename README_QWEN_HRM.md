# Qwen3 HRM Conversion

This branch adds an MLX-native, no-training HRM conversion path for Qwen3-style
decoder checkpoints. It is designed to load existing MLX-community quantized
checkpoints directly, split the pretrained layer stack into lower `L` and upper
`H` modules, and run gated recurrent latent refinement before projecting logits.

Recommended starting checkpoint:

```bash
hrm-mlx-qwen-hrm \
  --model mlx-community/Qwen3-1.7B-4bit \
  --chat-template \
  --prompt "Solve: if 3x + 7 = 31, what is x? Think carefully." \
  --max-tokens 128 \
  --temperature 0
```

Fast smoke-test checkpoint:

```bash
hrm-mlx-qwen-hrm \
  --model mlx-community/Qwen3-0.6B-4bit \
  --chat-template \
  --prompt "Solve: if 3x + 7 = 31, what is x? Think carefully." \
  --max-tokens 128 \
  --temperature 0
```

## Modes

Base model only:

```bash
hrm-mlx-qwen-hrm \
  --model mlx-community/Qwen3-1.7B-4bit \
  --chat-template \
  --prompt "..." \
  --h-cycles 0 \
  --logit-blend 0
```

Default conservative HRM blend:

```bash
hrm-mlx-qwen-hrm \
  --model mlx-community/Qwen3-1.7B-4bit \
  --chat-template \
  --prompt "..." \
  --h-cycles 1 \
  --l-cycles 1 \
  --alpha-l 0.03 \
  --alpha-h 0.05 \
  --beta-l 0.01 \
  --beta-h 0.01 \
  --update-mix-l 1.0 \
  --update-mix-h 1.0 \
  --refined-delta-scale 1.0 \
  --logit-blend 0.2
```

HRM logits only:

```bash
hrm-mlx-qwen-hrm \
  --model mlx-community/Qwen3-1.7B-4bit \
  --chat-template \
  --prompt "..." \
  --logit-blend 1
```

## Architecture

For a 28-layer Qwen3 checkpoint:

```text
Embedding
  -> base path: layers 0..27 -> final norm -> base logits
  -> HRM warm path:
       L = layers 0..13
       H = layers 14..27
  -> recurrent refinement:
       repeat H_cycles:
         repeat L_cycles:
           z_L = L(RMSNorm(z_L + alpha_l * z_H + beta_l * embedding))
         z_H = H(RMSNorm(z_H + alpha_h * z_L + beta_h * embedding))
  -> final norm -> refined logits
  -> blended logits = (1 - blend) * base + blend * refined
```

The wrapper preserves separate KV caches for the base pass, warm split pass, and
each recurrent `L`/`H` pass. That is heavier than a normal decode, but it is the
closest no-training version to HRM while keeping the pretrained model's original
forward path available as an anchor.

The defaults are intentionally conservative. In a no-training conversion, larger
gates or a high logit blend can move the hidden states off the pretrained
manifold quickly. Use `--logit-blend 0` as the exact base-model control, then
sweep gates/blend upward.

## Current Probe Results

The benchmark script:

```bash
python benchmarks/qwen_hrm_reasoning_probe.py \
  --model mlx-community/Qwen3-0.6B-4bit \
  --include-base
```

Corrected 17-question multiple-choice probe:

| Model | Base | HRM default | Delta |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B-4bit | 5 / 17 | 7 / 17 | +2 |
| Qwen3-1.7B-4bit | 10 / 17 | 10 / 17 | 0 |

The current gain is real but narrow: the no-training recurrence helps the smaller
0.6B checkpoint on this probe, while the stronger 1.7B checkpoint only changes
logit margins and latency. Treat this as an experimental conversion baseline,
not a broad HRM-level result yet.

Additional stability knobs:

- `--update-mix-l` and `--update-mix-h` under-relax recurrent state updates.
  Values below `1.0` keep part of the previous `z_L`/`z_H` state instead of
  fully replacing it with the next recurrent block output.
- `--refined-delta-scale` scales the final hidden delta from the base model to
  the refined state before logits are projected.
