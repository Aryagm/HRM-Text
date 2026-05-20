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

The default split is size-aware:

- Qwen3-0.6B uses the symmetric split, `L = 14`, `H = 14`.
- Qwen3-1.7B uses `L = 10`, `H = 18`, which was better than the symmetric
  split on the corrected probe.

The wrapper preserves separate KV caches for the base pass, warm split pass, and
each recurrent `L`/`H` pass. That is heavier than a normal decode, but it is the
closest no-training version to HRM while keeping the pretrained model's original
forward path available as an anchor.

The defaults are intentionally conservative. In a no-training conversion, larger
gates or a high logit blend can move the hidden states off the pretrained
manifold quickly. Use `--logit-blend 0` as the exact base-model control, then
sweep gates/blend upward.

## Current Probe Results

The multiple-choice benchmark script:

```bash
python benchmarks/qwen_hrm_reasoning_probe.py \
  --model mlx-community/Qwen3-0.6B-4bit \
  --include-base
```

Corrected 17-question multiple-choice probe:

| Model | Base | HRM default | Delta |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B-4bit | 5 / 17 | 7 / 17 | +2 |
| Qwen3-1.7B-4bit | 10 / 17 | 11 / 17 | +1 |

The current gain is real but narrow: the no-training recurrence improves both
tracked checkpoints on this probe, with the stronger effect on 0.6B. Treat this
as an experimental conversion baseline, not a broad HRM-level result yet.

This MC set is a fast smoke/regression test, not an optimal reasoning benchmark.
It is small, hand-built, and can reward candidate-option calibration instead of
open-ended reasoning. To catch that, this branch also tracks a stricter
exact-answer generative probe:

```bash
python benchmarks/exact_answer_probe.py \
  --engine qwen \
  --model mlx-community/Qwen3-1.7B-4bit \
  --mode hrm \
  --out outputs/qwen_hrm_metrics/exact_qwen3_1_7b_hrm.csv \
  --max-tokens 128
```

Current exact-answer results:

| Model | Mode | Correct |
| --- | --- | ---: |
| Qwen3-0.6B-4bit | Base | 1 / 17 |
| Qwen3-0.6B-4bit | Qwen-HRM | 2 / 17 |
| Qwen3-1.7B-4bit | Base | 6 / 17 |
| Qwen3-1.7B-4bit | Qwen-HRM | 6 / 17 |
| HRM-Text-1B MLX 4-bit | Native HRM-Text | 16 / 17 |
| HRM-Text-1B MLX BF16 | Native HRM-Text | 16 / 17 |

The exact-answer probe is stricter than the MC probe and shows the current
conversion is far from a decisive HRM-level jump. HRM-Text must be evaluated
with its documented boxed-answer prompt and enough generation budget; the earlier
short `Final answer:` prompt undercounted both 4-bit and BF16 badly.

Follow-up Qwen3-1.7B chat-template runs with boxed final answers did not rescue
the generative exact score:

| Run | Correct |
| --- | ---: |
| Qwen3-1.7B Base, chat + boxed | 6 / 17 |
| Qwen3-1.7B Qwen-HRM, chat + boxed, blend 0.20 | 5 / 17 |
| Qwen3-1.7B Qwen-HRM, chat + boxed, blend 0.10 | 5 / 17 |
| Qwen3-1.7B Qwen-HRM, chat + boxed, blend 0.05 | 5 / 17 |
| Qwen3-1.7B Qwen-HRM, chat + boxed, agreement gate | 6 / 17 |
| Qwen3-1.7B Qwen-HRM, chat + boxed, confidence gate | 6 / 17 |
| Qwen3-1.7B Qwen-HRM, base prefix then final-answer blend | 6 / 17 |
| Qwen3-1.7B Qwen-HRM, base prefix then final-answer confidence gate | 6 / 17 |

That points to a real limitation in the current no-training fusion: fixed logit
blending can improve candidate log-probability ranking while still damaging
open-ended chat generation. Agreement/confidence gates recover the base score on
this probe, and delayed fusion recovers the base score by keeping the reasoning
prefix on base logits until a final-answer marker appears. Open-ended generation
should use base-anchored or final-answer-triggered fusion rather than blindly
blend refined logits on every generated token. The next strategy should try a
stronger verifier/reranker use of the refined path if we want gains above base.

Candidate reranking was also tested over saved Qwen3-1.7B exact outputs. The
combined raw/chat candidate pool has an oracle union of 11 / 17 correct cases,
but naive no-training rerankers did not exploit it:

| Reranker | Correct |
| --- | ---: |
| Raw base + HRM candidates, base yes/no verifier | 6 / 17 |
| Raw base + HRM candidates, HRM agreement yes/no verifier | 6 / 17 |
| Raw base + HRM candidates, base answer-likelihood ranker | 6 / 17 |
| Raw + chat/gated candidates, base answer-likelihood ranker | 6 / 17 |

So the candidate-generation side has useful diversity, but the verifier needs to
be substantially better than single-pass yes/no or answer-likelihood logits.

ARC-Challenge validation slice:

| Run | Correct |
| --- | ---: |
| Qwen3-0.6B Base, validation[:50] | 21 / 50 |
| Qwen3-0.6B Qwen-HRM, fixed blend | 21 / 50 |
| Qwen3-0.6B Qwen-HRM, agreement gate | 21 / 50 |
| Qwen3-1.7B Base, validation[:50] | 37 / 50 |
| Qwen3-1.7B Qwen-HRM, fixed blend | 35 / 50 |
| Qwen3-1.7B Qwen-HRM, agreement gate | 37 / 50 |

This dataset-backed slice weakens the earlier custom-MC story: the 0.6B gain did
not transfer to ARC, and fixed blend is risky outside the tiny hand-written MC
probe. Agreement-gated fusion is the safer no-training default, but it is
currently a preservation strategy rather than a reliable improvement strategy.

For broader evaluation, the repo's standard benchmark runner now has an MLX
Qwen-HRM engine. Use it for small Apple Silicon slices before spending time on
full benchmark passes:

```bash
python -m evaluation.main \
  config=evaluation/config/mlx_qwen_hrm_benchmarking.yaml
```

The included config starts with limited GSM8K and ARC-Challenge slices. Those are
better direction tests than the 17-item probe because they are dataset-backed and
less exposed to hand-written option priors.

Additional stability knobs:

- `--update-mix-l` and `--update-mix-h` under-relax recurrent state updates.
  Values below `1.0` keep part of the previous `z_L`/`z_H` state instead of
  fully replacing it with the next recurrent block output.
- `--refined-delta-scale` scales the final hidden delta from the base model to
  the refined state before logits are projected.
- `--split-index` overrides the automatic `L`/`H` layer split.
- `--logit-fusion` can test alternate output fusion. Current tracked options are
  `blend`, `delta`, `prob_blend`, `confidence_gate`, and `agreement_blend`.
  Confidence-gated, agreement-gated, probability-space, and extrapolated-delta
  fusion did not beat fixed logit blending on the corrected MC probe, so `blend`
  remains the scoring default. For open-ended generation, prefer
  `agreement_blend` or `confidence_gate` to avoid the fixed-blend degradation
  observed in chat-template exact-answer runs.
- `--fusion-after` delays the configured Qwen-HRM fusion until generated text
  contains one of the supplied comma-separated triggers. For exact-answer
  generation, `--fusion-after '\boxed,Final Answer,final answer'` keeps the
  reasoning prefix anchored to base logits and only enables HRM fusion near the
  final answer.
