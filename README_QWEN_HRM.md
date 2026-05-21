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
| Qwen3-0.6B Qwen-HRM, pure refined output | 14 / 50 |
| Qwen3-0.6B Qwen-HRM, agreement gate | 21 / 50 |
| Qwen3-1.7B Base, validation[:50] | 37 / 50 |
| Qwen3-1.7B Qwen-HRM, fixed blend | 35 / 50 |
| Qwen3-1.7B Qwen-HRM, pure refined output | 9 / 50 |
| Qwen3-1.7B Qwen-HRM, agreement gate | 37 / 50 |

This dataset-backed slice weakens the earlier custom-MC story: the 0.6B gain did
not transfer to ARC, and fixed blend is risky outside the tiny hand-written MC
probe. Agreement-gated fusion is the safer no-training default, but it is
currently a preservation strategy rather than a reliable improvement strategy.
Pure refined-output scoring is much worse, which means the current conversion's
recurrent path is not usable as a standalone HRM substitute without calibration
or training.

Exploratory ARC margin switch:

| Run | Correct |
| --- | ---: |
| Qwen3-0.6B Base | 21 / 50 |
| Qwen3-0.6B HRM when base margin <= 0.25 | 23 / 50 |
| Qwen3-1.7B Base | 37 / 50 |
| Qwen3-1.7B HRM when base margin <= 0.25 | 36 / 50 |

Validation on the next ARC slice rejected this rule:

| Run | Correct |
| --- | ---: |
| Qwen3-0.6B Base, validation[50:100] | 17 / 50 |
| Qwen3-0.6B fixed HRM, validation[50:100] | 17 / 50 |
| Qwen3-0.6B margin switch, validation[50:100] | 16 / 50 |

So the margin threshold was overfit to `validation[:50]`; do not use it as a
production rule.

Targeted 0.6B ARC architecture checks also stayed flat:

| Run | Correct |
| --- | ---: |
| Qwen3-0.6B agreement gate, default H=1 split | 21 / 50 |
| Qwen3-0.6B agreement gate, H=2 | 21 / 50 |
| Qwen3-0.6B agreement gate, split_index=10 | 21 / 50 |

So neither deeper recurrence nor moving the L/H split earlier improved the
dataset-backed 0.6B ARC slice.

Full ARC-Challenge validation:

| Run | Correct |
| --- | ---: |
| Qwen3-0.6B Base, validation full | 112 / 299 |
| Qwen3-0.6B agreement gate, validation full | 112 / 299 |
| Qwen3-0.6B Base, option-prior calibrated, validation full | 121 / 299 |
| Qwen3-0.6B agreement gate, option-prior calibrated, validation full | 121 / 299 |
| Qwen3-1.7B Base, validation full | 207 / 299 |
| Qwen3-1.7B agreement gate, validation full | 207 / 299 |
| Qwen3-1.7B Base, answer-prior calibrated, validation full | 211 / 299 |
| Qwen3-1.7B agreement gate, answer-prior calibrated, validation full | 211 / 299 |
| Qwen3-1.7B Base, answer-prior calibrated weight 1.7, validation full | 216 / 299 |
| Qwen3-1.7B agreement gate, answer-prior calibrated weight 1.7, validation full | 216 / 299 |

This confirms that agreement-gated fusion preserves base behavior on the full
validation split for both tested Qwen sizes, but does not create a measurable
ARC improvement.
No-training calibration does create a measurable ARC improvement: the smaller
Qwen3-0.6B checkpoint benefits from subtracting an options-only prior, while
Qwen3-1.7B benefits from subtracting a bare answer-label prior. Agreement-gated
Qwen-HRM preserves both calibrated gains.
Post-hoc sweeps over the saved raw/prior scores show Qwen3-1.7B has more
headroom with a stronger answer-prior subtraction: weight `1.7` reaches
`216 / 299` on full validation for both base and agreement-gated Qwen-HRM. Treat
that weight as tuned on ARC validation until it is tested on another benchmark.

Calibration transfer checks:

| Fit split | Eval split | Run | Selected weight | Eval correct | Eval no-cal | Eval oracle |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ARC-Challenge validation[:50] | ARC-Challenge validation[50:100] | Qwen3-0.6B Base, option-prior | 1.2 | 20 / 50 | 17 / 50 | 22 / 50 at 1.0 |
| ARC-Challenge validation[:50] | ARC-Challenge validation[50:100] | Qwen3-0.6B agreement gate, option-prior | 1.2 | 20 / 50 | 17 / 50 | 22 / 50 at 1.0 |
| ARC-Challenge validation[:50] | ARC-Challenge validation[50:100] | Qwen3-1.7B Base, answer-prior | 1.0 | 35 / 50 | 33 / 50 | 36 / 50 at 1.4 |
| ARC-Challenge validation[:50] | ARC-Challenge validation[50:100] | Qwen3-1.7B agreement gate, answer-prior | 1.0 | 35 / 50 | 33 / 50 | 36 / 50 at 1.4 |
| ARC-Challenge validation full | ARC-Easy validation full | Qwen3-0.6B Base, option-prior | 1.0 | 355 / 570 | 348 / 570 | 365 / 570 at 0.8 |
| ARC-Challenge validation full | ARC-Easy validation full | Qwen3-0.6B agreement gate, option-prior | 1.0 | 355 / 570 | 348 / 570 | 365 / 570 at 0.8 |
| ARC-Challenge validation full | ARC-Easy validation full | Qwen3-1.7B Base, answer-prior | 1.7 | 489 / 570 | 489 / 570 | 490 / 570 at 0.4 |

This makes the current best no-training rule more conservative: use
option-prior calibration for Qwen3-0.6B, where it transfers across held-out ARC
slices and ARC-Easy; do not use the aggressive Qwen3-1.7B weight `1.7` as a
general default because it fails to transfer to ARC-Easy.

Auto-calibration smoke runs:

| Run | Resolved calibration | Correct |
| --- | --- | ---: |
| Qwen3-0.6B Base, ARC-Challenge validation[:10], `--calibration auto` | `options_prior` weight `1.0` | 3 / 10 |
| Qwen3-1.7B Base, ARC-Challenge validation[:10], `--calibration auto` | `answer_prior` weight `1.0` | 8 / 10 |

These are not new headline benchmark claims; they verify the CLI resolves the
transferred policy correctly for both checkpoint sizes.

Tiny score-adapter validation:

| Train split | Eval split | Run | Eval raw | Eval fixed calibration | Eval adapter |
| --- | --- | --- | ---: | ---: | ---: |
| ARC-Challenge validation[:50] | ARC-Challenge validation[50:100] | Qwen3-1.7B recurrent score adapter over raw/prior option scores | 33 / 50 | 35 / 50 | 37 / 50 |
| ARC-Challenge validation[:50] | ARC-Challenge validation[50:100] | Qwen3-0.6B recurrent score adapter over raw/prior option scores | 17 / 50 | 22 / 50 | 20 / 50 |

This is the first quick positive sign for the adapter direction, but it is
partial: the trained adapter helps Qwen3-1.7B on the next held-out ARC slice and
hurts Qwen3-0.6B versus simple fixed calibration. Treat it as evidence that
small trained controllers over frozen Qwen score surfaces can help, not as proof
that the full HRM adapter strategy is solved.

ARC-Easy transfer check:

| Run | Correct |
| --- | ---: |
| Qwen3-0.6B Base, validation full | 348 / 570 |
| Qwen3-0.6B Base, option-prior calibrated, validation full | 355 / 570 |
| Qwen3-0.6B Base, option-prior calibrated weight 0.8, validation full post-hoc sweep | 365 / 570 |
| Qwen3-0.6B agreement gate, option-prior calibrated, validation full | 355 / 570 |
| Qwen3-0.6B agreement gate, option-prior calibrated weight 0.8, validation full post-hoc sweep | 365 / 570 |
| Qwen3-1.7B Base, validation full | 489 / 570 |
| Qwen3-1.7B Base, answer-prior calibrated weight 1.0, validation full | 488 / 570 |
| Qwen3-1.7B Base, answer-prior calibrated weight 0.4, validation full post-hoc sweep | 490 / 570 |
| Qwen3-1.7B Base, validation[:200] | 173 / 200 |
| Qwen3-1.7B Base, answer-prior calibrated weight 1.0, validation[:200] | 173 / 200 |
| Qwen3-1.7B Base, answer-prior calibrated weight 1.7, validation[:200] | 172 / 200 |
| Qwen3-1.7B agreement gate, answer-prior calibrated weight 1.0, validation[:200] | 173 / 200 |

So the 0.6B option-prior gain transfers to full ARC-Easy validation, and
weight tuning on the saved full-set scores gives a larger `+17` over base. The
1.7B answer-prior calibration does not transfer meaningfully to full ARC-Easy:
its best swept weight is only `+1` over base. The 1.7B weight `1.7` setting
looks ARC-Challenge-specific and should not be treated as a general default yet.

ARC scoring-surface follow-up:

| Run | Correct |
| --- | ---: |
| Qwen3-0.6B Base, choice text, validation[:50] | 14 / 50 |
| Qwen3-0.6B agreement gate, choice text, validation[:50] | 14 / 50 |
| Qwen3-0.6B Base, label + text, validation[:50] | 14 / 50 |
| Qwen3-0.6B agreement gate, label + text, validation[:50] | 14 / 50 |
| Qwen3-1.7B Base, choice text, validation[:50] | 20 / 50 |
| Qwen3-1.7B agreement gate, choice text, validation[:50] | 20 / 50 |
| Qwen3-1.7B Base, label + text, validation[:50] | 37 / 50 |
| Qwen3-1.7B agreement gate, label + text, validation[:50] | 37 / 50 |
| Qwen3-1.7B label + text margin switch, validation[:50] | 38 / 50 |
| Qwen3-1.7B Base, validation[50:100] | 33 / 50 |
| Qwen3-1.7B label + text margin switch, validation[50:100] | 33 / 50 |

The richer continuation surface creates one selectable first-slice win for
Qwen3-1.7B, but it does not improve the next slice. Equal z-score ensembles
across label, choice-text, and label-text scores were flat for 0.6B and worse
for 1.7B. Treat this as a useful diagnostic surface, not a solid inference
upgrade.

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
- `--arc-config` selects `ARC-Challenge` or `ARC-Easy` in
  `benchmarks/qwen_arc_probe.py`.
- `--calibration` on `benchmarks/qwen_arc_probe.py` subtracts a no-training
  prior from each ARC answer score. Tracked options are `answer_prior` and
  `options_prior`; full-validation runs show `options_prior` is best so far for
  Qwen3-0.6B and `answer_prior` is best so far for Qwen3-1.7B.
- `--calibration auto` applies the conservative transferred defaults:
  Qwen3-0.6B uses `options_prior` at weight `1.0`, and Qwen3-1.7B uses
  `answer_prior` at weight `1.0`. It deliberately avoids the Qwen3-1.7B
  ARC-Challenge-tuned `1.7` weight because that setting failed to transfer to
  ARC-Easy.
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
