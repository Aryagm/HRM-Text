# HRM-Text MLX benchmark runs

This runner evaluates local MLX HRM-Text checkpoints with the benchmark classes
from `evaluation/benchmarks.py`. It writes incremental JSONL predictions and a
summary JSON per benchmark so long Apple Silicon runs can be resumed.

## Commands

ARC BF16:

```bash
PYTHON=.venv/bin/python \
MODEL_DIR=exports/hrm-text-1b-mlx-bf16 \
RUN_NAME=hrm_text_bf16 \
BENCHMARKS=ARC \
bash benchmarks/run_hrm_text_mlx_benchmarks.sh
```

ARC 4-bit MXFP4:

```bash
PYTHON=.venv/bin/python \
MODEL_DIR=exports/hrm-text-1b-mlx-mxfp4 \
RUN_NAME=hrm_text_mxfp4 \
BENCHMARKS=ARC \
bash benchmarks/run_hrm_text_mlx_benchmarks.sh
```

Other direct-answer MCQ benchmarks:

```bash
PYTHON=.venv/bin/python \
MODEL_DIR=exports/hrm-text-1b-mlx-bf16 \
RUN_NAME=hrm_text_bf16_mcq \
BENCHMARKS=MMLU,HellaSwag,Winogrande,BoolQ \
bash benchmarks/run_hrm_text_mlx_benchmarks.sh
```

Free-form benchmarks:

```bash
PYTHON=.venv/bin/python \
MODEL_DIR=exports/hrm-text-1b-mlx-bf16 \
RUN_NAME=hrm_text_bf16_freeform \
BENCHMARKS=DROP,MATH \
bash benchmarks/run_hrm_text_mlx_benchmarks.sh
```

Smoke test:

```bash
PYTHON=.venv/bin/python \
MODEL_DIR=exports/hrm-text-1b-mlx-bf16 \
RUN_NAME=smoke \
BENCHMARKS=Winogrande \
EXTRA_ARGS="--limit 10" \
bash benchmarks/run_hrm_text_mlx_benchmarks.sh
```

## Measured ARC-Challenge results

Protocol: ARC-Challenge test split, 25-shot, direct condition, max context
4096, one generated answer token, 1172 examples.

| Model | Accuracy | Correct | Wrong | Invalid | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX BF16 | 81.74% | 958/1172 | 214 | 0 | 3546.90s |
| MLX 4-bit MXFP4 g32 | 80.97% | 949/1172 | 223 | 0 | 3666.09s |

The public HRM-Text-1B ARC-C number is 81.91%, so the BF16 MLX run is within
0.17 percentage points. The MXFP4 run is 0.77 percentage points below BF16.

Prediction diff:

- 59 predictions changed between BF16 and MXFP4.
- 929 examples were correct for both.
- 194 examples were wrong for both.
- 29 examples were correct in BF16 and wrong in MXFP4.
- 20 examples were wrong in BF16 and correct in MXFP4.

## Runtime note

ARC took about one hour per checkpoint on the measured M4 Max run. Running every
benchmark for both BF16 and 4-bit is not expected to fit in 12 hours with this
sequential MLX runner. The direct-answer MCQ tasks are the best next targets;
MATH and DROP are substantially longer because they require free-form
generation.
