#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-exports/hrm-text-1b-mlx-bf16}"
RUN_NAME="${RUN_NAME:-hrm_text_mlx_bf16}"
BENCHMARKS="${BENCHMARKS:-ARC}"
OUT_DIR="${OUT_DIR:-outputs/hrm_text_mlx_eval}"
DTYPE="${DTYPE:-bfloat16}"
PYTHON="${PYTHON:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

"$PYTHON" -m benchmarks.hrm_text_mlx_eval \
  --model-dir "$MODEL_DIR" \
  --run-name "$RUN_NAME" \
  --benchmark "$BENCHMARKS" \
  --out-dir "$OUT_DIR" \
  --dtype "$DTYPE" \
  --resume \
  $EXTRA_ARGS
