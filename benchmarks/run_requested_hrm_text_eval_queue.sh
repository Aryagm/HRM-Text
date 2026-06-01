#!/usr/bin/env bash
set -u

OUT_DIR="${OUT_DIR:-outputs/hrm_text_mlx_eval}"
PYTHON="${PYTHON:-.venv/bin/python}"
STATUS_FILE="$OUT_DIR/requested_benchmarks_queue.status.tsv"
QUEUE_LOG="$OUT_DIR/requested_benchmarks_queue.log"

mkdir -p "$OUT_DIR"

timestamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

bench_lc() {
  printf "%s" "$1" | tr "[:upper:]" "[:lower:]"
}

rows_written() {
  local path="$1"
  if [ -f "$path" ]; then
    wc -l < "$path" | tr -d " "
  else
    printf "0"
  fi
}

write_status() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(timestamp)" "$1" "$2" "$3" "$4" "$5" | tee -a "$STATUS_FILE"
}

run_job() {
  local model_label="$1"
  local model_dir="$2"
  local run_name="$3"
  local benchmark="$4"
  local lower
  lower="$(bench_lc "$benchmark")"

  local prediction_path="$OUT_DIR/${run_name}_${lower}_predictions.jsonl"
  local summary_path="$OUT_DIR/${run_name}_${lower}_summary.json"
  local log_path="$OUT_DIR/${run_name}_${lower}.log"

  if [ -f "$summary_path" ]; then
    write_status "$model_label" "$benchmark" "SKIP_SUMMARY_EXISTS" "$(rows_written "$prediction_path")" "$summary_path"
    return 0
  fi

  write_status "$model_label" "$benchmark" "START" "$(rows_written "$prediction_path")" "$summary_path"
  (
    echo
    echo "===== $(timestamp) START $model_label $benchmark ====="
    time PYTHON="$PYTHON" MODEL_DIR="$model_dir" RUN_NAME="$run_name" BENCHMARKS="$benchmark" OUT_DIR="$OUT_DIR" \
      bash benchmarks/run_hrm_text_mlx_benchmarks.sh
    rc=$?
    echo "===== $(timestamp) END $model_label $benchmark rc=$rc rows=$(rows_written "$prediction_path") ====="
    exit "$rc"
  ) >> "$log_path" 2>&1

  local rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "$model_label" "$benchmark" "FAIL_RC_$rc" "$(rows_written "$prediction_path")" "$summary_path"
    return "$rc"
  fi

  write_status "$model_label" "$benchmark" "DONE" "$(rows_written "$prediction_path")" "$summary_path"
}

{
  echo "===== $(timestamp) queue start ====="
  echo "status_file=$STATUS_FILE"
  echo "out_dir=$OUT_DIR"
  echo "python=$PYTHON"
} >> "$QUEUE_LOG"

if [ ! -f "$STATUS_FILE" ]; then
  printf "timestamp\tmodel\tbenchmark\tstatus\trows\tsummary\n" > "$STATUS_FILE"
fi

write_status "bf16" "ARC" "DONE_PREVIOUS" "1172" "arc_mlx_bf16_arc_challenge_predictions.jsonl"
write_status "mxfp4" "ARC" "DONE_PREVIOUS" "1172" "arc_mlx_4bit_mxfp4_arc_challenge_predictions.jsonl"

run_job "bf16" "exports/hrm-text-1b-mlx-bf16" "hrm_text_bf16" "MMLU" || exit $?
run_job "mxfp4" "exports/hrm-text-1b-mlx-mxfp4" "hrm_text_mxfp4" "MMLU" || exit $?
run_job "bf16" "exports/hrm-text-1b-mlx-bf16" "hrm_text_bf16" "DROP" || exit $?
run_job "mxfp4" "exports/hrm-text-1b-mlx-mxfp4" "hrm_text_mxfp4" "DROP" || exit $?
run_job "bf16" "exports/hrm-text-1b-mlx-bf16" "hrm_text_bf16" "MATH" || exit $?
run_job "mxfp4" "exports/hrm-text-1b-mlx-mxfp4" "hrm_text_mxfp4" "MATH" || exit $?

write_status "all" "requested" "ALL_DONE" "-" "$OUT_DIR"
echo "===== $(timestamp) queue done =====" >> "$QUEUE_LOG"
