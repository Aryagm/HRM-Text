from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def score_labels(row: dict[str, str]) -> list[str]:
    labels = []
    for key, value in row.items():
        if not key.startswith("raw_score_") or value in ("", "NA"):
            continue
        label = key.removeprefix("raw_score_")
        if row.get(f"calibration_score_{label}") not in ("", "NA", None):
            labels.append(label)
    return sorted(labels)


def parse_weights(args: argparse.Namespace) -> list[float]:
    if args.weights:
        return [float(item) for item in args.weights.split(",") if item]
    count = int(round((args.stop - args.start) / args.step))
    return [round(args.start + idx * args.step, 10) for idx in range(count + 1)]


def evaluate_weight(rows: list[dict[str, str]], weight: float) -> tuple[int, list[str]]:
    correct = 0
    correct_cases = []
    for row in rows:
        labels = score_labels(row)
        if not labels:
            raise ValueError("Rows must include raw_score_* and calibration_score_* columns")
        scores = {
            label: float(row[f"raw_score_{label}"]) - weight * float(row[f"calibration_score_{label}"])
            for label in labels
        }
        prediction = max(scores.items(), key=lambda item: item[1])[0]
        if prediction == row["answer"]:
            correct += 1
            correct_cases.append(row["case_index"])
    return correct, correct_cases


def best_weight(rows: list[dict[str, str]], weights: list[float], tie_weight: float) -> tuple[float, int]:
    scored = [(weight, evaluate_weight(rows, weight)[0]) for weight in weights]
    weight, correct = max(scored, key=lambda item: (item[1], -abs(item[0] - tie_weight)))
    return weight, correct


def first_metadata(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("No rows available")
    return rows[0]


def result_row(
    *,
    fit_file: Path,
    eval_file: Path,
    fit_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    weights: list[float],
    tie_weight: float,
) -> dict[str, object]:
    fit_first = first_metadata(fit_rows)
    eval_first = first_metadata(eval_rows)
    selected_weight, fit_correct = best_weight(fit_rows, weights, tie_weight)
    eval_correct, eval_cases = evaluate_weight(eval_rows, selected_weight)
    eval_no_cal, _ = evaluate_weight(eval_rows, 0.0)
    eval_weight_1, _ = evaluate_weight(eval_rows, 1.0)
    eval_oracle_weight, eval_oracle_correct = best_weight(eval_rows, weights, tie_weight)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": fit_first["model"],
        "mode": fit_first["mode"],
        "answer_scoring": fit_first.get("answer_scoring", "label"),
        "calibration": fit_first["calibration"],
        "fit_arc_config": fit_first.get("arc_config", "ARC-Challenge"),
        "fit_split": fit_first["split"],
        "fit_limit": fit_first["limit"],
        "fit_source_file": fit_file.name,
        "fit_weight": selected_weight,
        "fit_correct": fit_correct,
        "fit_total": len(fit_rows),
        "fit_accuracy": fit_correct / max(1, len(fit_rows)),
        "eval_arc_config": eval_first.get("arc_config", "ARC-Challenge"),
        "eval_split": eval_first["split"],
        "eval_limit": eval_first["limit"],
        "eval_source_file": eval_file.name,
        "eval_correct": eval_correct,
        "eval_total": len(eval_rows),
        "eval_accuracy": eval_correct / max(1, len(eval_rows)),
        "eval_no_cal_correct": eval_no_cal,
        "eval_weight_1_correct": eval_weight_1,
        "eval_oracle_weight": eval_oracle_weight,
        "eval_oracle_correct": eval_oracle_correct,
        "eval_delta_vs_no_cal": eval_correct - eval_no_cal,
        "eval_delta_vs_weight_1": eval_correct - eval_weight_1,
        "eval_correct_cases": ", ".join(eval_cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit an ARC calibration weight on one saved score CSV and evaluate it on another."
    )
    parser.add_argument("--fit-csv", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--weights", default=None, help="Comma-separated explicit weights. Overrides start/stop/step.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--tie-weight", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    fit_rows = read_rows(args.fit_csv)
    eval_rows = read_rows(args.eval_csv)
    weights = parse_weights(args)
    row = result_row(
        fit_file=args.fit_csv,
        eval_file=args.eval_csv,
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        weights=weights,
        tie_weight=args.tie_weight,
    )

    fields = [
        "timestamp_utc",
        "model",
        "mode",
        "answer_scoring",
        "calibration",
        "fit_arc_config",
        "fit_split",
        "fit_limit",
        "fit_source_file",
        "fit_weight",
        "fit_correct",
        "fit_total",
        "fit_accuracy",
        "eval_arc_config",
        "eval_split",
        "eval_limit",
        "eval_source_file",
        "eval_correct",
        "eval_total",
        "eval_accuracy",
        "eval_no_cal_correct",
        "eval_weight_1_correct",
        "eval_oracle_weight",
        "eval_oracle_correct",
        "eval_delta_vs_no_cal",
        "eval_delta_vs_weight_1",
        "eval_correct_cases",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    write_header = not args.append or not args.out.exists()
    with args.out.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"{args.fit_csv.name} -> {args.eval_csv.name}: "
        f"fit weight={row['fit_weight']} eval={row['eval_correct']}/{row['eval_total']} "
        f"delta_no_cal={row['eval_delta_vs_no_cal']} oracle={row['eval_oracle_weight']}:{row['eval_oracle_correct']}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
