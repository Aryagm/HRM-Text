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
        if key.startswith("raw_score_") and value not in ("", "NA"):
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
        scores = {
            label: float(row[f"raw_score_{label}"]) - weight * float(row[f"calibration_score_{label}"])
            for label in labels
        }
        prediction = max(scores.items(), key=lambda item: item[1])[0]
        if prediction == row["answer"]:
            correct += 1
            correct_cases.append(row["case_index"])
    return correct, correct_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ARC calibration weights from saved raw/prior score CSVs.")
    parser.add_argument("--score-csv", type=Path, required=True)
    parser.add_argument("--weights", default=None, help="Comma-separated explicit weights. Overrides start/stop/step.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.score_csv)
    if not rows:
        raise ValueError(f"No rows in {args.score_csv}")
    weights = parse_weights(args)
    first = rows[0]
    total = len(rows)

    out_rows = []
    for weight in weights:
        correct, correct_cases = evaluate_weight(rows, weight)
        out_rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": first["model"],
                "split": first["split"],
                "limit": first["limit"],
                "mode": first["mode"],
                "answer_scoring": first.get("answer_scoring", "label"),
                "calibration": first["calibration"],
                "source_weight": first["calibration_weight"],
                "sweep_weight": weight,
                "correct": correct,
                "total": total,
                "accuracy": correct / max(1, total),
                "correct_cases": ", ".join(correct_cases),
                "source_file": args.score_csv.name,
            }
        )

    fields = [
        "timestamp_utc",
        "model",
        "split",
        "limit",
        "mode",
        "answer_scoring",
        "calibration",
        "source_weight",
        "sweep_weight",
        "correct",
        "total",
        "accuracy",
        "correct_cases",
        "source_file",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    best = max(out_rows, key=lambda row: (int(row["correct"]), -abs(float(row["sweep_weight"]) - 1.0)))
    print(f"{args.score_csv.name}: best weight={best['sweep_weight']} {best['correct']}/{best['total']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
