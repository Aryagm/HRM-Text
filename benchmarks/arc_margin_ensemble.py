from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["case_index"]): row for row in csv.DictReader(handle)}


def score_margin(row: dict[str, str]) -> float:
    scores = sorted(float(value) for key, value in row.items() if key.startswith("score_") and value != "")
    if len(scores) < 2:
        return 0.0
    return scores[-1] - scores[-2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine ARC base and HRM probe CSVs with a base-margin switch rule.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--hrm", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_rows(args.base)
    hrm_rows = read_rows(args.hrm)
    rows = []
    for case_index in sorted(base_rows):
        base = base_rows[case_index]
        hrm = hrm_rows[case_index]
        margin = score_margin(base)
        use_hrm = margin <= args.threshold
        selected = hrm if use_hrm else base
        rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": base["model"],
                "split": base["split"],
                "limit": base["limit"],
                "threshold": args.threshold,
                "case_index": case_index,
                "answer": base["answer"],
                "base_prediction": base["prediction"],
                "hrm_prediction": hrm["prediction"],
                "prediction": selected["prediction"],
                "is_correct": selected["prediction"] == base["answer"],
                "base_margin": margin,
                "used_hrm": use_hrm,
                "base_source": args.base.name,
                "hrm_source": args.hrm.name,
            }
        )

    correct = sum(row["is_correct"] for row in rows)
    total = len(rows)
    fields = [
        "timestamp_utc",
        "model",
        "split",
        "limit",
        "threshold",
        "correct",
        "total",
        "accuracy",
        "case_index",
        "answer",
        "base_prediction",
        "hrm_prediction",
        "prediction",
        "is_correct",
        "base_margin",
        "used_hrm",
        "base_source",
        "hrm_source",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"correct": correct, "total": total, "accuracy": correct / max(1, total), **row})
    print(f"margin ensemble threshold={args.threshold}: {correct}/{total}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
