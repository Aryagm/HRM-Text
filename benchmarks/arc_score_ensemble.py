from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev


def read_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["case_index"]): row for row in csv.DictReader(handle)}


def score_labels(row: dict[str, str]) -> list[str]:
    return sorted(key.removeprefix("score_") for key, value in row.items() if key.startswith("score_") and value != "")


def normalized_scores(row: dict[str, str], labels: list[str], normalize: str) -> dict[str, float]:
    scores = {label: float(row[f"score_{label}"]) for label in labels}
    if normalize == "none":
        return scores
    values = list(scores.values())
    if normalize == "center":
        offset = mean(values)
        return {label: score - offset for label, score in scores.items()}
    if normalize == "zscore":
        offset = mean(values)
        scale = pstdev(values)
        if scale <= 1e-12:
            return {label: 0.0 for label in scores}
        return {label: (score - offset) / scale for label, score in scores.items()}
    raise ValueError(f"Unsupported normalize mode {normalize!r}")


def parse_weights(raw: str, count: int) -> list[float]:
    weights = [float(item) for item in raw.split(",") if item]
    if len(weights) != count:
        raise ValueError(f"Expected {count} weights, got {len(weights)}")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine ARC probe score CSVs with fixed no-training score weights.")
    parser.add_argument("--score-csv", type=Path, action="append", required=True)
    parser.add_argument("--weights", default=None, help="Comma-separated weights; defaults to equal weights.")
    parser.add_argument("--normalize", choices=("none", "center", "zscore"), default="zscore")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sources = [read_rows(path) for path in args.score_csv]
    weights = parse_weights(args.weights, len(sources)) if args.weights else [1.0 / len(sources)] * len(sources)

    rows = []
    for case_index in sorted(set.intersection(*(set(source) for source in sources))):
        base = sources[0][case_index]
        labels = score_labels(base)
        combined = {label: 0.0 for label in labels}
        for weight, source in zip(weights, sources, strict=True):
            row = source[case_index]
            scores = normalized_scores(row, labels, args.normalize)
            for label in labels:
                combined[label] += weight * scores[label]
        prediction = max(combined.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": base["model"],
                "split": base["split"],
                "limit": base["limit"],
                "normalize": args.normalize,
                "weights": ",".join(str(weight) for weight in weights),
                "source_files": "|".join(path.name for path in args.score_csv),
                "case_index": case_index,
                "answer": base["answer"],
                "prediction": prediction,
                "is_correct": prediction == base["answer"],
                **{f"score_{label}": combined[label] for label in labels},
            }
        )

    correct = sum(row["is_correct"] for row in rows)
    total = len(rows)
    fields = [
        "timestamp_utc",
        "model",
        "split",
        "limit",
        "normalize",
        "weights",
        "correct",
        "total",
        "accuracy",
        "case_index",
        "answer",
        "prediction",
        "is_correct",
        "source_files",
    ]
    extra_fields = sorted({key for row in rows for key in row if key.startswith("score_")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"correct": correct, "total": total, "accuracy": correct / max(1, total), **row})
    print(f"score ensemble normalize={args.normalize} weights={weights}: {correct}/{total}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
