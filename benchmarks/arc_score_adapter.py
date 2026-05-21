from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OptionGroup:
    features: np.ndarray
    labels: list[str]
    answer_idx: int
    row: dict[str, str]


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


def make_groups(rows: list[dict[str, str]]) -> list[OptionGroup]:
    groups = []
    for row in rows:
        labels = score_labels(row)
        raw = np.array([float(row[f"raw_score_{label}"]) for label in labels], dtype=np.float64)
        prior = np.array([float(row[f"calibration_score_{label}"]) for label in labels], dtype=np.float64)
        raw_rank = (-raw).argsort().argsort()
        prior_rank = (-prior).argsort().argsort()
        features = []
        for idx, label in enumerate(labels):
            label_one_hot = [1.0 if pos == idx else 0.0 for pos in range(5)]
            features.append(
                [
                    raw[idx],
                    prior[idx],
                    raw[idx] - prior[idx],
                    raw[idx] - raw.mean(),
                    prior[idx] - prior.mean(),
                    float(raw_rank[idx]),
                    float(prior_rank[idx]),
                    float(len(labels)),
                    *label_one_hot,
                ]
            )
        groups.append(
            OptionGroup(
                features=np.array(features, dtype=np.float64),
                labels=labels,
                answer_idx=labels.index(row["answer"]),
                row=row,
            )
        )
    return groups


def normalize_features(groups: list[OptionGroup]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack([group.features for group in groups])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def train_linear(
    groups: list[OptionGroup],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[tuple[np.ndarray, float], np.ndarray, np.ndarray]:
    mean, std = normalize_features(groups)
    weights = np.zeros(groups[0].features.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        grad_w = np.zeros_like(weights)
        grad_b = 0.0
        for group in groups:
            x = (group.features - mean) / std
            probs = softmax(x @ weights + bias)
            probs[group.answer_idx] -= 1.0
            grad_w += x.T @ probs
            grad_b += probs.sum()
        scale = 1.0 / max(1, len(groups))
        weights -= lr * (scale * grad_w + weight_decay * weights)
        bias -= lr * scale * grad_b
    return (weights, bias), mean, std


def recurrent_scores(x: np.ndarray, params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    w_in, w_rec, bias, w_out, out_bias = params
    hidden = np.zeros((x.shape[0], w_rec.shape[0]), dtype=np.float64)
    states = []
    for _ in range(2):
        pre = x @ w_in.T + hidden @ w_rec.T + bias
        hidden = np.tanh(pre)
        states.append((hidden, pre))
    return hidden @ w_out + out_bias, states


def train_recurrent(
    groups: list[OptionGroup],
    *,
    seed: int,
    hidden_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray, np.ndarray]:
    mean, std = normalize_features(groups)
    rng = np.random.default_rng(seed)
    dim = groups[0].features.shape[1]
    w_in = rng.normal(0.0, 0.1, size=(hidden_size, dim))
    w_rec = rng.normal(0.0, 0.04, size=(hidden_size, hidden_size))
    bias = np.zeros(hidden_size, dtype=np.float64)
    w_out = rng.normal(0.0, 0.1, size=hidden_size)
    out_bias = 0.0

    for _ in range(epochs):
        grad_in = np.zeros_like(w_in)
        grad_rec = np.zeros_like(w_rec)
        grad_bias = np.zeros_like(bias)
        grad_out = np.zeros_like(w_out)
        grad_out_bias = 0.0
        for group in groups:
            x = (group.features - mean) / std
            scores, states = recurrent_scores(x, (w_in, w_rec, bias, w_out, out_bias))
            probs = softmax(scores)
            probs[group.answer_idx] -= 1.0
            grad_out += states[-1][0].T @ probs
            grad_out_bias += probs.sum()
            grad_hidden = np.outer(probs, w_out)
            previous_hidden = [np.zeros_like(states[0][0]), states[0][0]]
            for step in reversed(range(2)):
                hidden, _ = states[step]
                grad_pre = grad_hidden * (1.0 - hidden * hidden)
                grad_in += grad_pre.T @ x
                grad_rec += grad_pre.T @ previous_hidden[step]
                grad_bias += grad_pre.sum(axis=0)
                grad_hidden = grad_pre @ w_rec
        scale = 1.0 / max(1, len(groups))
        w_in -= lr * (scale * grad_in + weight_decay * w_in)
        w_rec -= lr * (scale * grad_rec + weight_decay * w_rec)
        bias -= lr * scale * grad_bias
        w_out -= lr * (scale * grad_out + weight_decay * w_out)
        out_bias -= lr * scale * grad_out_bias
    return (w_in, w_rec, bias, w_out, out_bias), mean, std


def predict_linear(group: OptionGroup, params: tuple[tuple[np.ndarray, float], np.ndarray, np.ndarray]) -> tuple[int, np.ndarray]:
    (weights, bias), mean, std = params
    scores = ((group.features - mean) / std) @ weights + bias
    return int(scores.argmax()), scores


def predict_recurrent(
    group: OptionGroup,
    params: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray, np.ndarray],
) -> tuple[int, np.ndarray]:
    weights, mean, std = params
    scores, _ = recurrent_scores((group.features - mean) / std, weights)
    return int(scores.argmax()), scores


def baseline_prediction(group: OptionGroup, weight: float) -> int:
    labels = group.labels
    scores = {
        label: float(group.row[f"raw_score_{label}"]) - weight * float(group.row[f"calibration_score_{label}"])
        for label in labels
    }
    return labels.index(max(scores, key=scores.get))


def evaluate(
    groups: list[OptionGroup],
    *,
    adapter: str,
    params,
) -> tuple[int, list[dict[str, object]]]:
    correct = 0
    rows = []
    predictor = predict_linear if adapter == "linear" else predict_recurrent
    for group in groups:
        pred_idx, scores = predictor(group, params)
        is_correct = pred_idx == group.answer_idx
        correct += int(is_correct)
        rows.append(
            {
                "case_index": group.row["case_index"],
                "answer": group.row["answer"],
                "prediction": group.labels[pred_idx],
                "is_correct": is_correct,
                "adapter_margin": float(np.sort(scores)[-1] - np.sort(scores)[-2]) if len(scores) > 1 else 0.0,
                "raw_prediction": group.labels[baseline_prediction(group, 0.0)],
                "weight1_prediction": group.labels[baseline_prediction(group, 1.0)],
            }
        )
    return correct, rows


def count_baseline(groups: list[OptionGroup], weight: float) -> int:
    return sum(baseline_prediction(group, weight) == group.answer_idx for group in groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny adapter over saved ARC raw/prior option scores.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", choices=("linear", "recurrent"), default="recurrent")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.003)
    args = parser.parse_args()

    train_groups = make_groups(read_rows(args.train_csv))
    eval_groups = make_groups(read_rows(args.eval_csv))
    if args.adapter == "linear":
        params = train_linear(train_groups, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
    else:
        params = train_recurrent(
            train_groups,
            seed=args.seed,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    train_correct, _ = evaluate(train_groups, adapter=args.adapter, params=params)
    eval_correct, eval_rows = evaluate(eval_groups, adapter=args.adapter, params=params)
    train_first = train_groups[0].row
    eval_first = eval_groups[0].row
    timestamp = datetime.now(timezone.utc).isoformat()
    summary = {
        "timestamp_utc": timestamp,
        "adapter": args.adapter,
        "seed": args.seed,
        "hidden_size": args.hidden_size if args.adapter == "recurrent" else 0,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model": train_first["model"],
        "mode": train_first["mode"],
        "answer_scoring": train_first.get("answer_scoring", "label"),
        "calibration": train_first["calibration"],
        "train_source_file": args.train_csv.name,
        "train_arc_config": train_first.get("arc_config", "ARC-Challenge"),
        "train_split": train_first["split"],
        "train_limit": train_first["limit"],
        "train_correct": train_correct,
        "train_total": len(train_groups),
        "eval_source_file": args.eval_csv.name,
        "eval_arc_config": eval_first.get("arc_config", "ARC-Challenge"),
        "eval_split": eval_first["split"],
        "eval_limit": eval_first["limit"],
        "eval_correct": eval_correct,
        "eval_total": len(eval_groups),
        "eval_accuracy": eval_correct / max(1, len(eval_groups)),
        "eval_raw_correct": count_baseline(eval_groups, 0.0),
        "eval_weight1_correct": count_baseline(eval_groups, 1.0),
    }
    fields = [
        *summary.keys(),
        "case_index",
        "answer",
        "prediction",
        "is_correct",
        "adapter_margin",
        "raw_prediction",
        "weight1_prediction",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in eval_rows:
            writer.writerow({**summary, **row})
    print(
        f"{args.adapter} adapter {args.train_csv.name} -> {args.eval_csv.name}: "
        f"{eval_correct}/{len(eval_groups)} "
        f"(raw={summary['eval_raw_correct']}, weight1={summary['eval_weight1_correct']})"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
