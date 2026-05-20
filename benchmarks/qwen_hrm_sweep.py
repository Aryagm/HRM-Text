from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time

import mlx.core as mx
from transformers import AutoTokenizer

from benchmarks.qwen_hrm_reasoning_probe import CASES, evaluate_mode
from mlx_hrm_text.qwen_hrm import QwenHrmForCausalLM


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_str_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep no-training Qwen3 HRM conversion settings.")
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--out", type=Path, default=Path("outputs/qwen_hrm_metrics/sweep_results.csv"))
    parser.add_argument("--h-cycles", default="1,2")
    parser.add_argument("--l-cycles", default="1")
    parser.add_argument("--logit-blends", default="0.0,0.02,0.05,0.1,0.2,0.35")
    parser.add_argument("--alpha-l", default="0.0,0.01,0.03,0.05")
    parser.add_argument("--alpha-h", default="0.0,0.02,0.05,0.08")
    parser.add_argument("--beta", default="0.0,0.01")
    parser.add_argument("--update-mix", default="0.25,0.5,1.0")
    parser.add_argument("--refined-delta-scale", default="0.25,0.5,1.0")
    parser.add_argument("--split-index", default="", help="Comma-separated L/H split points. Empty uses the checkpoint default.")
    parser.add_argument("--logit-fusion", default="blend", help="Comma-separated fusion modes: blend,confidence_gate,agreement_blend.")
    parser.add_argument("--fusion-threshold", default="0.0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = QwenHrmForCausalLM.from_pretrained(args.model)
    mx.eval(model.parameters())

    configs = []
    split_indices = parse_int_list(args.split_index) if args.split_index else [model.config.lower_layers]
    configs.append(
        {
            "label": "base",
            "h_cycles": 0,
            "l_cycles": 1,
            "logit_blend": 0.0,
            "alpha_l": 0.0,
            "alpha_h": 0.0,
            "beta_l": 0.0,
            "beta_h": 0.0,
            "update_mix_l": 1.0,
            "update_mix_h": 1.0,
            "refined_delta_scale": 0.0,
            "split_index": model.config.lower_layers,
            "logit_fusion": "blend",
            "fusion_threshold": 0.0,
        }
    )

    for split_index in split_indices:
        for h_cycles in parse_int_list(args.h_cycles):
            for l_cycles in parse_int_list(args.l_cycles):
                for blend in parse_float_list(args.logit_blends):
                    if blend <= 0:
                        continue
                    for logit_fusion in parse_str_list(args.logit_fusion):
                        for fusion_threshold in parse_float_list(args.fusion_threshold):
                            for alpha_l in parse_float_list(args.alpha_l):
                                for alpha_h in parse_float_list(args.alpha_h):
                                    for beta in parse_float_list(args.beta):
                                        for update_mix in parse_float_list(args.update_mix):
                                            for delta_scale in parse_float_list(args.refined_delta_scale):
                                                configs.append(
                                                    {
                                                        "label": "hrm",
                                                        "h_cycles": h_cycles,
                                                        "l_cycles": l_cycles,
                                                        "logit_blend": blend,
                                                        "alpha_l": alpha_l,
                                                        "alpha_h": alpha_h,
                                                        "beta_l": beta,
                                                        "beta_h": beta,
                                                        "update_mix_l": update_mix,
                                                        "update_mix_h": update_mix,
                                                        "refined_delta_scale": delta_scale,
                                                        "split_index": split_index,
                                                        "logit_fusion": logit_fusion,
                                                        "fusion_threshold": fusion_threshold,
                                                    }
                                                )

    if args.limit is not None:
        configs = configs[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "model",
        "run_id",
        "label",
        "accuracy",
        "correct",
        "total",
        "elapsed_s",
        "h_cycles",
        "l_cycles",
        "logit_blend",
        "alpha_l",
        "alpha_h",
        "beta_l",
        "beta_h",
        "update_mix_l",
        "update_mix_h",
        "refined_delta_scale",
        "split_index",
        "logit_fusion",
        "fusion_threshold",
        "case",
        "expected",
        "prediction",
        "is_correct",
        "score_A",
        "score_B",
        "score_C",
        "score_D",
    ]

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for run_id, cfg in enumerate(configs):
            start = time.perf_counter()
            correct, rows = evaluate_mode(
                model,
                tokenizer,
                h_cycles=int(cfg["h_cycles"]),
                l_cycles=int(cfg["l_cycles"]),
                logit_blend=float(cfg["logit_blend"]),
                alpha_l=float(cfg["alpha_l"]),
                alpha_h=float(cfg["alpha_h"]),
                beta_l=float(cfg["beta_l"]),
                beta_h=float(cfg["beta_h"]),
                update_mix_l=float(cfg["update_mix_l"]),
                update_mix_h=float(cfg["update_mix_h"]),
                refined_delta_scale=float(cfg["refined_delta_scale"]),
                split_index=int(cfg["split_index"]),
                logit_fusion=str(cfg["logit_fusion"]),
                fusion_threshold=float(cfg["fusion_threshold"]),
            )
            elapsed = time.perf_counter() - start
            accuracy = correct / len(CASES)
            print(
                f"{run_id:04d} {cfg['label']} acc={correct}/{len(CASES)} elapsed={elapsed:.2f}s "
                f"H={cfg['h_cycles']} blend={cfg['logit_blend']} a=({cfg['alpha_l']},{cfg['alpha_h']}) "
                f"beta={cfg['beta_l']} mix={cfg['update_mix_l']} delta={cfg['refined_delta_scale']} "
                f"split={cfg['split_index']} fusion={cfg['logit_fusion']} threshold={cfg['fusion_threshold']}",
                flush=True,
            )

            timestamp = datetime.now(timezone.utc).isoformat()
            for case, prediction, scores in rows:
                writer.writerow(
                    {
                        "timestamp_utc": timestamp,
                        "model": args.model,
                        "run_id": run_id,
                        "label": cfg["label"],
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": len(CASES),
                        "elapsed_s": elapsed,
                        **cfg,
                        "case": case.name,
                        "expected": case.answer,
                        "prediction": prediction,
                        "is_correct": prediction == case.answer,
                        "score_A": scores.get("A"),
                        "score_B": scores.get("B"),
                        "score_C": scores.get("C"),
                        "score_D": scores.get("D"),
                    }
                )
            handle.flush()

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
