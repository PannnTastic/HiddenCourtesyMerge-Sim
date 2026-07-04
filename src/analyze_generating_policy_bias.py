"""Analyze feature-distribution shift across data-generating ego policies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


FEATURES = {
    "ard": "observed_abs_relative_distance",
    "mvs": "observed_merge_speed",
    "mva": "observed_merge_acceleration",
}
COURTESY_TYPES = ("cooperative", "non_cooperative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="First dataset directory, e.g. dataset_belief_ego")
    parser.add_argument("--target", required=True, help="Second dataset directory, e.g. dataset_rule_ego")
    parser.add_argument("--output", default="generating_policy_bias_analysis")
    return parser.parse_args()


def load_steps(dataset_dir: str) -> pd.DataFrame:
    path = Path(dataset_dir) / "steps_with_belief.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    return df[df["in_interaction_window"].astype(bool)].dropna(subset=list(FEATURES.values()))


def fit_diag_model(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    model: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in COURTESY_TYPES:
        sub = df[df["hidden_courtesy"] == mode][list(FEATURES.values())].to_numpy(dtype=float)
        model[mode] = {
            "mu": sub.mean(axis=0),
            "sigma": np.maximum(sub.std(axis=0, ddof=1), 1e-6),
        }
    return model


def predict_diag(model: Dict[str, Dict[str, np.ndarray]], df: pd.DataFrame) -> float:
    x = df[list(FEATURES.values())].to_numpy(dtype=float)
    scores = []
    for mode in COURTESY_TYPES:
        mu = model[mode]["mu"]
        sigma = model[mode]["sigma"]
        logp = -0.5 * np.sum(((x - mu) / sigma) ** 2 + np.log(2 * math.pi * sigma**2), axis=1)
        scores.append(logp)
    pred_idx = np.argmax(np.vstack(scores).T, axis=1)
    true_idx = np.array([COURTESY_TYPES.index(v) for v in df["hidden_courtesy"]])
    return float(np.mean(pred_idx == true_idx))


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    source = load_steps(args.source)
    target = load_steps(args.target)
    rows = []
    for mode in COURTESY_TYPES:
        s = source[source["hidden_courtesy"] == mode]
        t = target[target["hidden_courtesy"] == mode]
        for feat, col in FEATURES.items():
            s_mu, t_mu = float(s[col].mean()), float(t[col].mean())
            s_sd, t_sd = float(s[col].std(ddof=1)), float(t[col].std(ddof=1))
            pooled = math.sqrt(max((s_sd**2 + t_sd**2) / 2.0, 1e-12))
            rows.append({
                "courtesy": mode,
                "feature": feat,
                "source_mean": s_mu,
                "target_mean": t_mu,
                "source_sd": s_sd,
                "target_sd": t_sd,
                "standardized_mean_difference": (t_mu - s_mu) / pooled,
                "source_n": int(len(s)),
                "target_n": int(len(t)),
            })
    shift = pd.DataFrame(rows)
    shift.to_csv(out / "feature_shift.csv", index=False)

    source_model = fit_diag_model(source)
    target_model = fit_diag_model(target)
    transfer = {
        "source_dataset": args.source,
        "target_dataset": args.target,
        "source_to_source_acc": predict_diag(source_model, source),
        "source_to_target_acc": predict_diag(source_model, target),
        "target_to_target_acc": predict_diag(target_model, target),
        "target_to_source_acc": predict_diag(target_model, source),
        "source_in_window_steps": int(len(source)),
        "target_in_window_steps": int(len(target)),
    }
    (out / "transfer_accuracy.json").write_text(json.dumps(transfer, indent=2), encoding="utf-8")
    print(shift.to_string(index=False))
    print(json.dumps(transfer, indent=2))


if __name__ == "__main__":
    main()
