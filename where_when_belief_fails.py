"""
Diagnose where and when belief fails in closed-loop evaluation.

The script focuses on belief_policy steps from an evaluation run and reports:
  - belief accuracy by urgency quartile
  - belief accuracy by exposure count within the interaction window
  - non_cooperative-only versions of both analyses
  - action timing / close-call context by exposure count
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BELIEF_COLUMNS = ["belief_cooperative", "belief_non_cooperative"]
LABELS = np.array(["cooperative", "non_cooperative"])


def bootstrap_mean_ci(values: pd.Series, n_boot: int = 2000, seed: int = 12345) -> tuple[float, float]:
    arr = values.astype(float).to_numpy()
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed + len(arr))
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, default="eval_final_v6_cleanobs")
    parser.add_argument("--output", type=str, default="where_when_cleanobs")
    return parser.parse_args()


def add_belief_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    belief = out[BELIEF_COLUMNS].to_numpy()
    pred_idx = np.argmax(belief, axis=1)
    true_idx = (out["hidden_courtesy"].to_numpy() == "non_cooperative").astype(int)
    out["predicted_courtesy"] = LABELS[pred_idx]
    out["belief_correct"] = pred_idx == true_idx
    out["true_class_belief"] = belief[np.arange(len(out)), true_idx]
    out["belief_confidence"] = np.max(belief, axis=1)
    return out


def interaction_exposure_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["episode_id", "timestep"]).copy()
    out["interaction_exposure_step"] = np.nan
    mask = out["in_interaction_window"].astype(bool)
    out.loc[mask, "interaction_exposure_step"] = (
        out.loc[mask].groupby("episode_id").cumcount() + 1
    )
    return out


def summarize(grouped: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
    out = grouped.agg(
        steps=("timestep", "count"),
        episodes=("episode_id", "nunique"),
        belief_accuracy=("belief_correct", "mean"),
        mean_true_class_belief=("true_class_belief", "mean"),
        mean_confidence=("belief_confidence", "mean"),
        close_call_rate=("close_call", "mean"),
        collision_step_rate=("collision", "mean"),
        mean_urgency=("observed_urgency", "mean"),
        mean_abs_relative_distance=("observed_abs_relative_distance", "mean"),
        mean_merge_speed=("observed_merge_speed", "mean"),
        mean_merge_acceleration=("observed_merge_acceleration", "mean"),
    ).reset_index()
    ci_rows = []
    for _, group in grouped:
        lo, hi = bootstrap_mean_ci(group["belief_correct"])
        ci_rows.append((lo, hi))
    out["belief_accuracy_ci95_low"] = [x[0] for x in ci_rows]
    out["belief_accuracy_ci95_high"] = [x[1] for x in ci_rows]
    return out


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = pd.read_csv(eval_dir / "policy_steps.csv")
    steps = steps[steps["policy_name"] == "belief_policy"].copy()
    steps = add_belief_fields(steps)
    steps = interaction_exposure_index(steps)
    in_window = steps[steps["in_interaction_window"].astype(bool)].copy()

    in_window["urgency_quartile"] = pd.qcut(
        in_window["observed_urgency"].rank(method="first"),
        4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
    )
    summarize(in_window.groupby("urgency_quartile", observed=False)).to_csv(
        output_dir / "belief_accuracy_by_urgency_quartile.csv",
        index=False,
    )
    summarize(
        in_window[in_window["hidden_courtesy"] == "non_cooperative"].groupby("urgency_quartile", observed=False)
    ).to_csv(output_dir / "belief_accuracy_by_urgency_quartile_non_cooperative.csv", index=False)

    bins = [0, 1, 2, 3, 5, 10, 20, 10_000]
    labels = ["1", "2", "3", "4-5", "6-10", "11-20", "21+"]
    in_window["exposure_bin"] = pd.cut(
        in_window["interaction_exposure_step"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    summarize(in_window.groupby("exposure_bin", observed=False)).to_csv(
        output_dir / "belief_accuracy_by_interaction_exposure.csv",
        index=False,
    )
    summarize(
        in_window[in_window["hidden_courtesy"] == "non_cooperative"].groupby("exposure_bin", observed=False)
    ).to_csv(output_dir / "belief_accuracy_by_interaction_exposure_non_cooperative.csv", index=False)

    critical = in_window[in_window["close_call"].astype(bool)].copy()
    if not critical.empty:
        summarize(critical.groupby("exposure_bin", observed=False)).to_csv(
            output_dir / "belief_accuracy_close_call_steps_by_exposure.csv",
            index=False,
        )

    first_exposure = in_window.sort_values(["episode_id", "interaction_exposure_step"]).groupby("episode_id").head(5)
    summarize(first_exposure.groupby("hidden_courtesy")).to_csv(
        output_dir / "first_five_interaction_steps_by_courtesy.csv",
        index=False,
    )

    print(f"Saved where/when belief diagnostics to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
