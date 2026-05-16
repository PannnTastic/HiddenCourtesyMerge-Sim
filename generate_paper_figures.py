"""
Generate paper-facing figures for HiddenCourtesyMerge-Sim.

Inputs:
  - HiddenCourtesyMerge-Sim/steps_with_belief.csv
  - eval_final_v5/policy_summary.json

Outputs are written to paper_figures/ by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = [
    ("observed_urgency", "Urgency 1/TTC"),
    ("observed_abs_relative_distance", "Abs. relative distance (m)"),
    ("observed_relative_speed", "Relative speed (m/s)"),
    ("observed_merge_speed", "Merge speed (m/s)"),
    ("observed_merge_acceleration", "Merge acceleration (m/s^2)"),
]
COURTESY_ORDER = ["cooperative", "non_cooperative"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HiddenCourtesyMerge-Sim")
    parser.add_argument("--eval", type=str, default="eval_final_v5")
    parser.add_argument("--output", type=str, default="paper_figures")
    parser.add_argument("--sample-per-class", type=int, default=5000)
    return parser.parse_args()


def _sample_by_class(df: pd.DataFrame, n: int, seed: int = 123) -> pd.DataFrame:
    parts = []
    for courtesy in COURTESY_ORDER:
        sub = df[df["hidden_courtesy"] == courtesy]
        if len(sub) > n:
            sub = sub.sample(n=n, random_state=seed)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def feature_violin(dataset_dir: Path, output_dir: Path, sample_per_class: int) -> None:
    steps = pd.read_csv(dataset_dir / "steps_with_belief.csv")
    steps = steps[steps["in_interaction_window"].astype(bool)].copy()
    steps = _sample_by_class(steps, sample_per_class)

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(17, 4.5), sharex=False)
    colors = ["#4C78A8", "#E45756"]

    for ax, (col, title) in zip(axes, FEATURES):
        data = [
            steps.loc[steps["hidden_courtesy"] == courtesy, col].dropna().to_numpy()
            for courtesy in COURTESY_ORDER
        ]
        parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.72)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["coop", "non-coop"], rotation=20)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Observation Feature Distributions by Hidden Courtesy", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_violins_by_courtesy.png", dpi=180)
    plt.close(fig)


def _iter_tests(payload: Dict) -> Iterable[Dict[str, object]]:
    by_courtesy = payload.get("significance_tests_by_courtesy", {})
    for courtesy, tests in by_courtesy.items():
        for key, value in tests.items():
            metric = key.rsplit("__", 1)[-1]
            comparison = key[: -(len(metric) + 2)]
            effect = value.get("effect_size")
            if effect is None:
                continue
            yield {
                "courtesy": courtesy,
                "comparison": comparison,
                "metric": metric,
                "effect_size": float(effect),
                "p_adjusted": float(value.get("p_adjusted", np.nan)),
            }


def effect_size_heatmap(eval_dir: Path, output_dir: Path) -> None:
    with open(eval_dir / "policy_summary.json", encoding="utf-8") as f:
        payload = json.load(f)

    rows = pd.DataFrame(list(_iter_tests(payload)))
    if rows.empty:
        raise ValueError("No by-courtesy significance tests found in policy_summary.json")

    keep_metrics = ["total_reward", "collision", "close_call_rate", "min_ttc", "success"]
    rows = rows[rows["metric"].isin(keep_metrics)]
    rows["label"] = rows["courtesy"] + " | " + rows["comparison"]

    pivot = rows.pivot_table(index="label", columns="metric", values="effect_size", aggfunc="first")
    pivot = pivot.reindex(columns=[m for m in keep_metrics if m in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.34 * len(pivot))))
    values = pivot.to_numpy(dtype=float)
    finite = np.isfinite(values)
    vmax = float(np.nanmax(np.abs(values[finite]))) if finite.any() else 1.0
    vmax = max(vmax, 0.25)

    im = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title("By-Courtesy Paired Effect Sizes")
    fig.colorbar(im, ax=ax, label="Cohen d or Cohen h")
    fig.tight_layout()
    fig.savefig(output_dir / "by_courtesy_effect_size_heatmap.png", dpi=180)
    plt.close(fig)


def policy_summary_figure(eval_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(eval_dir / "policy_summary.csv")
    order = [p for p in ["random_policy", "rule_policy", "belief_policy", "oracle_policy"] if p in set(summary["policy_name"])]
    summary = summary.set_index("policy_name").loc[order].reset_index()

    specs = [
        ("collision_rate", "Collision rate", "lower"),
        ("mean_reward", "Mean reward", "higher"),
        ("mean_min_ttc", "Mean min TTC (s)", "higher"),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(13, 4))
    for ax, (col, title, _) in zip(axes, specs):
        y = summary[col].to_numpy()
        yerr = None
        low_col = f"{col}_ci95_low"
        high_col = f"{col}_ci95_high"
        if low_col in summary.columns and high_col in summary.columns:
            yerr = np.vstack([y - summary[low_col].to_numpy(), summary[high_col].to_numpy() - y])
        ax.bar(summary["policy_name"], y, color="#4C78A8", yerr=yerr, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "policy_comparison_main_metrics.png", dpi=180)
    plt.close(fig)


def belief_gain_sweep_figure(output_dir: Path, gain_dir: Path = Path("belief_gain_sweep")) -> None:
    summary_path = gain_dir / "belief_gain_sweep_summary.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for policy, label in [
        ("rule_policy", "rule"),
        ("belief_policy", "belief"),
        ("oracle_policy", "oracle"),
    ]:
        axes[0].plot(df["gain"], df[f"{policy}_collision_rate"], marker="o", label=label)
        axes[1].plot(df["gain"], df[f"{policy}_mean_reward"], marker="o", label=label)

    axes[0].set_title("Collision Rate vs Belief Gain")
    axes[0].set_xlabel("caution gain")
    axes[0].set_ylabel("collision rate")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Reward vs Belief Gain")
    axes[1].set_xlabel("caution gain")
    axes[1].set_ylabel("mean reward")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "belief_gain_sweep.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    eval_dir = Path(args.eval)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_violin(dataset_dir, output_dir, args.sample_per_class)
    effect_size_heatmap(eval_dir, output_dir)
    policy_summary_figure(eval_dir, output_dir)
    belief_gain_sweep_figure(output_dir)
    print(f"Saved paper figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
