from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 1.9,
        "lines.markersize": 5,
    }
)

COLORS = {
    "pomcp_policy": "#E76F51",
    "oracle_policy": "#0072B2",
    "belief_policy": "#2A9D8F",
    "rule_policy": "#E9C46A",
    "random_policy": "#8C8C8C",
}
LABELS = {
    "pomcp_policy": "POMCP",
    "oracle_policy": "Oracle",
    "belief_policy": "Belief",
    "rule_policy": "Rule",
    "random_policy": "Random",
}
ORDER = ["pomcp_policy", "oracle_policy", "belief_policy", "rule_policy", "random_policy"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", "nan", "NaN") else float("nan")


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def load_policy_summary() -> dict[str, dict[str, str]]:
    base = read_csv(ROOT / "eval_final_v6_cleanobs" / "policy_summary.csv")
    pomcp = read_csv(ROOT / "pomcp_300ep_100sims" / "policy_summary.csv")
    rows = {row["policy_name"]: row for row in base + pomcp}
    missing = [p for p in ORDER if p not in rows]
    if missing:
        raise RuntimeError(f"Missing policy rows: {missing}")
    return rows


def plot_policy_comparison() -> None:
    rows = load_policy_summary()
    metrics = [
        (
            "collision_rate",
            "Collision rate (%)",
            "Lower is better",
            "collision_rate_ci95_low",
            "collision_rate_ci95_high",
            100.0,
            (0, 95),
        ),
        (
            "mean_reward",
            "Mean reward",
            "Higher is better",
            "mean_reward_ci95_low",
            "mean_reward_ci95_high",
            1.0,
            (34, 59),
        ),
        (
            "mean_min_ttc",
            "Mean min TTC (s)",
            "Higher is better",
            "mean_min_ttc_ci95_low",
            "mean_min_ttc_ci95_high",
            1.0,
            (1.5, 3.4),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    x = np.arange(len(ORDER))
    for ax, (metric, ylabel, title, low_key, high_key, scale, ylim) in zip(axes, metrics):
        means = np.array([as_float(rows[p], metric) * scale for p in ORDER])
        lows = np.array([as_float(rows[p], low_key) * scale for p in ORDER])
        highs = np.array([as_float(rows[p], high_key) * scale for p in ORDER])
        yerr = np.vstack([means - lows, highs - means])
        bars = ax.bar(
            x,
            means,
            yerr=yerr,
            capsize=2.5,
            color=[COLORS[p] for p in ORDER],
            edgecolor="white",
            linewidth=0.6,
            width=0.74,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
        )
        for bar, value in zip(bars, means):
            label = f"{value:.1f}" if metric != "collision_rate" else f"{value:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] - ylim[0]) * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#333333",
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[p] for p in ORDER], rotation=35, ha="right")
        ax.set_ylim(*ylim)
        ax.set_axisbelow(True)

    fig.suptitle("Matched policy evaluation (n=300 episodes per policy)", y=1.04, fontsize=10.5, fontweight="bold")
    save(fig, "policy_comparison_main_metrics")


def plot_belief_convergence() -> None:
    rows = read_csv(ROOT / "where_when_cleanobs" / "belief_accuracy_by_interaction_exposure_non_cooperative.csv")
    labels = [row["exposure_bin"] for row in rows]
    x = np.arange(len(labels))
    acc = np.array([as_float(row, "belief_accuracy") * 100.0 for row in rows])
    lo = np.array([as_float(row, "belief_accuracy_ci95_low") * 100.0 for row in rows])
    hi = np.array([as_float(row, "belief_accuracy_ci95_high") * 100.0 for row in rows])

    fig, ax = plt.subplots(figsize=(4.9, 2.65), constrained_layout=True)
    ax.fill_between(x, lo, hi, color="#2A9D8F", alpha=0.18, linewidth=0)
    ax.plot(x, acc, color="#2A9D8F", marker="o", label="Belief accuracy")
    ax.axhline(50, color="#8C8C8C", linewidth=1, linestyle="--", label="Chance")
    ax.axvspan(-0.35, 4.35, color="#E76F51", alpha=0.08, linewidth=0)
    ax.text(2.0, 54.5, "early window", ha="center", va="center", fontsize=8, color="#9A3D2D")
    for xi, yi in zip(x, acc):
        ax.text(xi, yi + 2.0, f"{yi:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Interaction-window exposure step")
    ax.set_ylabel("Belief accuracy (%)")
    ax.set_ylim(45, 98)
    ax.set_title("Non-cooperative belief convergence")
    ax.legend(loc="lower right")
    save(fig, "belief_convergence")


def plot_gain_sweep() -> None:
    rows = read_csv(ROOT / "belief_gain_sweep" / "belief_gain_sweep_summary.csv")
    gains = np.array([as_float(row, "gain") for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55), constrained_layout=True)
    series = [
        ("pomcp_policy", None),
        ("oracle_policy", "o"),
        ("belief_policy", "s"),
        ("rule_policy", "^"),
    ]

    for ax, metric, ylabel, scale in [
        (axes[0], "collision_rate", "Collision rate (%)", 100.0),
        (axes[1], "mean_reward", "Mean reward", 1.0),
    ]:
        for policy, marker in series:
            if policy == "pomcp_policy":
                continue
            values = np.array([as_float(row, f"{policy}_{metric}") * scale for row in rows])
            ax.plot(gains, values, marker=marker, color=COLORS[policy], label=LABELS[policy])
        ax.set_xlabel("Belief gain g")
        ax.set_ylabel(ylabel)
        ax.set_xscale("linear")
        ax.set_xticks(gains)
        ax.set_xticklabels(["0.05", "0.10", "0.185", "0.30", "0.50"], rotation=25, ha="right")
        ax.set_axisbelow(True)

    axes[0].set_title("Collision")
    axes[0].set_ylim(3, 21)
    axes[1].set_title("Reward")
    axes[1].set_ylim(54.2, 57.7)
    axes[0].legend(loc="upper right")
    fig.suptitle("Belief-gain sweep: oracle responds, belief remains flat", y=1.04, fontsize=10.5, fontweight="bold")
    save(fig, "belief_gain_sweep")


def main() -> None:
    plot_policy_comparison()
    plot_belief_convergence()
    plot_gain_sweep()
    print("Generated publication figures in", FIG_DIR)


if __name__ == "__main__":
    main()
