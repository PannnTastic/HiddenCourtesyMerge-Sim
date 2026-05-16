"""
Run reviewer-risk analyses for HiddenCourtesyMerge-Sim.

Outputs:
  - review_analyses_cleanobs/corrected_power_analysis.csv
  - review_analyses_cleanobs/feature_correlation_pearson.csv
  - review_analyses_cleanobs/feature_correlation_spearman.csv
  - review_analyses_cleanobs/observation_feature_ablation.csv
  - review_analyses_cleanobs/lambda_ablation.csv
  - review_analyses_cleanobs/interaction_only_policy_summary.csv
  - review_analyses_cleanobs/interaction_only_significance_tests.json
  - review_analyses_cleanobs/tables/*.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from generate_hidden_courtesy_merge_dataset import COURTESY_TYPES
from evaluate_policies import policy_significance_tests, significance_tests_by_courtesy


FEATURE_MAP = {
    "urg": "observed_urgency",
    "ard": "observed_abs_relative_distance",
    "rs": "observed_relative_speed",
    "mvs": "observed_merge_speed",
    "mva": "observed_merge_acceleration",
}
FEATURE_SUBSETS = {
    "full": ("urg", "ard", "rs", "mvs", "mva"),
    "final_ard_mvs_mva": ("ard", "mvs", "mva"),
    "no_urg": ("ard", "rs", "mvs", "mva"),
    "no_ard": ("urg", "rs", "mvs", "mva"),
    "no_rs": ("urg", "ard", "mvs", "mva"),
    "no_mvs": ("urg", "ard", "rs", "mva"),
    "mvs_mva_only": ("mvs", "mva"),
    "rs_mva_only": ("rs", "mva"),
    "urg_ard_mva": ("urg", "ard", "mva"),
}
LAMBDA_VALUES = (0.00, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HiddenCourtesyMerge-Sim-cleanobs")
    parser.add_argument("--eval", type=str, default="eval_final_v6_cleanobs")
    parser.add_argument("--obs-model", type=str, default="observation_model.json")
    parser.add_argument("--output", type=str, default="review_analyses_cleanobs")
    return parser.parse_args()


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def z_for_cdf(p: float) -> float:
    # Acklam-style approximation is unnecessary here; binary search is stable.
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cohens_h(p1: float, p2: float) -> float:
    p1 = min(max(float(p1), 0.0), 1.0)
    p2 = min(max(float(p2), 0.0), 1.0)
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def two_prop_power_from_h(h: float, n_per_policy: int, alpha: float = 0.05) -> float:
    h_abs = abs(float(h))
    if h_abs < 1e-12:
        return alpha
    z_alpha = z_for_cdf(1.0 - alpha / 2.0)
    delta = h_abs * math.sqrt(n_per_policy / 2.0)
    return norm_cdf(-z_alpha - delta) + (1.0 - norm_cdf(z_alpha - delta))


def n_for_power_from_h(h: float, target_power: float, alpha: float = 0.05) -> float:
    h_abs = abs(float(h))
    if h_abs < 1e-12:
        return math.inf
    z_alpha = z_for_cdf(1.0 - alpha / 2.0)
    z_beta = z_for_cdf(target_power)
    return 2.0 * ((z_alpha + z_beta) / h_abs) ** 2


def corrected_power_analysis(eval_dir: Path, output_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(eval_dir / "policy_summary.csv")
    rows = []
    policies = sorted(summary["policy_name"].unique())
    for i, p1 in enumerate(policies):
        for p2 in policies[i + 1 :]:
            r1 = float(summary.loc[summary["policy_name"] == p1, "collision_rate"].iloc[0])
            r2 = float(summary.loc[summary["policy_name"] == p2, "collision_rate"].iloc[0])
            n = int(summary.loc[summary["policy_name"] == p1, "episodes"].iloc[0])
            h = cohens_h(r1, r2)
            rows.append(
                {
                    "pair": f"{p1}_vs_{p2}",
                    "p1_collision": r1,
                    "p2_collision": r2,
                    "cohens_h": h,
                    "abs_h": abs(h),
                    "n_achieved_per_policy": n,
                    "power_at_n": two_prop_power_from_h(h, n),
                    "n_per_policy_for_80pct_power": n_for_power_from_h(h, 0.80),
                    "n_per_policy_for_90pct_power": n_for_power_from_h(h, 0.90),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "corrected_power_analysis.csv", index=False)
    return out


def load_obs_model(path: Path) -> Dict[str, Dict[str, float]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {mode: {k: float(v) for k, v in raw[mode].items() if isinstance(v, (int, float))} for mode in COURTESY_TYPES}


def gaussian_log_pdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    z = (float(x) - float(mu)) / sigma
    return -0.5 * z * z - math.log(sigma)


def likelihood(row: pd.Series, obs_model: Dict[str, Dict[str, float]], features: Sequence[str]) -> np.ndarray:
    log_scores = np.zeros(len(COURTESY_TYPES), dtype=float)
    valid = False
    for feat in features:
        value = row.get(FEATURE_MAP[feat], np.nan)
        if pd.isna(value):
            continue
        valid = True
        for i, mode in enumerate(COURTESY_TYPES):
            log_scores[i] += gaussian_log_pdf(value, obs_model[mode][f"{feat}_mu"], obs_model[mode][f"{feat}_sigma"])
    if not valid:
        return np.ones(len(COURTESY_TYPES), dtype=float)
    log_scores -= float(np.max(log_scores))
    return np.maximum(np.exp(log_scores), 1e-9)


def recompute_beliefs(
    steps: pd.DataFrame,
    obs_model: Dict[str, Dict[str, float]],
    features: Sequence[str],
    lambda_reg: float,
) -> pd.DataFrame:
    rows = []
    labels = list(COURTESY_TYPES)
    for _, group in steps.sort_values(["episode_id", "timestep"]).groupby("episode_id", sort=False):
        belief = np.ones(len(COURTESY_TYPES), dtype=float) / len(COURTESY_TYPES)
        for _, row in group.iterrows():
            if bool(row.get("in_interaction_window", False)):
                post = belief * likelihood(row, obs_model, features)
                post = post / max(float(np.sum(post)), 1e-12)
                uniform = np.ones_like(post) / len(post)
                belief = (1.0 - lambda_reg) * post + lambda_reg * uniform
                belief = belief / float(np.sum(belief))
            true_idx = labels.index(str(row["hidden_courtesy"]))
            rows.append(
                {
                    "episode_id": int(row["episode_id"]),
                    "timestep": int(row["timestep"]),
                    "hidden_courtesy": row["hidden_courtesy"],
                    "in_interaction_window": bool(row.get("in_interaction_window", False)),
                    "belief_cooperative": belief[0],
                    "belief_non_cooperative": belief[1],
                    "true_idx": true_idx,
                    "true_prob": belief[true_idx],
                    "predicted": labels[int(np.argmax(belief))],
                }
            )
    return pd.DataFrame(rows)


def belief_metrics(beliefs: pd.DataFrame) -> Dict[str, float]:
    labels = np.array(COURTESY_TYPES)
    sub = beliefs[beliefs["in_interaction_window"].astype(bool)]
    if sub.empty:
        sub = beliefs
    b = sub[["belief_cooperative", "belief_non_cooperative"]].to_numpy()
    true = sub["hidden_courtesy"].to_numpy()
    true_idx = sub["true_idx"].to_numpy(dtype=int)
    one_hot = np.eye(len(COURTESY_TYPES))[true_idx]
    final = beliefs.sort_values(["episode_id", "timestep"]).groupby("episode_id").tail(1)
    final_b = final[["belief_cooperative", "belief_non_cooperative"]].to_numpy()
    final_pred = labels[np.argmax(final_b, axis=1)]
    conf = np.max(b, axis=1)
    pred = np.argmax(b, axis=1)
    correct = pred == true_idx
    ece = 0.0
    for lo in np.linspace(0.5, 0.9, 5):
        hi = lo + 0.1
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(conf[mask])) - float(np.mean(correct[mask])))
    return {
        "belief_accuracy": float(np.mean(labels[np.argmax(b, axis=1)] == true)),
        "final_belief_accuracy": float(np.mean(final_pred == final["hidden_courtesy"].to_numpy())),
        "brier_score": float(np.mean(np.sum((b - one_hot) ** 2, axis=1))),
        "mean_nll": float(-np.mean(np.log(np.maximum(sub["true_prob"].to_numpy(dtype=float), 1e-9)))),
        "ece": ece,
        "n_steps_in_window": int(len(sub)),
        "mean_final_belief_non_cooperative": float(final["belief_non_cooperative"].mean()),
    }


def feature_correlation(steps: pd.DataFrame, output_dir: Path) -> None:
    sub = steps[steps["in_interaction_window"].astype(bool)]
    cols = [FEATURE_MAP[f] for f in FEATURE_MAP]
    renamed = sub[cols].rename(columns={v: k for k, v in FEATURE_MAP.items()})
    renamed.corr(method="pearson").to_csv(output_dir / "feature_correlation_pearson.csv")
    renamed.corr(method="spearman").to_csv(output_dir / "feature_correlation_spearman.csv")


def observation_feature_ablation(steps: pd.DataFrame, obs_model: Dict[str, Dict[str, float]], output_dir: Path) -> pd.DataFrame:
    rows = []
    for name, features in FEATURE_SUBSETS.items():
        beliefs = recompute_beliefs(steps, obs_model, features, lambda_reg=0.08)
        rows.append({"model": name, "features": ",".join(features), **belief_metrics(beliefs)})
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "observation_feature_ablation.csv", index=False)
    return out


def lambda_ablation(steps: pd.DataFrame, obs_model: Dict[str, Dict[str, float]], output_dir: Path) -> pd.DataFrame:
    rows = []
    features = FEATURE_SUBSETS["final_ard_mvs_mva"]
    for value in LAMBDA_VALUES:
        beliefs = recompute_beliefs(steps, obs_model, features, lambda_reg=value)
        rows.append({"lambda": value, **belief_metrics(beliefs)})
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "lambda_ablation.csv", index=False)
    return out


def interaction_only_analysis(eval_dir: Path, output_dir: Path) -> None:
    episodes = pd.read_csv(eval_dir / "policy_episodes.csv")
    steps = pd.read_csv(eval_dir / "policy_steps.csv")
    active_ids = set(
        steps.loc[steps["in_interaction_window"].astype(bool), "episode_id"].astype(int).unique().tolist()
    )
    filtered = episodes[episodes["episode_id"].astype(int).isin(active_ids)].copy()
    summary = (
        filtered.groupby("policy_name")
        .agg(
            episodes=("episode_id", "nunique"),
            collision_rate=("collision", "mean"),
            success_rate=("success", "mean"),
            close_call_rate=("close_call_rate", "mean"),
            mean_reward=("total_reward", "mean"),
            mean_min_ttc=("min_ttc", "mean"),
            mean_episode_length=("episode_length", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "interaction_only_policy_summary.csv", index=False)
    tests = {
        "definition": "episode_id has at least one in_interaction_window step in any evaluated policy",
        "n_episode_ids": len(active_ids),
        "significance_tests_pooled": policy_significance_tests(filtered),
        "significance_tests_by_courtesy": significance_tests_by_courtesy(filtered),
    }
    (output_dir / "interaction_only_significance_tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")


def write_tex_tables(output_dir: Path, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    for stem in [
        "corrected_power_analysis",
        "observation_feature_ablation",
        "lambda_ablation",
        "interaction_only_policy_summary",
    ]:
        path = output_dir / f"{stem}.csv"
        if path.exists():
            df = pd.read_csv(path)
            (tables_dir / f"table_{stem}.tex").write_text(
                df.to_latex(index=False, float_format="%.4f"),
                encoding="utf-8",
            )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    eval_dir = Path(args.eval)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = pd.read_csv(dataset_dir / "steps_with_belief.csv")
    obs_model = load_obs_model(Path(args.obs_model))

    corrected_power_analysis(eval_dir, output_dir)
    feature_correlation(steps, output_dir)
    observation_feature_ablation(steps, obs_model, output_dir)
    lambda_ablation(steps, obs_model, output_dir)
    interaction_only_analysis(eval_dir, output_dir)
    write_tex_tables(output_dir, output_dir / "tables")

    print(f"Saved review analyses to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
