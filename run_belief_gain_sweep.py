"""
Belief-action gain sweep for HiddenCourtesyMerge-Sim.

This tests whether belief fails because the current heuristic action mapping is
too insensitive to belief certainty.  The sweep keeps the same simulator,
observation model, seeds, and policy structure, but varies the gain multiplying
|2 * P(cooperative) - 1| in the IDLE/SLOWER action table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import evaluate_policies as ev
from generate_hidden_courtesy_merge_dataset import COURTESY_TYPES, GeneratorConfig


GAINS = (0.05, 0.10, 0.185, 0.30, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=str, default="belief_gain_sweep")
    parser.add_argument("--gains", type=float, nargs="+", default=list(GAINS))
    parser.add_argument("--policies", nargs="+", default=["belief_policy", "rule_policy", "oracle_policy"])
    return parser.parse_args()


def make_gain_action(caution_gain: float):
    clear_gain = 0.125 * (caution_gain / 0.185)

    def choose_gain_action(
        belief: np.ndarray,
        ttc: float,
        front_gap: float,
        rng: np.random.Generator,
        merge_speed: float = np.nan,
        ego_speed: float = np.nan,
        relative_distance: float = np.nan,
    ) -> int:
        b = np.asarray(belief, dtype=float)
        p_coop = float(b[0])
        confidence = abs(2.0 * p_coop - 1.0)

        if not np.isnan(ttc) and ttc < 3.0:
            return int(rng.choice([1, 4], p=[0.02, 0.98]))
        if not np.isnan(front_gap) and front_gap < 8.0:
            return int(rng.choice([1, 4], p=[0.02, 0.98]))

        if not np.isnan(ttc) and 0.0 < ttc < 5.0:
            p_slower = float(np.clip(0.735 + caution_gain * confidence, 0.735, 0.98))
            return int(rng.choice([1, 4], p=[1.0 - p_slower, p_slower]))

        p_slower = float(np.clip(0.205 + clear_gain * confidence, 0.205, 0.55))
        return int(rng.choice([1, 4], p=[1.0 - p_slower, p_slower]))

    return choose_gain_action


def run_one_gain(gain: float, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    ev.choose_ego_action = make_gain_action(gain)
    ev.load_obs_model()

    rng = np.random.default_rng(args.seed)
    config = GeneratorConfig(episodes=args.episodes, max_steps=60, seed=args.seed, output=str(output_dir))
    courtesy_schedule = ev.make_courtesy_schedule(args.episodes, rng)
    traffic = rng.uniform(*config.traffic_density_range, size=args.episodes)
    noise = rng.uniform(*config.observation_noise_range, size=args.episodes)

    episode_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for policy_name in args.policies:
        for episode_id in range(args.episodes):
            seed = args.seed + episode_id
            episode, steps = ev.evaluate_episode(
                episode_id=episode_id,
                policy_name=policy_name,
                seed=seed,
                courtesy=courtesy_schedule[episode_id],
                traffic_density=float(traffic[episode_id]),
                observation_noise=float(noise[episode_id]),
                config=config,
                q_tables=None,
            )
            episode_rows.append(episode)
            step_rows.extend(steps)

    episodes = pd.DataFrame(episode_rows)
    steps = pd.DataFrame(step_rows)
    gain_dir = output_dir / f"gain_{gain:.3f}".replace(".", "p")
    gain_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(gain_dir / "policy_episodes.csv", index=False)
    steps.to_csv(gain_dir / "policy_steps.csv", index=False)

    summary = ev.summarize(episodes, ["policy_name"])
    summary_by_courtesy = ev.summarize(episodes, ["policy_name", "hidden_courtesy"])
    tests = ev.policy_significance_tests(episodes)
    tests_by_courtesy = ev.significance_tests_by_courtesy(episodes)
    summary.to_csv(gain_dir / "policy_summary.csv", index=False)
    summary_by_courtesy.to_csv(gain_dir / "policy_summary_by_courtesy.csv", index=False)
    (gain_dir / "policy_summary.json").write_text(
        json.dumps(
            {
                "gain": gain,
                "clear_gain": 0.125 * (gain / 0.185),
                "policies": args.policies,
                "episodes_per_policy": args.episodes,
                "significance_tests_pooled": tests,
                "significance_tests_by_courtesy": tests_by_courtesy,
                "summary": summary.to_dict(orient="records"),
                "summary_by_courtesy": summary_by_courtesy.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    row: Dict[str, Any] = {"gain": gain}
    for _, srow in summary.iterrows():
        prefix = str(srow["policy_name"])
        row[f"{prefix}_collision_rate"] = float(srow["collision_rate"])
        row[f"{prefix}_success_rate"] = float(srow["success_rate"])
        row[f"{prefix}_mean_reward"] = float(srow["mean_reward"])
        row[f"{prefix}_mean_min_ttc"] = float(srow["mean_min_ttc"])
    for key in [
        "belief_policy_vs_rule_policy__collision",
        "belief_policy_vs_rule_policy__total_reward",
        "oracle_policy_vs_rule_policy__collision",
        "oracle_policy_vs_rule_policy__total_reward",
    ]:
        result = tests.get(key, {})
        row[f"{key}_p_adjusted"] = result.get("p_adjusted")
        row[f"{key}_effect_size"] = result.get("effect_size")
        row[f"{key}_rate_or_mean_diff"] = result.get("rate_diff", result.get("mean_diff"))
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for gain in args.gains:
        print(f"[gain_sweep] running gain={gain}")
        rows.append(run_one_gain(float(gain), args, output_dir))

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "belief_gain_sweep_summary.csv", index=False)
    (output_dir / "table_belief_gain_sweep.tex").write_text(
        out.to_latex(index=False, float_format="%.4f"),
        encoding="utf-8",
    )
    print(f"Saved belief gain sweep to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
