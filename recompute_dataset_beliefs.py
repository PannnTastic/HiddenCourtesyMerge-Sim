"""
Recompute belief columns for an existing generated dataset without rerunning simulation.

This is used after changing the active observation likelihood feature set.  It
keeps episodes.csv and steps.csv fixed, recomputes only steps_with_belief.csv,
copies the active observation_model.json, and updates config.json belief
validation metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from generate_hidden_courtesy_merge_dataset import (
    COURTESY_TYPES,
    _BELIEF_COLUMNS,
    _OBS_FEATURES,
    belief_quality_over_time,
    load_obs_model,
    compute_belief_metrics,
    update_belief,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="HiddenCourtesyMerge-Sim")
    parser.add_argument("--output", type=str, default="HiddenCourtesyMerge-Sim-cleanobs")
    parser.add_argument("--obs-model", type=str, default="observation_model.json")
    parser.add_argument("--regularization", type=float, default=0.08)
    return parser.parse_args()


def recompute_beliefs(steps: pd.DataFrame, regularization: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, group in steps.sort_values(["episode_id", "timestep"]).groupby("episode_id", sort=False):
        belief = np.ones(len(COURTESY_TYPES), dtype=float) / len(COURTESY_TYPES)
        for _, row in group.iterrows():
            if bool(row.get("in_interaction_window", False)):
                belief = update_belief(
                    belief,
                    float(row.get("observed_urgency", np.nan)),
                    float(row.get("observed_abs_relative_distance", np.nan)),
                    float(row.get("observed_relative_speed", np.nan)),
                    float(row.get("observed_merge_speed", np.nan)),
                    float(row.get("observed_merge_acceleration", np.nan)),
                    regularization=regularization,
                )
            out = row.to_dict()
            out["belief_cooperative"] = float(belief[0])
            out["belief_non_cooperative"] = float(belief[1])
            rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    load_obs_model(args.obs_model)

    episodes = pd.read_csv(src / "episodes.csv")
    steps = pd.read_csv(src / "steps.csv")
    old_beliefs = pd.read_csv(src / "steps_with_belief.csv")
    new_beliefs = recompute_beliefs(old_beliefs, args.regularization)

    episodes.to_csv(dst / "episodes.csv", index=False)
    steps.to_csv(dst / "steps.csv", index=False)
    new_beliefs.to_csv(dst / "steps_with_belief.csv", index=False)
    shutil.copyfile(args.obs_model, dst / "observation_model.json")
    if (src / "README.md").exists():
        shutil.copyfile(src / "README.md", dst / "README.md")
    if (src / "figures").exists():
        if (dst / "figures").exists():
            shutil.rmtree(dst / "figures")
        shutil.copytree(src / "figures", dst / "figures")

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    config["observation_features"] = [name for name, _, _ in _OBS_FEATURES]
    config["belief_recomputed_from"] = str(src)
    config["belief_regularization"] = float(args.regularization)
    config["validation_summary"]["belief_validation"] = compute_belief_metrics(new_beliefs)
    config["validation_summary"]["missing_value_check"]["steps_with_belief"] = (
        new_beliefs.isna().sum().astype(int).to_dict()
    )
    (dst / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    quality = belief_quality_over_time(new_beliefs)
    if not quality.empty:
        quality.to_csv(dst / "belief_quality_over_time.csv", index=False)

    print(f"Recomputed clean-observation beliefs at {dst.resolve()}")
    print(json.dumps(config["validation_summary"]["belief_validation"], indent=2))


if __name__ == "__main__":
    main()
