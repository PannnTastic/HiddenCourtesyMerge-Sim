"""
Empirically verify merge-vehicle selection in highway-env merge-v0.

This script checks whether select_merge_vehicle() usually picks a vehicle on the
ramp or merge lane at episode reset. It writes per-episode diagnostics and fails
if the verified selection rate is below the requested threshold.

Example:
    python verify_merge_selection.py --episodes 100 --output merge_selection_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from generate_hidden_courtesy_merge_dataset import (
    compute_gaps,
    lane_index,
    make_env,
    merge_vehicle_priority,
    safe_get_xy_speed,
    select_merge_vehicle,
    vehicle_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--traffic-density", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="merge_selection_check")
    parser.add_argument(
        "--min-verified-rate",
        type=float,
        default=0.90,
        help="Minimum fraction with priority 0 ramp or priority 1 merge-lane selection.",
    )
    return parser.parse_args()


def lane_repr(vehicle: Any) -> str:
    lane = getattr(vehicle, "lane_index", None)
    return str(tuple(lane)) if isinstance(lane, tuple) else str(lane)


def inspect_episode(seed: int, traffic_density: float) -> Dict[str, Any]:
    env = make_env(seed, traffic_density)
    ego = getattr(env.unwrapped, "vehicle", None)
    selected = select_merge_vehicle(env)

    ego_x, ego_y, ego_speed = safe_get_xy_speed(ego)
    merge_x, merge_y, merge_speed = safe_get_xy_speed(selected) if selected else (np.nan, np.nan, np.nan)
    front_gap, rear_gap = compute_gaps(env, ego)
    priority = merge_vehicle_priority(selected) if selected is not None else 99

    row = {
        "seed": seed,
        "selected": selected is not None,
        "merge_vehicle_id": vehicle_id(selected),
        "merge_vehicle_class": type(selected).__name__ if selected is not None else "",
        "merge_lane_index_raw": lane_repr(selected),
        "merge_lane": lane_index(selected) if selected is not None else -1,
        "merge_vehicle_priority": priority,
        "verified_ramp_or_merge": priority in (0, 1),
        "ego_x": ego_x,
        "ego_y": ego_y,
        "ego_speed": ego_speed,
        "merge_x": merge_x,
        "merge_y": merge_y,
        "merge_speed": merge_speed,
        "relative_distance": merge_x - ego_x if not np.isnan(merge_x) else np.nan,
        "relative_speed": merge_speed - ego_speed if not np.isnan(merge_speed) else np.nan,
        "front_gap": front_gap,
        "rear_gap": rear_gap,
    }
    env.close()
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = [
        inspect_episode(args.seed + episode_id, args.traffic_density)
        for episode_id in range(args.episodes)
    ]
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(output_dir / "merge_selection_diagnostics.csv", index=False)

    verified_rate = float(diagnostics["verified_ramp_or_merge"].mean()) if len(diagnostics) else 0.0
    priority_distribution = diagnostics["merge_vehicle_priority"].value_counts().sort_index().to_dict()
    lane_distribution = diagnostics["merge_lane_index_raw"].value_counts().to_dict()
    class_distribution = diagnostics["merge_vehicle_class"].value_counts().to_dict()
    summary = {
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "traffic_density": float(args.traffic_density),
        "min_verified_rate": float(args.min_verified_rate),
        "verified_ramp_or_merge_rate": verified_rate,
        "passed": bool(verified_rate >= args.min_verified_rate),
        "priority_distribution": {str(k): int(v) for k, v in priority_distribution.items()},
        "lane_index_distribution": {str(k): int(v) for k, v in lane_distribution.items()},
        "vehicle_class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "priority_definition": {
            "0": "ramp lane from-node j/k",
            "1": "merge lane ('b', 'c', 2)",
            "2": "geometric fallback",
            "99": "no selected vehicle",
        },
    }
    (output_dir / "merge_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not summary["passed"]:
        raise SystemExit(
            f"merge selection verification failed: verified rate {verified_rate:.3f} "
            f"< required {args.min_verified_rate:.3f}"
        )


if __name__ == "__main__":
    main()
