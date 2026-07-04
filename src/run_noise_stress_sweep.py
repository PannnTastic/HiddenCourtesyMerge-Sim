"""Run observation-quality stress tests for policy evaluation.

This script keeps the matched episode sequence fixed and varies only the
observation-noise multiplier. It is intended to test whether online planning is
more robust than belief-conditioned heuristic control under degraded
observation quality.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=str, default="noise_stress_sweep")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["rule_policy", "belief_policy", "oracle_policy", "pomcp_policy"],
    )
    parser.add_argument("--pomcp-sims", type=int, default=100)
    parser.add_argument("--pomcp-horizon", type=int, default=10)
    parser.add_argument("--belief-model", choices=("diagonal", "full_cov"), default="diagonal")
    return parser.parse_args()


def scale_tag(scale: float) -> str:
    return f"{int(round(scale * 100)):03d}"


def main() -> None:
    args = parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for scale in args.scales:
        run_dir = root / f"noise_{scale_tag(scale)}"
        cmd = [
            sys.executable,
            "evaluate_policies.py",
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--output",
            str(run_dir),
            "--belief-model",
            args.belief_model,
            "--observation-noise-mode",
            "scaled",
            "--observation-noise-scale",
            str(scale),
            "--policies",
            *args.policies,
        ]
        if "pomcp_policy" in args.policies:
            cmd.extend(["--pomcp-sims", str(args.pomcp_sims), "--pomcp-horizon", str(args.pomcp_horizon)])
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        summary = pd.read_csv(run_dir / "policy_summary.csv")
        summary["noise_scale"] = scale
        rows.append(summary)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(root / "noise_stress_summary.csv", index=False)
    metadata = {
        "episodes": args.episodes,
        "seed": args.seed,
        "scales": args.scales,
        "policies": args.policies,
        "belief_model": args.belief_model,
    }
    (root / "noise_stress_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
