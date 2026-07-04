"""Run high-powered matched evaluations for cheap heuristic policies."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1960)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=str, default="eval_confirmatory_1960")
    parser.add_argument("--belief-model", choices=("diagonal", "full_cov"), default="diagonal")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["belief_policy", "rule_policy", "oracle_policy", "random_policy"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "evaluate_policies.py",
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--output",
        args.output,
        "--belief-model",
        args.belief_model,
        "--policies",
        *args.policies,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
