"""
Calibrate the Gaussian observation model Z(o | courtesy) from simulator rollouts.

For each courtesy mode it runs N_CAL episodes with zero observation noise, the ego
driven by the same conservative (uniform-belief) heuristic the dataset's rule_policy
uses, and collects the observable merge-vehicle features

    urg = 1 / TTC   (urgency; 0 when not closing)
    ard = |merge_x - ego_x|
    rs  = merge_speed - ego_speed
    mvs = merge vehicle absolute speed
    mva = merge vehicle acceleration

*only on steps inside the directional interaction window* (merge vehicle ahead
and within _INTERACTION_RANGE_M longitudinally) — the only steps where courtesy
actually shows up — then fits a Gaussian (mean, std) per (mode, feature) and writes
observation_model.json.

Run this BEFORE generate_hidden_courtesy_merge_dataset.py so the main scripts pick
up the calibrated model automatically.

Example:
    python calibrate_observation_model.py --episodes 60 --output observation_model.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from generate_hidden_courtesy_merge_dataset import (
    COURTESY_TYPES,
    _POLICY_DT,
    choose_ego_action,
    compute_gaps,
    configure_courtesy_vehicle,
    estimate_ttc,
    in_interaction_window,
    make_env,
    safe_get_xy_speed,
    select_merge_vehicle,
    ttc_to_urgency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=60,
                        help="Episodes per courtesy mode for calibration.")
    parser.add_argument("--seed",     type=int, default=999)
    parser.add_argument("--output",   type=str, default="observation_model.json")
    parser.add_argument("--traffic-density", type=float, default=1.0)
    return parser.parse_args()


_UNIFORM_BELIEF = np.ones(len(COURTESY_TYPES), dtype=float) / len(COURTESY_TYPES)


def _collect_features(
    courtesy: str,
    n_episodes: int,
    seed: int,
    traffic_density: float,
) -> Dict[str, List[float]]:
    """Run rollouts and collect merge-vehicle features on interaction-window steps."""
    urg, ard, rs, mvs, mva_list = [], [], [], [], []
    rows: List[Dict[str, float]] = []
    rng_global = np.random.default_rng(seed)

    for ep in range(n_episodes):
        ep_seed = seed + ep
        env = make_env(ep_seed, traffic_density)  # no obs noise
        mv = select_merge_vehicle(env)
        configure_courtesy_vehicle(mv, courtesy, np.random.default_rng(ep_seed + 13))
        prev_mv_speed: Optional[float] = None  # reset per episode

        for _ in range(60):
            ego = getattr(env.unwrapped, "vehicle", None)
            if ego is None:
                break
            ego_x, _, ego_speed = safe_get_xy_speed(ego)
            mv_x, _, mv_speed = safe_get_xy_speed(mv) if mv else (np.nan, np.nan, np.nan)

            # Merge vehicle acceleration (finite difference; NaN on first step)
            if prev_mv_speed is not None and not math.isnan(mv_speed) and not math.isnan(prev_mv_speed):
                mv_acc = (mv_speed - prev_mv_speed) / _POLICY_DT
            else:
                mv_acc = float("nan")
            if not math.isnan(mv_speed):
                prev_mv_speed = mv_speed

            rel_dist = mv_x - ego_x if not math.isnan(mv_x) else np.nan
            rel_spd  = mv_speed - ego_speed if not math.isnan(mv_speed) else np.nan
            ttc_val  = estimate_ttc(rel_dist, rel_spd)
            fg, _    = compute_gaps(env, ego)

            if in_interaction_window(rel_dist, mv_speed):
                u = ttc_to_urgency(ttc_val, rel_dist, rel_spd)
                abs_rel_dist = abs(float(rel_dist)) if not math.isnan(rel_dist) else float("nan")
                if not math.isnan(u):
                    urg.append(float(np.clip(u, 0.0, 1.5)))
                if not math.isnan(abs_rel_dist):
                    ard.append(float(np.clip(abs_rel_dist, 0.0, 40.0)))
                if not math.isnan(rel_spd):
                    rs.append(float(np.clip(rel_spd, -25.0, 5.0)))
                if not math.isnan(mv_speed):
                    mvs.append(float(np.clip(mv_speed, 0.0, 35.0)))
                if not math.isnan(mv_acc):
                    mva_list.append(float(np.clip(mv_acc, -8.0, 7.0)))
                rows.append({
                    "hidden_courtesy": courtesy,
                    "urg": float(np.clip(u, 0.0, 1.5)) if not math.isnan(u) else float("nan"),
                    "ard": float(np.clip(abs_rel_dist, 0.0, 40.0)) if not math.isnan(abs_rel_dist) else float("nan"),
                    "rs": float(np.clip(rel_spd, -25.0, 5.0)) if not math.isnan(rel_spd) else float("nan"),
                    "mvs": float(np.clip(mv_speed, 0.0, 35.0)) if not math.isnan(mv_speed) else float("nan"),
                    "mva": float(np.clip(mv_acc, -8.0, 7.0)) if not math.isnan(mv_acc) else float("nan"),
                })

            # Drive the ego with the same conservative heuristic the dataset uses
            # (uniform belief == rule_policy) so the interaction looks representative.
            action = choose_ego_action(
                _UNIFORM_BELIEF, ttc_val, fg, rng_global,
                merge_speed=mv_speed, ego_speed=ego_speed, relative_distance=rel_dist,
            )
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        env.close()

    return {"urg": urg, "ard": ard, "rs": rs, "mvs": mvs, "mva": mva_list, "_rows": rows}


def _fit_gaussian(values: List[float], min_sigma: float) -> Dict[str, float]:
    if len(values) < 2:
        return {"mu": float(np.nanmean(values) if values else 0.0), "sigma": min_sigma}
    mu    = float(np.nanmean(values))
    sigma = max(float(np.nanstd(values, ddof=1)), min_sigma)
    return {"mu": mu, "sigma": sigma}


_MIN_SIGMA = {"urg": 0.05, "ard": 2.0, "rs": 1.0, "mvs": 1.0, "mva": 0.5}
ACTIVE_FEATURES = ("ard", "mvs", "mva")


def calibrate(n_episodes: int, seed: int, traffic_density: float) -> Dict[str, Any]:
    model: Dict[str, Any] = {}
    all_rows: List[Dict[str, float]] = []
    for courtesy in COURTESY_TYPES:
        print(f"Collecting rollouts for '{courtesy}' ... ", end="", flush=True)
        feats = _collect_features(courtesy, n_episodes, seed, traffic_density)
        all_rows.extend(feats["_rows"])
        print(f"{len(feats['urg'])} in-window observations.")

        urg_fit = _fit_gaussian(feats["urg"], _MIN_SIGMA["urg"])
        ard_fit = _fit_gaussian(feats["ard"], _MIN_SIGMA["ard"])
        rs_fit  = _fit_gaussian(feats["rs"],  _MIN_SIGMA["rs"])
        mvs_fit = _fit_gaussian(feats["mvs"], _MIN_SIGMA["mvs"])
        mva_fit = _fit_gaussian(feats["mva"], _MIN_SIGMA["mva"])

        model[courtesy] = {
            "urg_mu": urg_fit["mu"], "urg_sigma": urg_fit["sigma"],
            "ard_mu": ard_fit["mu"], "ard_sigma": ard_fit["sigma"],
            "rs_mu":  rs_fit["mu"],  "rs_sigma":  rs_fit["sigma"],
            "mvs_mu": mvs_fit["mu"], "mvs_sigma": mvs_fit["sigma"],
            "mva_mu": mva_fit["mu"], "mva_sigma": mva_fit["sigma"],
            "_n_urg": len(feats["urg"]), "_n_ard": len(feats["ard"]), "_n_rs": len(feats["rs"]),
            "_n_mvs": len(feats["mvs"]), "_n_mva": len(feats["mva"]),
        }
        print(f"  urg  ~ N({urg_fit['mu']:.3f}, {urg_fit['sigma']:.3f})")
        print(f"  ard  ~ N({ard_fit['mu']:.2f}, {ard_fit['sigma']:.2f})")
        print(f"  rs   ~ N({rs_fit['mu']:.2f}, {rs_fit['sigma']:.2f})")
        print(f"  mvs  ~ N({mvs_fit['mu']:.2f}, {mvs_fit['sigma']:.2f})")
        print(f"  mva  ~ N({mva_fit['mu']:.3f}, {mva_fit['sigma']:.3f})")

    separation = _report_separation(model)
    model["_validation"] = _validate_model(model, all_rows)
    model["_validation"]["feature_separation"] = separation
    return model


def _report_separation(model: Dict[str, Any]) -> Dict[str, float]:
    """Print, per feature, the max |delta-mu| between modes in units of mean sigma."""
    print("\nMode separation (|delta-mu| / mean sigma, larger is better; >= 0.5 is usable):")
    out: Dict[str, float] = {}
    for feat in ACTIVE_FEATURES:
        mus    = [model[m][f"{feat}_mu"]    for m in COURTESY_TYPES]
        sigmas = [model[m][f"{feat}_sigma"] for m in COURTESY_TYPES]
        spread = (max(mus) - min(mus)) / max(float(np.mean(sigmas)), 1e-9)
        flag = "OK" if spread >= 0.5 else "WEAK"
        out[feat] = float(spread)
        print(f"  {feat:4s}: {spread:.2f}  [{flag}]  "
              f"mus={[round(x, 3) for x in mus]}  sigmas={[round(x, 3) for x in sigmas]}")
    return out


def _log_pdf(x: float, mu: float, sigma: float) -> float:
    return -0.5 * ((x - mu) / max(sigma, 1e-6)) ** 2 - math.log(max(sigma, 1e-6))


def _validate_model(model: Dict[str, Any], rows: List[Dict[str, float]]) -> Dict[str, Any]:
    if not rows:
        return {}
    probs, true_idx = [], []
    for row in rows:
        log_scores = []
        for mode in COURTESY_TYPES:
            score = 0.0
            for feat in ACTIVE_FEATURES:
                value = row.get(feat, float("nan"))
                if math.isnan(value):
                    continue
                score += _log_pdf(value, model[mode][f"{feat}_mu"], model[mode][f"{feat}_sigma"])
            log_scores.append(score)
        arr = np.array(log_scores, dtype=float)
        arr -= arr.max()
        p = np.exp(arr)
        p = p / p.sum()
        probs.append(p)
        true_idx.append(list(COURTESY_TYPES).index(str(row["hidden_courtesy"])))

    prob_arr = np.vstack(probs)
    true = np.array(true_idx)
    pred = prob_arr.argmax(axis=1)
    one_hot = np.eye(len(COURTESY_TYPES))[true]
    true_probs = prob_arr[np.arange(len(prob_arr)), true]
    confidences = prob_arr.max(axis=1)
    correct = pred == true
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    bins = []
    for i in range(10):
        mask = (confidences >= edges[i]) & (confidences <= edges[i + 1] if i == 9 else confidences < edges[i + 1])
        count = int(mask.sum())
        if count:
            conf = float(confidences[mask].mean())
            acc = float(correct[mask].mean())
            ece += count / len(confidences) * abs(conf - acc)
        else:
            conf = acc = None
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "confidence": conf, "accuracy": acc, "count": count})
    return {
        "binary_belief_accuracy": float(correct.mean()),
        "brier_score": float(np.mean(np.sum((prob_arr - one_hot) ** 2, axis=1))),
        "mean_nll": float(-np.mean(np.log(np.maximum(true_probs, 1e-9)))),
        "uniform_nll": float(math.log(len(COURTESY_TYPES))),
        "ece": float(ece),
        "reliability_bins": bins,
        "n_validation_steps": int(len(rows)),
    }


def main() -> None:
    args = parse_args()
    print(
        f"Calibrating observation model ({args.episodes} episodes/mode, "
        f"seed={args.seed}, traffic_density={args.traffic_density})"
    )
    model = calibrate(args.episodes, args.seed, args.traffic_density)
    model["_meta"] = {
        "episodes_per_mode": int(args.episodes),
        "seed": int(args.seed),
        "traffic_density": float(args.traffic_density),
        "class_names": list(COURTESY_TYPES),
        "features": list(ACTIVE_FEATURES),
        "dropped_features": {
            "urg": "weak separation and worsened calibration in ablation",
            "rs": "near-duplicate of mvs in the final dataset (Pearson r≈0.991)",
        },
        "interaction_window_relative_distance_m": {
            "min": 2.0,
            "max": 40.0,
            "direction": "merge_vehicle_ahead_only",
        },
        "highway_env_version": getattr(__import__("highway_env"), "__version__", "unknown"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"\nSaved calibrated observation model to {out_path.resolve()}")
    print("Run generate_hidden_courtesy_merge_dataset.py and evaluate_policies.py "
          "from the same directory to pick up these parameters automatically.")


if __name__ == "__main__":
    main()
