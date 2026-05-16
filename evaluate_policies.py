"""
Evaluate baseline policies for HiddenCourtesyMerge-Sim.

Policies
--------
belief_policy : Bayesian belief over hidden courtesy → heuristic softmax action.
rule_policy   : Same action table as belief_policy but belief frozen at uniform prior.
                Isolates the marginal value of belief over a fixed heuristic.
random_policy : Uniform random action. Sanity floor.
oracle_policy : One-hot belief at true courtesy label. Upper bound for this action table.
qmdp_policy   : Optional exploratory QMDP approximation — a* = argmax_a sum_m b(m)*Q_m*(s,a), where Q_m* is
                obtained via tabular Q-learning in the fully observable MDP for mode m.
                Cached to qmdp_qtable.json after first solve. Not included in the default
                evaluation suite because the coarse 16-state abstraction is not a reliable
                main baseline.

Fairness
--------
For each episode index all policies share the same env seed, courtesy, traffic density,
and observation noise. Action sampling uses a separate per-policy RNG offset so it does
not advance the exogenous scenario stream.

Statistics
----------
* Bootstrap 95% CI for binary metrics (collision, success).
* Normal-approximation 95% CI for continuous metrics (n > 30).
* McNemar test for paired binary outcomes (collision, success).
* Paired t-test for continuous outcomes (reward, min_ttc).
* Holm–Bonferroni correction applied across all pairwise tests.
* Negotiation metric: action distribution conditional on (policy, true_courtesy).

Example
-------
    python evaluate_policies.py --episodes 300 --output policy_eval
    python evaluate_policies.py --episodes 300 --output policy_eval_qmdp --policies belief_policy rule_policy random_policy oracle_policy qmdp_policy oracle_qmdp
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_hidden_courtesy_merge_dataset import (
    ACTION_NAMES,
    COURTESY_TYPES,
    GeneratorConfig,
    _POLICY_DT,
    choose_ego_action,
    compute_gaps,
    configure_courtesy_vehicle,
    estimate_ttc,
    in_interaction_window,
    lane_index,
    load_obs_model,
    make_courtesy_schedule,
    make_env,
    merge_vehicle_priority,
    noisy,
    safe_get_xy_speed,
    select_merge_vehicle,
    ttc_to_urgency,
    update_belief,
    vehicle_id,
)


DEFAULT_POLICIES = (
    "belief_policy", "rule_policy", "random_policy",
    "oracle_policy",
)
POLICIES = (
    *DEFAULT_POLICIES, "qmdp_policy", "oracle_qmdp", "pomcp_policy",
)
POLICY_SEED_OFFSETS: Dict[str, int] = {
    "belief_policy": 101,
    "rule_policy":   202,
    "random_policy": 303,
    "oracle_policy": 404,
    "qmdp_policy":   505,
    "oracle_qmdp":   606,
    "pomcp_policy":  707,
}
# Policies that require the QMDP Q-tables.
_QMDP_POLICIES = {"qmdp_policy", "oracle_qmdp"}
# Policies that require a POMCPPolicy instance.
_POMCP_POLICIES = {"pomcp_policy"}

# State discretisation for QMDP Q-learning.
# State = (speed_bin, gap_bin)  →  4 × 4 = 16 states.
_SPEED_EDGES = [18.0, 24.0, 30.0]   # 4 bins: <18, 18-24, 24-30, ≥30
_GAP_EDGES   = [5.0,  12.0, 25.0]   # 4 bins: <5,  5-12,  12-25, ≥25
_N_SPEED = len(_SPEED_EDGES) + 1    # 4
_N_GAP   = len(_GAP_EDGES) + 1      # 4
_N_STATES = _N_SPEED * _N_GAP       # 16
_N_ACTIONS = 5


def _discretize_state(ego_speed: float, front_gap: float) -> int:
    s = int(np.searchsorted(_SPEED_EDGES, ego_speed))
    g = int(np.searchsorted(_GAP_EDGES, front_gap if not np.isnan(front_gap) else 25.0))
    return s * _N_GAP + g


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",        type=int,   default=300)
    parser.add_argument("--max-steps",       type=int,   default=60)
    parser.add_argument("--seed",            type=int,   default=17)
    parser.add_argument("--output",          type=str,   default="policy_eval")
    parser.add_argument(
        "--traffic-density",
        type=float,
        default=None,
        help="Use a fixed traffic density for all policies, useful for density ablations.",
    )
    parser.add_argument("--qmdp-episodes",   type=int,   default=50,
                        help="Q-learning episodes per courtesy mode for QMDP solver.")
    parser.add_argument("--qmdp-seed",       type=int,   default=1000)
    parser.add_argument("--qmdp-cache",      type=str,   default="qmdp_qtable.json")
    parser.add_argument("--pomcp-sims",      type=int,   default=100,
                        help="UCT simulations per POMCP action selection.")
    parser.add_argument("--pomcp-horizon",   type=int,   default=10,
                        help="POMCP planning horizon (steps).")
    parser.add_argument("--pomcp-rollout-d-thresh",   type=float, default=15.0,
                        help="Distance threshold (m) for rollout heuristic: SLOWER when d < thresh.")
    parser.add_argument("--pomcp-rollout-ttc-thresh", type=float, default=6.0,
                        help="TTC threshold (s) for rollout heuristic: SLOWER when TTC < thresh.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_POLICIES),
        choices=POLICIES,
        help=(
            "Policies to evaluate. Defaults to the defensible main suite. "
            "qmdp_policy/oracle_qmdp are optional exploratory baselines. "
            "pomcp_policy runs the POMCP solver."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# QMDP: per-mode Q-learning solver
# ---------------------------------------------------------------------------

def solve_courtesy_q_table(
    courtesy: str,
    n_episodes: int = 50,
    seed: int = 1000,
    gamma: float = 0.95,
    alpha: float = 0.1,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,
) -> np.ndarray:
    """Q-learning in the fully observable MDP for *courtesy* mode.

    Returns Q[_N_STATES, _N_ACTIONS] with values in the highway-env reward scale.
    The agent uses ε-greedy exploration with linear decay over episodes.
    """
    Q = np.zeros((_N_STATES, _N_ACTIONS), dtype=float)
    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        eps = epsilon_start + (epsilon_end - epsilon_start) * ep / max(n_episodes - 1, 1)
        ep_seed = seed + ep
        env = make_env(ep_seed, 1.0)  # nominal traffic density for QMDP training
        mv = select_merge_vehicle(env)
        ep_rng = np.random.default_rng(ep_seed + 7)
        configure_courtesy_vehicle(mv, courtesy, ep_rng)

        for _ in range(60):
            ego = getattr(env.unwrapped, "vehicle", None)
            if ego is None:
                break
            ego_speed = safe_get_xy_speed(ego)[2]
            fg, _ = compute_gaps(env, ego)
            s = _discretize_state(ego_speed, fg)

            a = int(rng.integers(0, _N_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))

            _, reward, term, trunc, _ = env.step(a)
            done = bool(term or trunc)

            ego2 = getattr(env.unwrapped, "vehicle", None)
            if ego2 is not None:
                speed2 = safe_get_xy_speed(ego2)[2]
                fg2, _ = compute_gaps(env, ego2)
                s2 = _discretize_state(speed2, fg2)
            else:
                s2 = s

            target = float(reward) + (0.0 if done else gamma * float(np.max(Q[s2])))
            Q[s, a] += alpha * (target - Q[s, a])

            if done:
                break
        env.close()

    return Q


_QMDP_META_KEYS = ("n_episodes", "seed", "n_states", "n_actions", "speed_edges", "gap_edges", "class_names")


def _qmdp_meta(n_episodes: int, seed: int) -> Dict[str, Any]:
    return {
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "n_states": int(_N_STATES),
        "n_actions": int(_N_ACTIONS),
        "speed_edges": list(_SPEED_EDGES),
        "gap_edges": list(_GAP_EDGES),
        "class_names": list(COURTESY_TYPES),
    }


def build_qmdp_table(
    cache_path: str = "qmdp_qtable.json",
    n_episodes: int = 50,
    seed: int = 1000,
) -> Dict[str, np.ndarray]:
    """Solve per-mode Q-tables and cache to *cache_path* with provenance metadata.

    If a cache exists but its metadata (episode count, seed, state/action layout)
    does not match the requested configuration, it is recomputed rather than
    silently reused.  Returns dict mapping courtesy → Q[n_states, n_actions].
    """
    p = Path(cache_path)
    want_meta = _qmdp_meta(n_episodes, seed)
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        have_meta = raw.get("_meta", {})
        if all(have_meta.get(k) == want_meta[k] for k in _QMDP_META_KEYS):
            tables = {mode: np.array(raw[mode]) for mode in COURTESY_TYPES}
            print(f"[qmdp] Loaded Q-tables from {p.resolve()} (meta matches).")
            return tables
        print(f"[qmdp] Cache {p.name} metadata mismatch (have={have_meta}, want={want_meta}); recomputing.")

    print(f"[qmdp] Solving per-mode MDPs ({n_episodes} episodes × {len(COURTESY_TYPES)} modes)…")
    tables: Dict[str, np.ndarray] = {}
    for mode in COURTESY_TYPES:
        tables[mode] = solve_courtesy_q_table(mode, n_episodes=n_episodes, seed=seed)
        print(f"  [qmdp] {mode} done — mean Q per action: {tables[mode].mean(axis=0).round(3)}")

    payload: Dict[str, Any] = {"_meta": want_meta}
    payload.update({mode: tables[mode].tolist() for mode in COURTESY_TYPES})
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[qmdp] Saved Q-tables to {p.resolve()}")
    return tables


# ---------------------------------------------------------------------------
# Policy action functions
# ---------------------------------------------------------------------------

def qmdp_policy_action(
    belief: np.ndarray,
    front_gap: float,
    ego_speed: float,
    rng: np.random.Generator,
    q_tables: Dict[str, np.ndarray],
) -> int:
    """QMDP: a* = argmax_a  Σ_m b(m) * Q_m*(s, a)  at the current abstract state s.

    No hand-coded safety override is applied: any risk-aversion must come from the
    learned Q-tables themselves.  This keeps qmdp_policy / oracle_qmdp on equal
    footing with the other policies (whose only action machinery is the table).
    """
    s = _discretize_state(ego_speed, front_gap)
    q_values = np.zeros(_N_ACTIONS, dtype=float)
    for i, mode in enumerate(COURTESY_TYPES):
        q_values += float(belief[i]) * q_tables[mode][s]
    noise = rng.uniform(0.0, 1e-3, size=_N_ACTIONS)
    return int(np.argmax(q_values + noise))


def rule_policy_action(
    ttc: float,
    front_gap: float,
    rng: np.random.Generator,
    merge_speed: float = np.nan,
    ego_speed: float = np.nan,
    relative_distance: float = np.nan,
) -> int:
    """Same action table as belief_policy with belief frozen at the uniform prior.
    Properly isolates the marginal value of belief estimation.
    """
    uniform_belief = np.ones(len(COURTESY_TYPES), dtype=float) / len(COURTESY_TYPES)
    return choose_ego_action(
        uniform_belief, ttc, front_gap, rng,
        merge_speed=merge_speed, ego_speed=ego_speed, relative_distance=relative_distance,
    )


def _one_hot_belief(courtesy: str) -> np.ndarray:
    oh = np.zeros(len(COURTESY_TYPES), dtype=float)
    oh[list(COURTESY_TYPES).index(courtesy)] = 1.0
    return oh


def select_action(
    policy_name: str,
    courtesy: str,
    belief: np.ndarray,
    ttc: float,
    front_gap: float,
    ego_speed: float,
    rng: np.random.Generator,
    q_tables: Optional[Dict[str, np.ndarray]] = None,
    merge_speed: float = np.nan,
    relative_distance: float = np.nan,
    pomcp_instance=None,
) -> int:
    if policy_name == "belief_policy":
        return choose_ego_action(
            belief, ttc, front_gap, rng,
            merge_speed=merge_speed, ego_speed=ego_speed, relative_distance=relative_distance,
        )
    if policy_name == "rule_policy":
        return rule_policy_action(
            ttc, front_gap, rng,
            merge_speed=merge_speed, ego_speed=ego_speed, relative_distance=relative_distance,
        )
    if policy_name == "random_policy":
        return int(rng.integers(0, 5))
    if policy_name == "oracle_policy":
        return choose_ego_action(
            _one_hot_belief(courtesy), ttc, front_gap, rng,
            merge_speed=merge_speed, ego_speed=ego_speed, relative_distance=relative_distance,
        )
    if policy_name in _QMDP_POLICIES:
        if q_tables is None:
            raise ValueError(f"q_tables must be provided for {policy_name}")
        b = _one_hot_belief(courtesy) if policy_name == "oracle_qmdp" else belief
        return qmdp_policy_action(b, front_gap, ego_speed, rng, q_tables)
    if policy_name in _POMCP_POLICIES:
        if pomcp_instance is None:
            raise ValueError("pomcp_instance must be provided for pomcp_policy")
        return pomcp_instance.plan(
            rel_dist=relative_distance,
            ego_speed=ego_speed,
            merge_speed=merge_speed,
            belief=belief,
        )
    raise ValueError(f"Unknown policy: {policy_name}")


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------

def evaluate_episode(
    episode_id: int,
    policy_name: str,
    seed: int,
    courtesy: str,
    traffic_density: float,
    observation_noise: float,
    config: GeneratorConfig,
    q_tables: Optional[Dict[str, np.ndarray]] = None,
    pomcp_instance=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    env_rng    = np.random.default_rng(seed)
    policy_rng = np.random.default_rng(seed + POLICY_SEED_OFFSETS[policy_name])

    # POMCP: reset tree and belief at the start of each episode
    if pomcp_instance is not None and policy_name in _POMCP_POLICIES:
        pomcp_instance.reset(seed=seed + POLICY_SEED_OFFSETS[policy_name])

    env = make_env(seed, traffic_density)  # make_env already calls env.reset(seed=seed)

    merge_vehicle = select_merge_vehicle(env)
    merge_vid = vehicle_id(merge_vehicle)
    merge_priority = merge_vehicle_priority(merge_vehicle) if merge_vehicle is not None else 99
    configure_courtesy_vehicle(merge_vehicle, courtesy, env_rng)

    belief = np.array(config.courtesy_prior, dtype=float)
    total_reward = 0.0
    collision = False
    lane_change_count = 0
    harsh_brake_count = 0
    speeds: List[float] = []
    ttcs: List[float] = []
    critical_ttcs: List[float] = []
    critical_abs_relative_distances: List[float] = []
    critical_speeds: List[float] = []
    rows: List[Dict[str, Any]] = []
    initial_ego_x = np.nan
    previous_lane: Optional[int] = None
    previous_speed: Optional[float] = None
    prev_merge_speed: Optional[float] = None
    close_call_count = 0

    for timestep in range(config.max_steps):
        ego = getattr(env.unwrapped, "vehicle", None)

        ego_x, ego_y, ego_speed = safe_get_xy_speed(ego)
        merge_x, merge_y, merge_speed = (
            safe_get_xy_speed(merge_vehicle) if merge_vehicle else (np.nan, np.nan, np.nan)
        )
        # Merge vehicle acceleration (finite difference; NaN on first step of episode)
        if prev_merge_speed is not None and not np.isnan(merge_speed) and not np.isnan(prev_merge_speed):
            mva = (merge_speed - prev_merge_speed) / _POLICY_DT
        else:
            mva = np.nan
        if not np.isnan(merge_speed):
            prev_merge_speed = merge_speed
        ego_lane = lane_index(ego)
        if np.isnan(initial_ego_x) and not np.isnan(ego_x):
            initial_ego_x = ego_x

        relative_distance = merge_x - ego_x if not np.isnan(merge_x) else np.nan
        relative_speed    = merge_speed - ego_speed if not np.isnan(merge_speed) else np.nan
        estimated_ttc = estimate_ttc(relative_distance, relative_speed)
        front_gap, rear_gap = compute_gaps(env, ego)

        observed_ttc                  = noisy(estimated_ttc,  observation_noise,       env_rng)
        observed_relative_distance    = noisy(relative_distance, observation_noise * 2.0, env_rng)
        observed_abs_relative_distance = (
            abs(float(observed_relative_distance)) if not np.isnan(observed_relative_distance) else np.nan
        )
        observed_relative_speed       = noisy(relative_speed, observation_noise,       env_rng)
        observed_merge_speed          = noisy(merge_speed,    observation_noise,       env_rng)
        observed_front_gap            = noisy(front_gap,      observation_noise * 2.0, env_rng)
        observed_merge_acceleration   = noisy(mva,            observation_noise * 4.0, env_rng)
        observed_urgency              = ttc_to_urgency(observed_ttc, relative_distance, relative_speed)
        urgency = ttc_to_urgency(estimated_ttc, relative_distance, relative_speed)
        in_window = in_interaction_window(relative_distance, merge_speed)
        if in_window:
            if not np.isnan(relative_distance):
                critical_abs_relative_distances.append(abs(float(relative_distance)))
            if not np.isnan(ego_speed):
                critical_speeds.append(float(ego_speed))
            if not np.isnan(estimated_ttc) and not np.isnan(relative_distance) and relative_distance > 0:
                critical_ttcs.append(float(estimated_ttc))
        if in_window:
            belief = update_belief(
                belief,
                observed_urgency, observed_abs_relative_distance,
                observed_relative_speed, observed_merge_speed,
                merge_vehicle_acceleration=observed_merge_acceleration,
                regularization=config.belief_regularization,
            )

        action = select_action(
            policy_name, courtesy, belief,
            observed_ttc, observed_front_gap, ego_speed,
            policy_rng, q_tables,
            merge_speed=observed_merge_speed,
            relative_distance=relative_distance,
            pomcp_instance=pomcp_instance,
        )
        _, env_reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        collision = collision or bool(getattr(ego, "crashed", False))

        if previous_lane is not None and ego_lane != previous_lane:
            lane_change_count += 1
        if previous_speed is not None and ego_speed - previous_speed < -2.5:
            harsh_brake_count += 1
        previous_lane  = ego_lane
        previous_speed = ego_speed
        speeds.append(ego_speed)
        if not np.isnan(estimated_ttc):
            ttcs.append(estimated_ttc)
        close_call = bool(
            (not np.isnan(estimated_ttc) and not np.isnan(relative_distance) and relative_distance > 0 and estimated_ttc < 3.0)
            or (not np.isnan(relative_distance) and 0 < relative_distance < 15.0)
        )
        close_call_count += int(close_call)
        reward = float(env_reward) - config.close_call_reward_penalty * float(close_call)
        total_reward += reward

        rows.append({
            "policy_name":        policy_name,
            "episode_id":         episode_id,
            "timestep":           timestep,
            "seed":               seed,
            "hidden_courtesy":    courtesy,
            "ego_x":              ego_x,
            "ego_y":              ego_y,
            "ego_speed":          ego_speed,
            "ego_lane":           ego_lane,
            "merge_x":            merge_x,
            "merge_y":            merge_y,
            "merge_speed":        merge_speed,
            "merge_vehicle_id":   merge_vid,
            "merge_vehicle_priority": merge_priority,
            "merge_lane":         lane_index(merge_vehicle) if merge_vehicle else -1,
            "relative_distance":  relative_distance,
            "abs_relative_distance": abs(float(relative_distance)) if not np.isnan(relative_distance) else np.nan,
            "relative_speed":     relative_speed,
            "estimated_ttc":      estimated_ttc,
            "urgency":            urgency,
            "observed_relative_distance": observed_relative_distance,
            "observed_abs_relative_distance": observed_abs_relative_distance,
            "observed_relative_speed": observed_relative_speed,
            "observed_urgency": observed_urgency,
            "observed_merge_speed": observed_merge_speed,
            "observed_merge_acceleration": observed_merge_acceleration,
            "front_gap":          front_gap,
            "rear_gap":           rear_gap,
            "merge_acceleration": mva,
            "in_interaction_window": bool(in_window),
            "belief_cooperative": float(belief[0]),
            "belief_non_cooperative": float(belief[1]),
            "action_id":          action,
            "action_name":        ACTION_NAMES.get(action, f"ACTION_{action}"),
            "reward":             float(reward),
            "env_reward":         float(env_reward),
            "close_call":         bool(close_call),
            "collision":          collision,
            "done":               done,
        })
        if done:
            break

    final_ego_x = rows[-1]["ego_x"] if rows else np.nan
    progress_m = (
        final_ego_x - initial_ego_x
        if not np.isnan(final_ego_x) and not np.isnan(initial_ego_x)
        else np.nan
    )
    success = bool(
        not collision and not np.isnan(progress_m) and progress_m >= config.success_min_progress_m
    )
    env.close()

    episode = {
        "policy_name":       policy_name,
        "episode_id":        episode_id,
        "seed":              seed,
        "hidden_courtesy":   courtesy,
        "traffic_density":   traffic_density,
        "observation_noise": observation_noise,
        "collision":         collision,
        "success":           success,
        "min_ttc":           float(np.nanmin(ttcs)) if ttcs else np.nan,
        "critical_mean_ttc": float(np.nanmean(critical_ttcs)) if critical_ttcs else np.nan,
        "critical_min_ttc":  float(np.nanmin(critical_ttcs)) if critical_ttcs else np.nan,
        "critical_mean_abs_relative_distance": (
            float(np.nanmean(critical_abs_relative_distances))
            if critical_abs_relative_distances else np.nan
        ),
        "mean_speed":        float(np.nanmean(speeds)) if speeds else np.nan,
        "ego_speed_std":     float(np.nanstd(speeds)) if speeds else np.nan,
        "critical_speed_std": float(np.nanstd(critical_speeds)) if critical_speeds else np.nan,
        "total_reward":      float(total_reward),
        "close_call_rate":   float(close_call_count / len(rows)) if rows else np.nan,
        "lane_change_count": lane_change_count,
        "harsh_brake_count": harsh_brake_count,
        "episode_length":    len(rows),
    }
    return episode, rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci95(
    values: np.ndarray,
    n_boot: int = 2000,
    rng_seed: int = 0,
) -> Tuple[float, float]:
    """Non-parametric bootstrap 95% CI for binary metrics."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(rng_seed)
    boot = np.fromiter(
        (np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)),
        dtype=float,
        count=n_boot,
    )
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def mean_ci95(values: np.ndarray) -> Tuple[float, float]:
    """Normal-approximation 95% CI for continuous metrics (n > 30)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean
    half_width = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sided paired t-test (normal approximation, valid for n > 30)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mask = ~np.isnan(diff)
    diff = diff[mask]
    n = len(diff)
    if n < 2:
        return np.nan, np.nan
    mean_diff = float(np.mean(diff))
    std_diff  = float(np.std(diff, ddof=1))
    if std_diff < 1e-12:
        return np.nan, np.nan
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    p_val  = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return float(t_stat), float(p_val)


def _chi2_cdf_1df(x: float) -> float:
    """CDF of the chi-squared distribution with 1 degree of freedom."""
    if x <= 0:
        return 0.0
    return math.erf(math.sqrt(x / 2.0))


def mcnemar_test(a_binary: np.ndarray, b_binary: np.ndarray) -> Tuple[float, float]:
    """McNemar's test for paired binary outcomes, continuity-corrected χ²₁ approx.

    Returns (chi2_statistic, p_value).  Uses Edwards' continuity correction
    (|n10 − n01| − 1)² / (n10 + n01); reasonable for discordant count ≳ 25.
    For small discordant counts an exact binomial test would be preferable.
    """
    a = np.asarray(a_binary, dtype=bool)
    b = np.asarray(b_binary, dtype=bool)
    mask = ~(np.isnan(a_binary) | np.isnan(b_binary))
    a, b = a[mask], b[mask]
    n10 = int(np.sum(a & ~b))   # a=1, b=0
    n01 = int(np.sum(~a & b))   # a=0, b=1
    discordant = n10 + n01
    if discordant == 0:
        return np.nan, np.nan
    chi2 = max(abs(n10 - n01) - 1.0, 0.0) ** 2 / discordant
    p_val = 1.0 - _chi2_cdf_1df(chi2)
    return float(chi2), float(p_val)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d = mean(a − b) / sd(a − b).  NaN if undefined."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(diff) / sd)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions: 2·asin(√p1) − 2·asin(√p2)."""
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return float(2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2)))


def n_for_two_proportion_power(p1: float, p2: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """Per-group n to detect a p1-vs-p2 difference at the given power (two-sided).

    Uses the normal-approximation formula based on Cohen's h:
        n = (z_{1-α/2} + z_{1-power})² / h²
    Returns inf when p1 == p2.
    """
    h = abs(cohens_h(p1, p2))
    if h < 1e-9:
        return float("inf")
    z_alpha = 1.959963985  # Φ⁻¹(0.975)
    z_power = {0.80: 0.8416212336, 0.90: 1.281551566, 0.95: 1.644853627}.get(round(power, 2), 0.8416212336)
    return float((z_alpha + z_power) ** 2 / h ** 2)


def power_analysis(episodes: pd.DataFrame) -> Dict[str, Any]:
    """Sample-size justification, focused on the (low-base-rate) collision metric.

    Reports the pooled collision rate, the achieved n per policy, and the n that
    would be needed to detect a few illustrative absolute differences (Δ) in
    collision rate at 80% power, α=0.05.  Also reports the same for success rate.
    """
    out: Dict[str, Any] = {}
    n_per_policy = int(episodes.groupby("policy_name")["episode_id"].count().min())
    out["n_per_policy_achieved"] = n_per_policy
    for col in ("collision", "success"):
        if col not in episodes.columns:
            continue
        base = float(episodes[col].astype(float).mean())
        deltas = [0.02, 0.03, 0.05, 0.10]
        needed = {
            f"delta_{d:.2f}": round(n_for_two_proportion_power(
                min(base + d, 1.0), base, power=0.8, alpha=0.05), 1)
            for d in deltas
        }
        out[col] = {
            "pooled_rate": round(base, 4),
            "n_needed_per_group_for_80pct_power": needed,
            "note": (
                "Achieved n per policy is "
                f"{n_per_policy}. If the table above asks for more than that, "
                f"the corresponding {col}-rate comparison is underpowered and should "
                "be reported as suggestive only."
            ),
        }
    return out


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Holm–Bonferroni correction.  Returns adjusted p-values in the same order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [np.nan] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = p_values[idx]
        if math.isnan(p):
            adjusted[idx] = np.nan
        else:
            corrected = min(1.0, max(running_max, (m - rank) * p))
            running_max = corrected
            adjusted[idx] = corrected
    return adjusted


_BINARY_COLUMNS = {"collision", "success"}


def confidence_intervals(episodes: pd.DataFrame, group_columns: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in episodes.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for column, prefix in [
            ("total_reward",      "mean_reward"),
            ("collision",         "collision_rate"),
            ("success",           "success_rate"),
            ("close_call_rate",   "close_call_rate"),
            ("min_ttc",           "mean_min_ttc"),
            ("critical_mean_ttc", "mean_critical_ttc"),
            ("critical_min_ttc",  "mean_critical_min_ttc"),
            ("critical_mean_abs_relative_distance", "mean_critical_abs_relative_distance"),
            ("ego_speed_std",     "mean_ego_speed_std"),
            ("critical_speed_std", "mean_critical_speed_std"),
            ("lane_change_count", "mean_lane_changes"),
            ("harsh_brake_count", "mean_harsh_brakes"),
        ]:
            vals = group[column].astype(float).dropna().to_numpy()
            if column in _BINARY_COLUMNS:
                low, high = bootstrap_ci95(vals)
                low = low if np.isnan(low) else max(0.0, low)
                high = high if np.isnan(high) else min(1.0, high)
            else:
                low, high = mean_ci95(vals)
                if prefix in {
                    "mean_min_ttc",
                    "mean_critical_ttc",
                    "mean_critical_min_ttc",
                    "mean_critical_abs_relative_distance",
                    "mean_ego_speed_std",
                    "mean_critical_speed_std",
                    "mean_lane_changes",
                    "mean_harsh_brakes",
                }:
                    low = low if np.isnan(low) else max(0.0, low)
                    high = high if np.isnan(high) else max(0.0, high)
            row[f"{prefix}_ci95_low"]  = low
            row[f"{prefix}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


_CONTINUOUS_METRICS = ("total_reward", "min_ttc", "close_call_rate")
_BINARY_METRICS = ("collision", "success")


def _pairwise_tests(episodes_subset: pd.DataFrame) -> Dict[str, Any]:
    """Pairwise tests over all policy pairs within a (matched-episode) subset.

    Paired t-test + paired Cohen's d for continuous metrics; McNemar + Cohen's h
    for binary metrics; Holm–Bonferroni correction applied across the whole batch.
    Returns {} if the subset is degenerate (fewer than 2 policies or unmatched n).
    """
    policies = sorted(episodes_subset["policy_name"].unique())
    if len(policies) < 2:
        return {}
    pairs = [(p1, p2) for i, p1 in enumerate(policies) for p2 in policies[i + 1:]]

    results: Dict[str, Any] = {}
    raw_p_list: List[float] = []
    key_list: List[str] = []

    for p1, p2 in pairs:
        g1 = episodes_subset[episodes_subset["policy_name"] == p1].sort_values("episode_id")
        g2 = episodes_subset[episodes_subset["policy_name"] == p2].sort_values("episode_id")
        # match on episode_id
        common = sorted(set(g1["episode_id"]) & set(g2["episode_id"]))
        if len(common) < 2:
            continue
        g1 = g1[g1["episode_id"].isin(common)].sort_values("episode_id")
        g2 = g2[g2["episode_id"].isin(common)].sort_values("episode_id")
        n = len(common)

        for metric in _CONTINUOUS_METRICS:
            a = g1[metric].to_numpy(dtype=float)
            b = g2[metric].to_numpy(dtype=float)
            t, p = paired_ttest(a, b)
            d = cohens_d_paired(a, b)
            key = f"{p1}_vs_{p2}__{metric}"
            results[key] = {
                "test": "paired_t",
                "statistic": round(t, 4) if not math.isnan(t) else None,
                "p_raw": round(p, 4) if not math.isnan(p) else None,
                "p_adjusted": None,
                "significant_at_alpha_0.05": None,
                "effect_size": round(d, 4) if not math.isnan(d) else None,
                "effect_size_kind": "cohens_d_paired",
                "n_pairs": n,
                "mean_diff": round(float(np.nanmean(a - b)), 4),
            }
            raw_p_list.append(p if not math.isnan(p) else 1.0)
            key_list.append(key)

        for metric in _BINARY_METRICS:
            a = g1[metric].to_numpy(dtype=float)
            b = g2[metric].to_numpy(dtype=float)
            chi2, p = mcnemar_test(a, b)
            h = cohens_h(float(np.nanmean(a)), float(np.nanmean(b)))
            key = f"{p1}_vs_{p2}__{metric}"
            results[key] = {
                "test": "mcnemar",
                "statistic": round(chi2, 4) if not math.isnan(chi2) else None,
                "p_raw": round(p, 4) if not math.isnan(p) else None,
                "p_adjusted": None,
                "significant_at_alpha_0.05": None,
                "effect_size": round(h, 4),
                "effect_size_kind": "cohens_h",
                "n_pairs": n,
                "rate_diff": round(float(np.nanmean(a) - np.nanmean(b)), 4),
            }
            raw_p_list.append(p if not math.isnan(p) else 1.0)
            key_list.append(key)

    adj_p = holm_bonferroni_correction(raw_p_list)
    for key, p_adj in zip(key_list, adj_p):
        results[key]["p_adjusted"] = round(float(p_adj), 4) if not math.isnan(p_adj) else None
        results[key]["significant_at_alpha_0.05"] = (
            bool(p_adj < 0.05) if not math.isnan(p_adj) else None
        )
    return results


def policy_significance_tests(episodes: pd.DataFrame) -> Dict[str, Any]:
    """Pooled pairwise significance tests (all courtesy modes together)."""
    return _pairwise_tests(episodes)


def significance_tests_by_courtesy(episodes: pd.DataFrame) -> Dict[str, Any]:
    """Pairwise tests run *within* each courtesy mode (Holm-corrected per mode).

    This is the scientifically central comparison: e.g. does belief_policy beat
    rule_policy specifically when the hidden driver is non_cooperative?
    """
    out: Dict[str, Any] = {}
    for courtesy, sub in episodes.groupby("hidden_courtesy"):
        out[str(courtesy)] = _pairwise_tests(sub)
    return out


def negotiation_metric(steps: pd.DataFrame) -> pd.DataFrame:
    """Action distribution conditional on (policy, true_courtesy).

    Reveals how each policy responds differently to each hidden driver type,
    which is the core of the negotiation behaviour being studied.
    """
    if steps.empty or "action_name" not in steps.columns:
        return pd.DataFrame()
    counts = (
        steps.groupby(["policy_name", "hidden_courtesy", "action_name"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        steps.groupby(["policy_name", "hidden_courtesy"])
        .size()
        .reset_index(name="total")
    )
    merged = counts.merge(totals, on=["policy_name", "hidden_courtesy"])
    merged["proportion"] = merged["count"] / merged["total"]
    return merged.sort_values(["policy_name", "hidden_courtesy", "action_name"])


def belief_accuracy_metrics(steps: pd.DataFrame) -> Dict[str, Any]:
    """Belief accuracy over in-window steps, grouped by (policy_name, hidden_courtesy).

    For each policy and hidden courtesy type, reports:
    - n_in_window_steps: how many steps fell inside the interaction window
    - accuracy: fraction of those steps where argmax(belief) == true courtesy
    - mean_true_class_belief: mean belief assigned to the correct class
    """
    if steps.empty:
        return {}

    courtesy_to_col = {
        "cooperative": "belief_cooperative",
        "non_cooperative": "belief_non_cooperative",
    }
    belief_cols = list(courtesy_to_col.values())
    if not all(c in steps.columns for c in belief_cols):
        return {}

    df = steps.copy()
    if "in_interaction_window" in df.columns:
        df["in_window"] = df["in_interaction_window"].astype(bool)
    else:
        rel = df["relative_distance"].to_numpy(dtype=float)
        mvs = df["merge_speed"].to_numpy(dtype=float)
        df["in_window"] = np.isfinite(rel) & np.isfinite(mvs) & (np.abs(rel) <= 80.0)
    true_cols = df["hidden_courtesy"].map(courtesy_to_col)
    belief_matrix = df[belief_cols].to_numpy(dtype=float)
    col_index = {col: idx for idx, col in enumerate(belief_cols)}
    true_indices = true_cols.map(col_index).to_numpy()
    valid_true = pd.notna(true_cols).to_numpy()
    df["true_class_belief"] = np.nan
    valid_rows = np.where(valid_true)[0]
    df.loc[valid_true, "true_class_belief"] = belief_matrix[valid_rows, true_indices[valid_true].astype(int)]
    df["belief_argmax"] = df[belief_cols].idxmax(axis=1).map({
        "belief_cooperative": "cooperative",
        "belief_non_cooperative": "non_cooperative",
    })
    df["belief_correct"] = df["belief_argmax"] == df["hidden_courtesy"]

    win = df[df["in_window"]]
    if win.empty:
        return {}

    first_correct = (
        win[win["belief_correct"]]
        .sort_values(["policy_name", "hidden_courtesy", "episode_id", "timestep"])
        .groupby(["policy_name", "hidden_courtesy", "episode_id"])["timestep"]
        .first()
        .reset_index(name="steps_to_correct_belief")
    )

    out: Dict[str, Any] = {}
    for policy_name in sorted(df["policy_name"].unique()):
        pw = win[win["policy_name"] == policy_name]
        out[policy_name] = {}
        for courtesy in COURTESY_TYPES:
            pc = pw[pw["hidden_courtesy"] == courtesy]
            if pc.empty:
                out[policy_name][courtesy] = {"n_in_window_steps": 0}
                continue
            out[policy_name][courtesy] = {
                "n_in_window_steps":      int(len(pc)),
                "accuracy":               round(float(pc["belief_correct"].mean()), 4),
                "mean_true_class_belief": round(float(pc["true_class_belief"].mean()), 4),
                "mean_steps_to_correct_belief": round(
                    float(
                        first_correct[
                            (first_correct["policy_name"] == policy_name)
                            & (first_correct["hidden_courtesy"] == courtesy)
                        ]["steps_to_correct_belief"].mean()
                    ),
                    3,
                ),
            }
        out[policy_name]["overall"] = {
            "n_in_window_steps": int(len(pw)),
            "accuracy":          round(float(pw["belief_correct"].mean()), 4),
        }
    return out


def summarize(episodes: pd.DataFrame, group_columns: List[str]) -> pd.DataFrame:
    agg = (
        episodes.groupby(group_columns)
        .agg(
            episodes         = ("episode_id",        "count"),
            collision_rate   = ("collision",          "mean"),
            success_rate     = ("success",            "mean"),
            close_call_rate  = ("close_call_rate",     "mean"),
            mean_reward      = ("total_reward",       "mean"),
            std_reward       = ("total_reward",       "std"),
            mean_min_ttc     = ("min_ttc",            "mean"),
            mean_critical_ttc= ("critical_mean_ttc",   "mean"),
            mean_critical_min_ttc= ("critical_min_ttc", "mean"),
            mean_critical_abs_relative_distance= (
                "critical_mean_abs_relative_distance", "mean"
            ),
            mean_speed       = ("mean_speed",         "mean"),
            mean_ego_speed_std= ("ego_speed_std",      "mean"),
            mean_critical_speed_std= ("critical_speed_std", "mean"),
            mean_lane_changes= ("lane_change_count",  "mean"),
            mean_harsh_brakes= ("harsh_brake_count",  "mean"),
            mean_episode_length= ("episode_length",   "mean"),
        )
        .reset_index()
        .fillna({"std_reward": 0.0})
        .sort_values("mean_reward", ascending=False)
    )
    ci = confidence_intervals(episodes, group_columns)
    return agg.merge(ci, on=group_columns, how="left")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    metrics = [
        ("mean_reward",   "Average Reward"),
        ("collision_rate","Collision Rate"),
        ("success_rate",  "Success Rate"),
        ("close_call_rate", "Close-Call Rate"),
        ("mean_min_ttc",  "Mean Minimum TTC (s)"),
        ("mean_critical_ttc", "Mean Critical-Window TTC (s)"),
        ("mean_critical_speed_std", "Critical-Window Speed Std. Dev."),
    ]
    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        policies = summary["policy_name"].tolist()
        values   = np.array(summary[metric].tolist(), dtype=float)

        lo_col, hi_col = f"{metric}_ci95_low", f"{metric}_ci95_high"
        if lo_col in summary.columns and hi_col in summary.columns:
            lo   = np.array(summary[lo_col].tolist(), dtype=float)
            hi   = np.array(summary[hi_col].tolist(), dtype=float)
            yerr = np.vstack([values - lo, hi - values])
            ax.bar(policies, values, color="#4c78a8", yerr=yerr, capsize=6,
                   error_kw={"elinewidth": 1.5, "ecolor": "#333333"})
        else:
            ax.bar(policies, values, color="#4c78a8")

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Policy")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figures / f"{metric}.png", dpi=160)
        plt.close(fig)


def save_belief_convergence_plot(steps: pd.DataFrame, output_dir: Path) -> None:
    """Mean belief in the true courtesy class over timestep, for belief_policy."""
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    bp = steps[steps["policy_name"] == "belief_policy"].copy()
    if bp.empty:
        return

    courtesy_to_col = {
        "cooperative": "belief_cooperative",
        "non_cooperative": "belief_non_cooperative",
    }

    def _true_belief(row: pd.Series) -> float:
        col = courtesy_to_col.get(row["hidden_courtesy"])
        return row[col] if col else np.nan

    if all(c in bp.columns for c in courtesy_to_col.values()):
        bp["true_belief"] = bp.apply(_true_belief, axis=1)
        agg = bp.groupby(["hidden_courtesy", "timestep"])["true_belief"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = {"cooperative": "#2c7fb8", "non_cooperative": "#f03b20"}
        for courtesy in COURTESY_TYPES:
            sub = agg[agg["hidden_courtesy"] == courtesy]
            ax.plot(sub["timestep"], sub["true_belief"], label=courtesy,
                    color=colors[courtesy], linewidth=1.8)
        ax.axhline(1 / len(COURTESY_TYPES), color="gray", linestyle="--", linewidth=1, label="uniform prior")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean belief in true courtesy class")
        ax.set_title("Belief convergence (belief_policy)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / "belief_convergence.png", dpi=160)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    load_obs_model()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        output=args.output,
        traffic_density=args.traffic_density,
    )

    q_tables: Optional[Dict[str, np.ndarray]] = None
    if any(policy in _QMDP_POLICIES for policy in args.policies):
        q_tables = build_qmdp_table(
            cache_path=args.qmdp_cache,
            n_episodes=args.qmdp_episodes,
            seed=args.qmdp_seed,
        )

    pomcp_instance = None
    if any(policy in _POMCP_POLICIES for policy in args.policies):
        from pomcp_policy import POMCPPolicy
        print(
            f"[POMCP] Building solver (n_sims={getattr(args, 'pomcp_sims', 100)}, "
            f"horizon={getattr(args, 'pomcp_horizon', 10)}, "
            f"rollout_d={getattr(args, 'pomcp_rollout_d_thresh', 15.0)}, "
            f"rollout_ttc={getattr(args, 'pomcp_rollout_ttc_thresh', 6.0)})...",
            flush=True,
        )
        pomcp_instance = POMCPPolicy.build(
            obs_model_path="observation_model.json",
            n_simulations=getattr(args, "pomcp_sims", 100),
            horizon=getattr(args, "pomcp_horizon", 10),
            rollout_d_thresh=getattr(args, "pomcp_rollout_d_thresh", 15.0),
            rollout_ttc_thresh=getattr(args, "pomcp_rollout_ttc_thresh", 6.0),
        )
        print("[POMCP] Solver ready.", flush=True)

    courtesy_schedule = make_courtesy_schedule(args.episodes, rng)
    traffic = (
        np.full(args.episodes, float(args.traffic_density))
        if args.traffic_density is not None
        else rng.uniform(*config.traffic_density_range, size=args.episodes)
    )
    noise   = rng.uniform(*config.observation_noise_range, size=args.episodes)

    episode_rows: List[Dict[str, Any]] = []
    step_rows:    List[Dict[str, Any]] = []

    for policy_name in args.policies:
        for episode_id in range(args.episodes):
            seed = args.seed + episode_id
            episode, steps = evaluate_episode(
                episode_id=episode_id,
                policy_name=policy_name,
                seed=seed,
                courtesy=courtesy_schedule[episode_id],
                traffic_density=float(traffic[episode_id]),
                observation_noise=float(noise[episode_id]),
                config=config,
                q_tables=q_tables if policy_name in _QMDP_POLICIES else None,
                pomcp_instance=pomcp_instance if policy_name in _POMCP_POLICIES else None,
            )
            episode_rows.append(episode)
            step_rows.extend(steps)

    episodes = pd.DataFrame(episode_rows)
    steps    = pd.DataFrame(step_rows)

    summary            = summarize(episodes, ["policy_name"])
    summary_by_courtesy= summarize(episodes, ["policy_name", "hidden_courtesy"])
    sig_tests_pooled   = policy_significance_tests(episodes)
    sig_tests_by_courtesy = significance_tests_by_courtesy(episodes)
    power               = power_analysis(episodes)
    neg_metric         = negotiation_metric(steps)
    belief_acc         = belief_accuracy_metrics(steps)

    episodes.to_csv(output_dir / "policy_episodes.csv", index=False)
    steps.to_csv(output_dir / "policy_steps.csv", index=False)
    summary.to_csv(output_dir / "policy_summary.csv", index=False)
    summary_by_courtesy.to_csv(output_dir / "policy_summary_by_courtesy.csv", index=False)
    if not neg_metric.empty:
        neg_metric.to_csv(output_dir / "negotiation_metric.csv", index=False)

    save_plots(summary, output_dir)
    save_belief_convergence_plot(steps, output_dir)

    if belief_acc:
        print("\n--- Belief Accuracy (in-window steps) ---")
        for pol, modes in sorted(belief_acc.items()):
            overall = modes.get("overall", {})
            print(f"  {pol}:")
            for courtesy in COURTESY_TYPES:
                cm = modes.get(courtesy, {})
                acc = cm.get("accuracy", float("nan"))
                mtb = cm.get("mean_true_class_belief", float("nan"))
                n   = cm.get("n_in_window_steps", 0)
                print(f"    {courtesy:12s}: acc={acc:.1%}  mean_true_belief={mtb:.3f}  n={n}")
            print(f"    {'overall':12s}: acc={overall.get('accuracy', float('nan')):.1%}  n={overall.get('n_in_window_steps', 0)}")

    payload = {
        "note": "Simulation-only policy comparison for controlled synthetic latent driver behaviour.",
        "policies": list(args.policies),
        "episodes_per_policy": args.episodes,
        "fairness_note": (
            "All policies share the same env seed, hidden courtesy, traffic density, and "
            "observation noise per episode index. Action sampling uses a separate per-policy "
            "RNG offset so it does not advance the exogenous scenario stream."
        ),
        "qmdp_note": (
            "QMDP baselines were not included in this run."
            if not any(policy in _QMDP_POLICIES for policy in args.policies)
            else (
                "Optional exploratory QMDP Q-tables are computed via tabular Q-learning "
                "in a coarse fully observable MDP for each courtesy mode, then cached to "
                "qmdp_qtable.json. This baseline is not treated as a main paper claim."
            )
        ),
        "oracle_note": (
            "oracle_policy feeds a one-hot belief at the true courtesy label into the same "
            "heuristic action table. Upper bound for this action table only."
        ),
        "oracle_qmdp_note": (
            "oracle_qmdp was not included in this run."
            if "oracle_qmdp" not in args.policies
            else (
                "oracle_qmdp is the optional exploratory QMDP baseline with a one-hot "
                "belief at the true courtesy label."
            )
        ),
        "rule_policy_note": (
            "rule_policy uses the same action table as belief_policy with belief frozen at "
            "uniform prior (0.5, 0.5). This isolates the marginal value of belief."
        ),
        "significance_tests_note": (
            "Paired t-test for reward/TTC; McNemar test for collision/success. "
            "Holm–Bonferroni correction applied within each test set. "
            "significance_tests_pooled: all courtesy modes together. "
            "significance_tests_by_courtesy: tests run *within* each courtesy mode "
            "(scientifically central: does belief_policy beat rule_policy when driver is non_cooperative?)."
        ),
        "power_analysis_note": (
            "Sample-size justification: achieved n per policy vs n needed to detect "
            "illustrative absolute differences (Δ) at 80% power, α=0.05."
        ),
        "significance_tests_pooled": sig_tests_pooled,
        "significance_tests_by_courtesy": sig_tests_by_courtesy,
        "power_analysis": power,
        "belief_accuracy": belief_acc,
        "summary": summary.to_dict(orient="records"),
        "summary_by_courtesy": summary_by_courtesy.to_dict(orient="records"),
    }
    output_dir.joinpath("policy_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
