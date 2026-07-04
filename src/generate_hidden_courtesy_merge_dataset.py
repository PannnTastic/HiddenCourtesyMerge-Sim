"""
Generate HiddenCourtesyMerge-Sim, a simulation-only benchmark dataset for:

POMDP-Based Merge Negotiation Under Hidden Driver Courtesy in Highway Simulation

The hidden variable is driver courtesy: cooperative or non_cooperative.
The ego vehicle never observes courtesy directly; belief states are estimated
from motion cues such as TTC, relative speed, and observed gap behavior.

Courtesy is injected by modifying IDM/MOBIL instance parameters (target_speed,
DISTANCE_WANTED, TIME_WANTED, COMFORT_ACC_MAX, POLITENESS) on the selected merge
vehicle once at episode start. No teleportation or per-step act() overrides are
used; vehicles evolve under highway-env's standard physics.

Dependencies (see requirements.txt):
    pip install -r requirements.txt

Example:
    python generate_hidden_courtesy_merge_dataset.py --episodes 200 --output HiddenCourtesyMerge-Sim
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import highway_env  # noqa: F401  # registers highway-env environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from highway_env.vehicle.behavior import IDMVehicle as _IDMVehicle
except ImportError:  # older or modified builds
    _IDMVehicle = None


COURTESY_TYPES = ("cooperative", "non_cooperative")
ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

# IDM/MOBIL parameters injected per courtesy type.
#
# "Courtesy" operationalises a controlled synthetic latent behaviour:
#   cooperative      = larger gaps, longer headway, higher politeness, smoother yielding
#   non_cooperative  = smaller accepted gaps, shorter headway, low politeness, assertive merging
#
# non_cooperative does not mean unrealistically high speed; the separation comes
# mainly from small-gap forcing IDM/MOBIL parameters, not teleportation or direct
# position mutation.
#
# Each entry: target_speed, DISTANCE_WANTED, TIME_WANTED, COMFORT_ACC_MAX,
# POLITENESS — all (lo, hi) ranges, sampled uniformly once per episode.
_COURTESY_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "cooperative": {
        "target_speed":     (24.0, 28.0),
        "DISTANCE_WANTED":  (5.0, 9.0),
        "TIME_WANTED":      (1.4, 2.2),
        "COMFORT_ACC_MAX":  (1.8, 2.8),
        "COMFORT_ACC_MIN":  (-3.0, -1.8),
        "POLITENESS":       (0.35, 0.65),
    },
    "non_cooperative": {
        "target_speed":     (10.0, 16.0),
        "DISTANCE_WANTED":  (1.5, 4.0),
        "TIME_WANTED":      (0.5, 1.0),
        "COMFORT_ACC_MAX":  (3.0, 5.5),
        "COMFORT_ACC_MIN":  (-5.5, -3.0),
        "POLITENESS":       (0.0, 0.2),
    },
}


@dataclass
class GeneratorConfig:
    dataset_name: str = "HiddenCourtesyMerge-Sim"
    environment: str = "merge-v0"
    episodes: int = 200
    max_steps: int = 60
    seed: int = 7
    output: str = "HiddenCourtesyMerge-Sim"
    policy_name: str = "belief_policy"
    traffic_density: Optional[float] = None
    observation_noise_range: Tuple[float, float] = (0.0, 0.35)
    traffic_density_range: Tuple[float, float] = (0.7, 1.1)
    courtesy_prior: Tuple[float, float] = (0.5, 0.5)
    success_min_progress_m: float = 55.0
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    belief_regularization: float = 0.08
    close_call_reward_penalty: float = 0.25
    note: str = (
        "Simulation-generated benchmark dataset for POMDP research. "
        "Not real-world autonomous driving data."
    )


def parse_args() -> GeneratorConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=GeneratorConfig.episodes)
    parser.add_argument("--max-steps", type=int, default=GeneratorConfig.max_steps)
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--output", type=str, default=GeneratorConfig.output)
    parser.add_argument(
        "--policy-name",
        type=str,
        default=GeneratorConfig.policy_name,
        choices=("belief_policy", "rule_policy", "random_policy"),
        help="Ego policy used to generate dataset trajectories.",
    )
    parser.add_argument(
        "--traffic-density",
        type=float,
        default=None,
        help="Use a fixed traffic density for all episodes, useful for ablations.",
    )
    args = parser.parse_args()
    return GeneratorConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        output=args.output,
        policy_name=args.policy_name,
        traffic_density=args.traffic_density,
    )


def dataset_ego_action(
    policy_name: str,
    belief: np.ndarray,
    observed_ttc: float,
    observed_front_gap: float,
    rng: np.random.Generator,
    merge_speed: float = np.nan,
    ego_speed: float = np.nan,
    relative_distance: float = np.nan,
) -> int:
    """Select the data-generating ego action for the requested dataset policy."""
    if policy_name == "belief_policy":
        action_belief = belief
    elif policy_name == "rule_policy":
        action_belief = np.ones(len(COURTESY_TYPES), dtype=float) / len(COURTESY_TYPES)
    elif policy_name == "random_policy":
        return int(rng.integers(0, len(ACTION_NAMES)))
    else:
        raise ValueError(f"Unsupported dataset policy_name={policy_name!r}")
    return choose_ego_action(
        action_belief,
        observed_ttc,
        observed_front_gap,
        rng,
        merge_speed=merge_speed,
        ego_speed=ego_speed,
        relative_distance=relative_distance,
    )


def make_env(seed: int, traffic_density: float) -> gym.Env:
    env = gym.make("merge-v0", render_mode=None)
    config = {
        "duration": 12,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "vehicles_count": int(np.clip(round(5 * traffic_density), 3, 7)),
        "controlled_vehicles": 1,
        "collision_reward": -100,
        "high_speed_reward": 1,
        "merging_speed_reward": -0.5,
        "reward_speed_range": [20, 30],
        "normalize_reward": False,
    }
    env.unwrapped.configure(config)
    env.reset(seed=seed)
    return env


def safe_get_xy_speed(vehicle: Any) -> Tuple[float, float, float]:
    pos = getattr(vehicle, "position", [np.nan, np.nan])
    speed = float(getattr(vehicle, "speed", np.nan))
    return float(pos[0]), float(pos[1]), speed


def lane_index(vehicle: Any) -> int:
    idx = getattr(vehicle, "lane_index", None)
    if isinstance(idx, tuple) and len(idx) >= 3:
        return int(idx[2])
    return -1


def other_vehicles(env: gym.Env) -> List[Any]:
    road = getattr(env.unwrapped, "road", None)
    ego = getattr(env.unwrapped, "vehicle", None)
    if road is None or ego is None:
        return []
    return [v for v in getattr(road, "vehicles", []) if v is not ego]


# highway-env merge-v0 road graph (verified at runtime):
#   'a'->'b' : 2 lanes        main approach
#   'b'->'c' : 3 lanes        post-merge; lane index 2 is the merge lane
#   'c'->'d' : 2 lanes        merge lane closed
#   'j'->'k' : 1 lane         ramp segment 1
#   'k'->'b' : 1 lane         ramp segment 2 (joins the main road at node 'b')
# The merging vehicle is created on the ramp, so at episode reset its lane_index
# from-node is 'j' (occasionally 'k').  After merging it occupies ('b','c',2),
# then ('b','c',1)/('b','c',0).  select_merge_vehicle is called ONCE at reset,
# so the from-node 'j'/'k' test is reliable; the ('b','c',2) and geometric tiers
# are belt-and-braces fallbacks.
_RAMP_FROM_NODES = ("j", "k")
_MERGE_LANE_BC = ("b", "c", 2)


def merge_vehicle_priority(vehicle: Any) -> int:
    """0 = on the ramp (definitive), 1 = on the b->c merge lane, 2 = other."""
    li = getattr(vehicle, "lane_index", None)
    if not isinstance(li, tuple) or len(li) < 3:
        return 2
    if li[0] in _RAMP_FROM_NODES:
        return 0
    if tuple(li) == _MERGE_LANE_BC:
        return 1
    return 2


def select_merge_vehicle(env: gym.Env) -> Optional[Any]:
    """Select the merge-conflict vehicle at episode reset.

    Priority: (1) any IDM vehicle whose lane from-node is 'j'/'k' (on the ramp);
    (2) a vehicle on the ('b','c',2) merge lane; (3) the laterally-offset, nearest
    vehicle as a geometric fallback.  Ties broken by |Δx| to the ego.

    Emits a RuntimeWarning if the selected vehicle is not an IDMVehicle (then
    configure_courtesy_vehicle has no effect) or if no ramp vehicle was found
    (the courtesy manipulation may be applied to the wrong car).
    """
    ego = getattr(env.unwrapped, "vehicle", None)
    if ego is None:
        return None
    ego_x, ego_y, _ = safe_get_xy_speed(ego)

    candidates = []
    for vehicle in other_vehicles(env):
        x, y, _ = safe_get_xy_speed(vehicle)
        if np.isnan(x):
            continue
        prio = merge_vehicle_priority(vehicle)
        # geometric tie-breaker: prefer laterally-offset vehicles (ramp is offset)
        lateral_offset_bonus = 0.0 if (not np.isnan(y) and abs(y - ego_y) > 2.0) else 1.0
        candidates.append((prio, lateral_offset_bonus, abs(x - ego_x), vehicle))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    best_priority, _, _, chosen = candidates[0]

    if _IDMVehicle is not None and not isinstance(chosen, _IDMVehicle):
        warnings.warn(
            f"select_merge_vehicle: chosen vehicle is {type(chosen).__name__}, not IDMVehicle. "
            "Courtesy parameter injection (configure_courtesy_vehicle) will have no effect.",
            RuntimeWarning, stacklevel=2,
        )
    if best_priority == 2:
        warnings.warn(
            "select_merge_vehicle: no vehicle on the ramp ('j'/'k') or merge lane ('b','c',2) "
            "at reset; falling back to a geometric guess. The courtesy manipulation may be "
            "applied to a non-merging vehicle in this episode.",
            RuntimeWarning, stacklevel=2,
        )
    return chosen


def vehicle_id(vehicle: Optional[Any]) -> str:
    if vehicle is None:
        return ""
    return str(getattr(vehicle, "id", id(vehicle)))


def configure_courtesy_vehicle(
    vehicle: Optional[Any], courtesy: str, rng: np.random.Generator
) -> None:
    """
    Inject courtesy-mode parameters into an IDMVehicle instance once at episode start.

    Sets instance attributes that shadow IDMVehicle class attributes, so highway-env's
    standard IDM/MOBIL physics computes behavior with courtesy-dependent parameters.
    No per-step position mutation or act() override is performed.
    """
    if vehicle is None:
        return
    params = _COURTESY_PARAMS[courtesy]

    ts_lo, ts_hi = params["target_speed"]
    vehicle.target_speed = float(rng.uniform(ts_lo, ts_hi))

    if hasattr(vehicle, "DISTANCE_WANTED"):
        lo, hi = params["DISTANCE_WANTED"]
        vehicle.DISTANCE_WANTED = float(rng.uniform(lo, hi))

    if hasattr(vehicle, "TIME_WANTED"):
        lo, hi = params["TIME_WANTED"]
        vehicle.TIME_WANTED = float(rng.uniform(lo, hi))

    if hasattr(vehicle, "COMFORT_ACC_MAX"):
        lo, hi = params["COMFORT_ACC_MAX"]
        vehicle.COMFORT_ACC_MAX = float(rng.uniform(lo, hi))

    if hasattr(vehicle, "COMFORT_ACC_MIN"):
        lo, hi = params["COMFORT_ACC_MIN"]
        vehicle.COMFORT_ACC_MIN = float(rng.uniform(lo, hi))

    if hasattr(vehicle, "POLITENESS"):
        lo, hi = params["POLITENESS"]
        vehicle.POLITENESS = float(rng.uniform(lo, hi))


def compute_gaps(env: gym.Env, ego: Any) -> Tuple[float, float]:
    ego_x, ego_y, _ = safe_get_xy_speed(ego)
    front = math.inf
    rear = math.inf
    for vehicle in other_vehicles(env):
        x, y, _ = safe_get_xy_speed(vehicle)
        if abs(y - ego_y) > 4.5:
            continue
        dx = x - ego_x
        if dx >= 0:
            front = min(front, dx)
        else:
            rear = min(rear, abs(dx))
    return finite_or_nan(front), finite_or_nan(rear)


def finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def estimate_ttc(relative_distance: float, relative_speed: float) -> float:
    if np.isnan(relative_distance) or np.isnan(relative_speed):
        return np.nan
    if relative_distance > 0:
        closing_speed = -relative_speed
    elif relative_distance < 0:
        closing_speed = relative_speed
    else:
        return 0.0
    if closing_speed <= 0.1:
        return np.nan
    return float(abs(relative_distance) / closing_speed)


def noisy(value: float, sigma: float, rng: np.random.Generator) -> float:
    if np.isnan(value):
        return value
    return float(value + rng.normal(0.0, sigma))


def choose_ego_action(
    belief: np.ndarray,
    ttc: float,
    front_gap: float,
    rng: np.random.Generator,
    merge_speed: float = np.nan,
    ego_speed: float = np.nan,
    relative_distance: float = np.nan,
) -> int:
    """Belief-aware but conservative data-generating ego policy.

    The generator deliberately avoids lane changes and FASTER to keep episodes
    long enough for merge negotiation.  It chooses only IDLE or SLOWER.
    """
    b = np.asarray(belief, dtype=float)
    p_coop = float(b[0])

    if not np.isnan(ttc) and ttc < 3.0:
        return int(rng.choice([1, 4], p=[0.02, 0.98]))
    if not np.isnan(front_gap) and front_gap < 8.0:
        return int(rng.choice([1, 4], p=[0.02, 0.98]))
    ttc_caution = 5.0
    # V-shaped: certainty in either direction warrants more caution than uniform prior.
    # Minimum at p_coop=0.5 (rule), maximum at either oracle (p_coop=0 or p_coop=1).
    if not np.isnan(ttc) and 0.0 < ttc < ttc_caution:
        p_slower = float(np.clip(0.735 + 0.185 * abs(2.0 * p_coop - 1.0), 0.735, 0.92))
        return int(rng.choice([1, 4], p=[1.0 - p_slower, p_slower]))

    p_slower = float(np.clip(0.205 + 0.125 * abs(2.0 * p_coop - 1.0), 0.205, 0.33))
    return int(rng.choice([1, 4], p=[1.0 - p_slower, p_slower]))


# ---------------------------------------------------------------------------
# Gaussian observation model  Z(o | courtesy)
#
# Observable features (all about the *merge vehicle*, the only thing whose courtesy
# is being inferred — general front/rear gaps to arbitrary traffic carry no courtesy
# signal and are deliberately excluded):
#   urg  = 1 / TTC to the merge vehicle  ("urgency"); 0 when there is no closing
#          motion.  Bounded and unimodal, unlike raw TTC, which is heavy-tailed and
#          heavy-tailed when there is no closing motion.
#   rs   = merge_speed − ego_speed       (relative speed)
#   mvs  = merge vehicle absolute speed  (directly tracks its target_speed)
#
# Default parameters below are physics-informed priors derived from _COURTESY_PARAMS.
# They are overwritten at runtime by data-driven values when observation_model.json
# is present (generated by calibrate_observation_model.py, which fits these on the
# *interaction window* only — see _INTERACTION_RANGE_M).  Run that script once before
# generating the main dataset.
# ---------------------------------------------------------------------------
_OBS_MODEL: Dict[str, Dict[str, float]] = {
    "cooperative": {
        "urg_mu": 0.455, "urg_sigma": 0.378,
        "ard_mu": 27.93, "ard_sigma": 9.65,
        "rs_mu": -9.66,  "rs_sigma": 4.17,
        "mvs_mu": 10.35, "mvs_sigma": 4.18,
        "mva_mu": -1.157,"mva_sigma": 1.614,
    },
    "non_cooperative": {
        "urg_mu": 0.340, "urg_sigma": 0.313,
        "ard_mu": 24.84, "ard_sigma": 10.30,
        "rs_mu": -5.84,  "rs_sigma": 1.60,
        "mvs_mu": 14.16, "mvs_sigma": 1.60,
        "mva_mu": -0.200,"mva_sigma": 0.783,
    },
}
_OBS_MODEL_LOADED: bool = False
_OBS_MODEL_PAYLOAD: Dict[str, Any] = {}

# The ego only updates courtesy belief while the selected merge vehicle is ahead
# and close enough to observe its merge behaviour.  After the vehicle has passed
# the ego, belief is carried forward unchanged because post-pass dynamics do not
# provide clean negotiation evidence.
_INTERACTION_RANGE_M: float = 40.0
_INTERACTION_NEAR_M: float  = 2.0
_POLICY_DT: float = 0.2  # seconds per policy step (policy_frequency = 5 Hz)

_BELIEF_MODEL: str = "diagonal"
_FULL_COV_MODEL: Dict[str, Dict[str, Any]] = {}


def in_interaction_window(relative_distance: float, mv_speed: float = np.nan) -> bool:
    """True while the selected merge vehicle is ahead in the interaction window."""
    if relative_distance is None or np.isnan(relative_distance):
        return False
    return _INTERACTION_NEAR_M < float(relative_distance) < _INTERACTION_RANGE_M


def ttc_to_urgency(observed_ttc: float, relative_distance: float, relative_speed: float) -> float:
    """Urgency = 1 / TTC.  0 = tracked but not closing; NaN = merge vehicle untracked."""
    if relative_distance is None or relative_speed is None \
            or np.isnan(relative_distance) or np.isnan(relative_speed):
        return np.nan
    if observed_ttc is None or np.isnan(observed_ttc):
        return 0.0
    return 1.0 / max(float(observed_ttc), 1e-3)


def load_obs_model(path: str = "observation_model.json") -> None:
    """Load calibrated Gaussian parameters from *path* into _OBS_MODEL (in-place).

    Safe to call multiple times; subsequent calls are no-ops unless force=True.
    Prints a one-line status message so the caller knows which model is in use.
    """
    global _OBS_MODEL, _OBS_MODEL_LOADED, _OBS_MODEL_PAYLOAD
    if _OBS_MODEL_LOADED:
        return
    _OBS_MODEL_LOADED = True
    p = Path(path)
    if p.exists():
        loaded: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        meta_classes = loaded.get("_meta", {}).get("class_names")
        if meta_classes is not None and list(meta_classes) != list(COURTESY_TYPES):
            print(
                f"[obs_model] Ignoring {p.resolve()} because class_names={meta_classes} "
                f"does not match current classes={list(COURTESY_TYPES)}"
            )
            return
        for mode in COURTESY_TYPES:
            if mode in loaded:
                _OBS_MODEL[mode].update(loaded[mode])
        _OBS_MODEL_PAYLOAD = loaded
        print(f"[obs_model] Loaded calibrated parameters from {p.resolve()}")
    else:
        print(
            "[obs_model] observation_model.json not found — using physics-informed defaults. "
            "Run calibrate_observation_model.py to generate calibrated parameters."
        )


def _gaussian_log_pdf(x: float, mu: float, sigma: float) -> float:
    return -0.5 * ((x - mu) / max(sigma, 1e-6)) ** 2 - math.log(max(sigma, 1e-6))


# (feature_name, (clip_lo, clip_hi), (mu_key, sigma_key))
_OBS_FEATURES = (
    ("ard", (0.0,  40.0), ("ard_mu", "ard_sigma")),   # absolute relative distance to merge vehicle
    ("mvs", (0.0,  35.0), ("mvs_mu", "mvs_sigma")),   # merge vehicle absolute speed
    ("mva", (-8.0,  7.0), ("mva_mu", "mva_sigma")),   # merge vehicle acceleration (m/s²)
)

_FULL_COV_COLUMNS = (
    "observed_abs_relative_distance",
    "observed_merge_speed",
    "observed_merge_acceleration",
)


def fit_full_cov_observation_model(
    steps_with_belief_path: str = "HiddenCourtesyMerge-Sim-cleanobs/steps_with_belief.csv",
) -> Dict[str, Dict[str, Any]]:
    """Fit a full-covariance Gaussian model on train-split in-window steps."""
    path = Path(steps_with_belief_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Full-covariance belief model requires {path}. "
            "Generate the canonical dataset or pass --full-cov-dataset."
        )
    df = pd.read_csv(path)
    required = {"split", "hidden_courtesy", "in_interaction_window", *_FULL_COV_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    train = df[(df["split"] == "train") & (df["in_interaction_window"].astype(bool))]
    train = train.dropna(subset=list(_FULL_COV_COLUMNS))
    model: Dict[str, Dict[str, Any]] = {}
    for mode in COURTESY_TYPES:
        sub = train[train["hidden_courtesy"] == mode][list(_FULL_COV_COLUMNS)].to_numpy(dtype=float)
        if len(sub) < 4:
            raise ValueError(f"Need at least 4 train samples for {mode}, found {len(sub)}")
        mu = sub.mean(axis=0)
        cov = np.cov(sub, rowvar=False)
        cov = np.asarray(cov, dtype=float) + np.eye(len(_FULL_COV_COLUMNS)) * 1e-4
        model[mode] = {
            "mean": mu,
            "cov": cov,
            "n": int(len(sub)),
        }
    return model


def configure_belief_model(
    belief_model: str = "diagonal",
    full_cov_dataset_path: str = "HiddenCourtesyMerge-Sim-cleanobs/steps_with_belief.csv",
) -> None:
    """Select the likelihood used by update_belief()."""
    global _BELIEF_MODEL, _FULL_COV_MODEL
    if belief_model not in {"diagonal", "full_cov"}:
        raise ValueError(f"Unknown belief_model={belief_model!r}")
    _BELIEF_MODEL = belief_model
    if belief_model == "full_cov":
        _FULL_COV_MODEL = fit_full_cov_observation_model(full_cov_dataset_path)
        ns = {mode: payload["n"] for mode, payload in _FULL_COV_MODEL.items()}
        print(f"[belief_model] Using full-covariance Gaussian fitted from {full_cov_dataset_path}: n={ns}")
    else:
        _FULL_COV_MODEL = {}
        print("[belief_model] Using diagonal product-of-Gaussians likelihood")


def _full_cov_log_pdf(values: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    k = int(len(values))
    cov = np.asarray(cov, dtype=float) + np.eye(k) * 1e-6
    diff = np.asarray(values, dtype=float) - np.asarray(mean, dtype=float)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + np.eye(k) * 1e-3
        sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.pinv(cov)
    quad = float(diff.T @ inv @ diff)
    return -0.5 * (k * math.log(2.0 * math.pi) + float(logdet) + quad)


def likelihoods(
    urgency: float,
    abs_relative_distance: float,
    relative_speed: float,
    merge_vehicle_speed: float,
    merge_vehicle_acceleration: float = np.nan,
) -> np.ndarray:
    """Return unnormalised Z(o | courtesy) via a Gaussian product likelihood.

    Features that are NaN (e.g. the merge vehicle is not currently tracked) are
    *omitted* from the product — treated as missing-at-random — rather than imputed
    with a fixed value, which would inject a courtesy-dependent bias.  If every
    feature is missing the likelihood is uniform.
    """
    raw = {
        "urg": urgency,
        "ard": abs_relative_distance,
        "rs":  relative_speed,
        "mvs": merge_vehicle_speed,
        "mva": merge_vehicle_acceleration,
    }
    present = []  # list of (feature_index, clipped_value, mu_key, sigma_key)
    for feat_idx, (name, (lo, hi), (mu_k, sg_k)) in enumerate(_OBS_FEATURES):
        v = raw[name]
        if v is None or np.isnan(v):
            continue
        present.append((feat_idx, float(np.clip(v, lo, hi)), mu_k, sg_k))

    n_modes = len(COURTESY_TYPES)
    if not present:
        return np.ones(n_modes, dtype=float) / n_modes

    log_scores = np.zeros(n_modes, dtype=float)
    if _BELIEF_MODEL == "full_cov":
        feature_positions = [idx for idx, _, _, _ in present]
        values = np.array([val for _, val, _, _ in present], dtype=float)
        for i, mode in enumerate(COURTESY_TYPES):
            payload = _FULL_COV_MODEL[mode]
            mean = np.asarray(payload["mean"], dtype=float)[feature_positions]
            cov = np.asarray(payload["cov"], dtype=float)[np.ix_(feature_positions, feature_positions)]
            log_scores[i] = _full_cov_log_pdf(values, mean, cov)
    else:
        for i, mode in enumerate(COURTESY_TYPES):
            m = _OBS_MODEL[mode]
            log_scores[i] = sum(_gaussian_log_pdf(val, m[mu_k], m[sg_k]) for _, val, mu_k, sg_k in present)
    log_scores -= log_scores.max()  # shift for numerical stability before exp
    return np.maximum(np.exp(log_scores), 1e-6)


def update_belief(
    prior: np.ndarray,
    urgency: float,
    abs_relative_distance: float,
    relative_speed: float,
    merge_vehicle_speed: float,
    merge_vehicle_acceleration: float = np.nan,
    regularization: float = 0.08,
) -> np.ndarray:
    posterior = prior * likelihoods(
        urgency, abs_relative_distance, relative_speed, merge_vehicle_speed, merge_vehicle_acceleration
    )
    posterior = posterior / posterior.sum()
    uniform = np.ones_like(posterior) / len(posterior)
    posterior = (1.0 - regularization) * posterior + regularization * uniform
    return posterior / posterior.sum()


def dataset_split(episode_id: int, total_episodes: int, ratios: Tuple[float, float, float]) -> str:
    train_cut = int(total_episodes * ratios[0])
    val_cut = train_cut + int(total_episodes * ratios[1])
    if episode_id < train_cut:
        return "train"
    if episode_id < val_cut:
        return "validation"
    return "test"


def run_episode(
    episode_id: int,
    seed: int,
    courtesy: str,
    traffic_density: float,
    observation_noise: float,
    config: GeneratorConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    env = make_env(seed, traffic_density)  # make_env already calls env.reset(seed=seed)
    split = dataset_split(episode_id, config.episodes, config.split_ratios)

    # Select merge vehicle once and inject courtesy params via IDM instance attributes.
    merge_vehicle = select_merge_vehicle(env)
    merge_vehicle_id = vehicle_id(merge_vehicle)
    merge_priority = merge_vehicle_priority(merge_vehicle) if merge_vehicle is not None else 99
    configure_courtesy_vehicle(merge_vehicle, courtesy, rng)

    step_rows: List[Dict[str, Any]] = []
    belief_rows: List[Dict[str, Any]] = []
    belief = np.array(config.courtesy_prior, dtype=float)
    total_reward = 0.0
    lane_change_count = 0
    harsh_brake_count = 0
    speeds: List[float] = []
    ttcs: List[float] = []
    critical_ttcs: List[float] = []
    critical_abs_relative_distances: List[float] = []
    critical_speeds: List[float] = []
    previous_lane: Optional[int] = None
    previous_speed: Optional[float] = None
    prev_merge_speed: Optional[float] = None
    collision = False
    initial_ego_x = np.nan
    close_call_count = 0

    for timestep in range(config.max_steps):
        ego = getattr(env.unwrapped, "vehicle", None)

        ego_x, ego_y, ego_speed = safe_get_xy_speed(ego)
        merge_x, merge_y, merge_speed = (
            safe_get_xy_speed(merge_vehicle) if merge_vehicle else (np.nan, np.nan, np.nan)
        )
        # Merge vehicle acceleration via finite difference over policy steps
        if prev_merge_speed is not None and not np.isnan(merge_speed) and not np.isnan(prev_merge_speed):
            mva = (merge_speed - prev_merge_speed) / _POLICY_DT
        else:
            mva = np.nan
        if not np.isnan(merge_speed):
            prev_merge_speed = merge_speed
        ego_lane = lane_index(ego)
        merge_lane = lane_index(merge_vehicle) if merge_vehicle else -1
        if np.isnan(initial_ego_x) and not np.isnan(ego_x):
            initial_ego_x = ego_x

        relative_distance = merge_x - ego_x if not np.isnan(merge_x) else np.nan
        relative_speed = merge_speed - ego_speed if not np.isnan(merge_speed) else np.nan
        estimated = estimate_ttc(relative_distance, relative_speed)
        front_gap, rear_gap = compute_gaps(env, ego)

        observed_ttc = noisy(estimated, observation_noise, rng)
        observed_relative_distance = noisy(relative_distance, observation_noise * 2.0, rng)
        observed_abs_relative_distance = (
            abs(float(observed_relative_distance)) if not np.isnan(observed_relative_distance) else np.nan
        )
        observed_relative_speed = noisy(relative_speed, observation_noise, rng)
        observed_merge_speed = noisy(merge_speed, observation_noise, rng)
        observed_front_gap = noisy(front_gap, observation_noise * 2.0, rng)
        observed_rear_gap = noisy(rear_gap, observation_noise * 2.0, rng)
        observed_merge_acceleration = noisy(mva, observation_noise * 4.0, rng)
        observed_urgency = ttc_to_urgency(observed_ttc, relative_distance, relative_speed)
        urgency = ttc_to_urgency(estimated, relative_distance, relative_speed)
        # The belief only updates while the merge vehicle is close enough to observe
        # its merge behaviour; otherwise it is carried forward unchanged.
        in_window = in_interaction_window(relative_distance, merge_speed)
        if in_window:
            if not np.isnan(relative_distance):
                critical_abs_relative_distances.append(abs(float(relative_distance)))
            if not np.isnan(ego_speed):
                critical_speeds.append(float(ego_speed))
            if not np.isnan(estimated) and not np.isnan(relative_distance) and relative_distance > 0:
                critical_ttcs.append(float(estimated))
        if in_window:
            belief = update_belief(
                belief,
                observed_urgency,
                observed_abs_relative_distance,
                observed_relative_speed,
                observed_merge_speed,
                merge_vehicle_acceleration=observed_merge_acceleration,
                regularization=config.belief_regularization,
            )
        action = dataset_ego_action(
            config.policy_name, belief, observed_ttc, observed_front_gap, rng,
            merge_speed=observed_merge_speed,
            ego_speed=ego_speed,
            relative_distance=relative_distance,
        )

        _, env_reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        collision = collision or bool(getattr(ego, "crashed", False))

        if previous_lane is not None and ego_lane != previous_lane:
            lane_change_count += 1
        if previous_speed is not None and ego_speed - previous_speed < -2.5:
            harsh_brake_count += 1
        previous_lane = ego_lane
        previous_speed = ego_speed
        speeds.append(ego_speed)
        # Only record forward TTC (merge vehicle ahead); rear-end TTC is a different risk.
        if not np.isnan(estimated) and not np.isnan(relative_distance) and relative_distance > 0:
            ttcs.append(estimated)
        close_call = bool(
            (not np.isnan(estimated) and not np.isnan(relative_distance) and relative_distance > 0 and estimated < 3.0)
            or (not np.isnan(relative_distance) and 0 < relative_distance < 15.0)
        )
        close_call_count += int(close_call)
        reward = float(env_reward) - config.close_call_reward_penalty * float(close_call)
        total_reward += reward

        base_row = {
            "episode_id": episode_id,
            "timestep": timestep,
            "seed": seed,
            "policy_name": config.policy_name,
            "split": split,
            "hidden_courtesy": courtesy,
            "ego_x": ego_x,
            "ego_y": ego_y,
            "ego_speed": ego_speed,
            "ego_lane": ego_lane,
            "merge_x": merge_x,
            "merge_y": merge_y,
            "merge_speed": merge_speed,
            "merge_vehicle_id": merge_vehicle_id,
            "merge_vehicle_priority": merge_priority,
            "merge_lane": merge_lane,
            "relative_distance": relative_distance,
            "abs_relative_distance": abs(float(relative_distance)) if not np.isnan(relative_distance) else np.nan,
            "relative_speed": relative_speed,
            "estimated_ttc": estimated,
            "urgency": urgency,
            "front_gap": front_gap,
            "rear_gap": rear_gap,
            "merge_acceleration": mva,
            "in_interaction_window": bool(in_window),
            "action_id": action,
            "action_name": ACTION_NAMES.get(action, f"ACTION_{action}"),
            "reward": float(reward),
            "env_reward": float(env_reward),
            "close_call": bool(close_call),
            "collision": collision,
            "done": done,
        }
        step_rows.append(base_row)
        belief_rows.append(
            {
                **base_row,
                "observed_ttc": observed_ttc,
                "observed_relative_distance": observed_relative_distance,
                "observed_abs_relative_distance": observed_abs_relative_distance,
                "observed_relative_speed": observed_relative_speed,
                "observed_merge_speed": observed_merge_speed,
                "observed_urgency": observed_urgency,
                "observed_front_gap": observed_front_gap,
                "observed_rear_gap": observed_rear_gap,
                "merge_acceleration": mva,
                "observed_merge_acceleration": observed_merge_acceleration,
                "in_interaction_window": bool(in_window),
                "belief_cooperative": float(belief[0]),
                "belief_non_cooperative": float(belief[1]),
            }
        )
        if done:
            break

    final_ego_x = step_rows[-1]["ego_x"] if step_rows else np.nan
    progress_m = (
        final_ego_x - initial_ego_x
        if not np.isnan(final_ego_x) and not np.isnan(initial_ego_x)
        else np.nan
    )
    success = bool(
        not collision and not np.isnan(progress_m) and progress_m >= config.success_min_progress_m
    )
    env.close()

    episode_row = {
        "episode_id": episode_id,
        "seed": seed,
        "policy_name": config.policy_name,
        "split": split,
        "hidden_courtesy": courtesy,
        "traffic_density": traffic_density,
        "observation_noise": observation_noise,
        "collision": collision,
        "success": success,
        "success_definition": (
            f"no collision and ego longitudinal progress >= {config.success_min_progress_m:.1f} m"
        ),
        "min_ttc": float(np.nanmin(ttcs)) if ttcs else np.nan,
        "critical_mean_ttc": float(np.nanmean(critical_ttcs)) if critical_ttcs else np.nan,
        "critical_min_ttc": float(np.nanmin(critical_ttcs)) if critical_ttcs else np.nan,
        "critical_mean_abs_relative_distance": (
            float(np.nanmean(critical_abs_relative_distances))
            if critical_abs_relative_distances else np.nan
        ),
        "mean_speed": float(np.nanmean(speeds)) if speeds else np.nan,
        "ego_speed_std": float(np.nanstd(speeds)) if speeds else np.nan,
        "critical_speed_std": float(np.nanstd(critical_speeds)) if critical_speeds else np.nan,
        "total_reward": float(total_reward),
        "close_call_rate": float(close_call_count / len(step_rows)) if step_rows else np.nan,
        "lane_change_count": lane_change_count,
        "harsh_brake_count": harsh_brake_count,
        "episode_length": len(step_rows),
    }
    return episode_row, step_rows, belief_rows


def validation_summary(
    episodes: pd.DataFrame, steps: pd.DataFrame, steps_with_belief: pd.DataFrame
) -> Dict[str, Any]:
    missing = {
        "episodes": episodes.isna().sum().to_dict(),
        "steps": steps.isna().sum().to_dict(),
        "steps_with_belief": steps_with_belief.isna().sum().to_dict(),
    }
    collision_rate = episodes.groupby("hidden_courtesy")["collision"].mean().to_dict()
    reward_mean = episodes.groupby("hidden_courtesy")["total_reward"].mean().to_dict()
    min_ttc_mean = episodes.groupby("hidden_courtesy")["min_ttc"].mean().to_dict()
    critical_ttc_mean = episodes.groupby("hidden_courtesy")["critical_mean_ttc"].mean().to_dict()
    critical_speed_std_mean = episodes.groupby("hidden_courtesy")["critical_speed_std"].mean().to_dict()
    belief_metrics = compute_belief_metrics(steps_with_belief)
    # Close call: forward TTC < 3 s (merge vehicle ahead only) or very close gap ahead.
    forward_ttc_low = (
        steps["estimated_ttc"].notna()
        & (steps["estimated_ttc"] < 3.0)
        & steps["relative_distance"].notna()
        & (steps["relative_distance"] > 0)
    )
    forward_gap_low = (
        steps["relative_distance"].notna()
        & (steps["relative_distance"] > 0)
        & (steps["relative_distance"] < 15.0)
    )
    close_call = steps.assign(close_call=(forward_ttc_low | forward_gap_low))
    close_call_rate = close_call.groupby("hidden_courtesy")["close_call"].mean().to_dict()
    episode_counts = episodes["hidden_courtesy"].value_counts().to_dict()
    risk_score = {
        courtesy: (
            2.0 * collision_rate.get(courtesy, 0.0)
            + close_call_rate.get(courtesy, 0.0)
            + 0.2 * episodes.loc[episodes["hidden_courtesy"] == courtesy, "harsh_brake_count"].mean()
        )
        for courtesy in COURTESY_TYPES
    }
    non_cooperative_more_risky = {
        "collision_rate_non_cooperative_ge_cooperative": bool(
            collision_rate.get("non_cooperative", np.nan) >= collision_rate.get("cooperative", np.nan)
        ),
        "close_call_rate_non_cooperative_ge_cooperative": bool(
            close_call_rate.get("non_cooperative", np.nan) >= close_call_rate.get("cooperative", np.nan)
        ),
        "mean_min_ttc_non_cooperative_le_cooperative": bool(
            min_ttc_mean.get("non_cooperative", np.nan) <= min_ttc_mean.get("cooperative", np.nan)
        ),
        "mean_reward_non_cooperative_le_cooperative": bool(
            reward_mean.get("non_cooperative", np.nan) <= reward_mean.get("cooperative", np.nan)
        ),
        "risk_score_non_cooperative_gt_cooperative": bool(
            risk_score.get("non_cooperative", np.nan) > risk_score.get("cooperative", np.nan)
        ),
    }
    non_cooperative_more_risky["passed"] = bool(
        all(non_cooperative_more_risky.values())
    )
    return {
        "courtesy_distribution": episodes["hidden_courtesy"].value_counts().to_dict(),
        "episodes_per_courtesy": episode_counts,
        "collision_rate_by_courtesy": collision_rate,
        "close_call_rate_by_courtesy": close_call_rate,
        "reward_by_courtesy": reward_mean,
        "min_ttc_by_courtesy": min_ttc_mean,
        "critical_ttc_by_courtesy": critical_ttc_mean,
        "critical_speed_std_by_courtesy": critical_speed_std_mean,
        "risk_score_by_courtesy": risk_score,
        "belief_validation": belief_metrics,
        "non_cooperative_risk_validation": non_cooperative_more_risky,
        "validation_note": (
            "Use enough episodes per courtesy type before treating risk ordering as evidence; "
            "collision-only validation can be underpowered in merge-v0, so close-call and TTC "
            "metrics are reported alongside collisions."
        ),
        "missing_value_check": missing,
    }


_BELIEF_COLUMNS = ["belief_cooperative", "belief_non_cooperative"]


def expected_calibration_error(
    belief_values: np.ndarray, true_idx: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Dict[str, float]]]:
    """Confidence-ECE for the top-1 predicted class.

    Returns (ece, bins) where bins is a list of per-bin
    {confidence, accuracy, count, lo, hi} for a reliability diagram.
    """
    confidences = belief_values.max(axis=1)
    predicted_idx = belief_values.argmax(axis=1)
    correct = (predicted_idx == true_idx).astype(float)
    n = len(confidences)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: List[Dict[str, float]] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidences > lo) & (confidences <= hi) if b > 0 else (confidences >= lo) & (confidences <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "confidence": None,
                         "accuracy": None, "count": 0})
            continue
        conf_b = float(confidences[mask].mean())
        acc_b = float(correct[mask].mean())
        ece += (cnt / n) * abs(acc_b - conf_b)
        bins.append({"lo": float(lo), "hi": float(hi), "confidence": conf_b,
                     "accuracy": acc_b, "count": cnt})
    return float(ece), bins


def compute_belief_metrics(steps_with_belief: pd.DataFrame) -> Dict[str, Any]:
    nan_result: Dict[str, Any] = {
        k: np.nan for k in [
            "belief_accuracy", "belief_accuracy_all_steps", "final_belief_accuracy",
            "brier_score", "mean_nll", "ece", "uniform_nll",
            "mean_final_belief_cooperative",
            "mean_final_belief_non_cooperative",
        ]
    }
    nan_result["reliability_bins"] = []
    nan_result["n_steps_total"] = 0
    nan_result["n_steps_in_window"] = 0
    if steps_with_belief.empty or any(c not in steps_with_belief for c in _BELIEF_COLUMNS):
        return nan_result

    labels = np.array(COURTESY_TYPES)
    all_belief = steps_with_belief[_BELIEF_COLUMNS].to_numpy()
    all_true   = steps_with_belief["hidden_courtesy"].to_numpy()
    acc_all = float(np.mean(labels[np.argmax(all_belief, axis=1)] == all_true))

    # Headline metrics are over the interaction window only: before the merge vehicle
    # comes close the belief is frozen at the uniform prior, so including those steps
    # would just dilute everything toward chance.  Fall back to all steps if the
    # column is absent (older datasets).
    if "in_interaction_window" in steps_with_belief.columns:
        sub = steps_with_belief[steps_with_belief["in_interaction_window"].astype(bool)]
        if sub.empty:
            sub = steps_with_belief
    else:
        sub = steps_with_belief

    belief_values = sub[_BELIEF_COLUMNS].to_numpy()
    true_labels = sub["hidden_courtesy"].to_numpy()
    true_idx = np.array([list(COURTESY_TYPES).index(c) for c in true_labels])
    one_hot = (true_labels[:, None] == labels[None, :]).astype(float)

    belief_accuracy = float(np.mean(labels[np.argmax(belief_values, axis=1)] == true_labels))
    brier_score = float(np.mean(np.sum((belief_values - one_hot) ** 2, axis=1)))
    true_probs = belief_values[np.arange(len(belief_values)), true_idx]
    mean_nll = float(-np.mean(np.log(np.maximum(true_probs, 1e-9))))
    ece, reliability_bins = expected_calibration_error(belief_values, true_idx)

    final_rows = (
        steps_with_belief.sort_values(["episode_id", "timestep"])
        .groupby("episode_id", as_index=False)
        .tail(1)
    )
    final_belief_values = final_rows[_BELIEF_COLUMNS].to_numpy()
    final_predicted = labels[np.argmax(final_belief_values, axis=1)]
    final_belief_accuracy = float(np.mean(final_predicted == final_rows["hidden_courtesy"].to_numpy()))

    return {
        "belief_accuracy": belief_accuracy,                # in-window steps
        "belief_accuracy_all_steps": acc_all,              # incl. frozen-prior steps
        "final_belief_accuracy": final_belief_accuracy,
        "brier_score": brier_score,
        "mean_nll": mean_nll,
        "uniform_nll": float(math.log(len(COURTESY_TYPES))),
        "ece": ece,
        "reliability_bins": reliability_bins,
        "n_steps_total": int(len(steps_with_belief)),
        "n_steps_in_window": int(len(sub)),
        "mean_final_belief_cooperative": float(final_rows["belief_cooperative"].mean()),
        "mean_final_belief_non_cooperative": float(final_rows["belief_non_cooperative"].mean()),
    }


def belief_quality_over_time(steps_with_belief: pd.DataFrame) -> pd.DataFrame:
    """Per-timestep mean Brier score, accuracy, and belief-in-true-class."""
    if steps_with_belief.empty or any(c not in steps_with_belief for c in _BELIEF_COLUMNS):
        return pd.DataFrame()
    labels = np.array(COURTESY_TYPES)
    rows: List[Dict[str, float]] = []
    for ts, grp in steps_with_belief.groupby("timestep"):
        bv = grp[_BELIEF_COLUMNS].to_numpy()
        tl = grp["hidden_courtesy"].to_numpy()
        ti = np.array([list(COURTESY_TYPES).index(c) for c in tl])
        oh = (tl[:, None] == labels[None, :]).astype(float)
        rows.append({
            "timestep": int(ts),
            "n": int(len(grp)),
            "mean_brier": float(np.mean(np.sum((bv - oh) ** 2, axis=1))),
            "accuracy": float(np.mean(labels[np.argmax(bv, axis=1)] == tl)),
            "mean_belief_true_class": float(np.mean(bv[np.arange(len(bv)), ti])),
        })
    return pd.DataFrame(rows).sort_values("timestep")


def write_readme(output_dir: Path, config: GeneratorConfig, summary: Dict[str, Any]) -> None:
    readme = f"""# {config.dataset_name}

This folder contains a simulation-generated benchmark dataset for POMDP research
on merge negotiation under hidden driver courtesy. Generated with `highway-env`
`merge-v0`. **Not real-world autonomous driving data.**

Courtesy type is a controlled synthetic latent variable that modifies the merge
vehicle's IDM/MOBIL parameters (target_speed, DISTANCE_WANTED, TIME_WANTED,
COMFORT_ACC_MAX, COMFORT_ACC_MIN, POLITENESS) once at episode start. No teleportation is used;
vehicles evolve under standard highway-env physics.

## Files

- `episodes.csv`: one row per episode with hidden courtesy labels and outcomes.
- `steps.csv`: raw step-level motion, action, reward, and termination data.
- `reward` is the task reward used for analysis: highway-env reward minus a small
  close-call penalty. `env_reward` preserves the raw simulator reward.
- `steps_with_belief.csv`: step data plus `belief_cooperative` and `belief_non_cooperative`.
- `observation_model.json`: copy of the calibrated Gaussian observation model used.
- `config.json`: generation settings and validation summary.
- `figures/`: dataset sanity-check visualizations.

## Splits

Each row contains a `split` column (`train`, `validation`, `test`) assigned
deterministically by episode_id with ratios {config.split_ratios}. Do not use
`hidden_courtesy` as a feature; it is the label to predict or condition on.

## Merge Vehicle Identity

`merge_vehicle_id` is fixed for the entire episode. The same vehicle is tracked
throughout; it does not switch identity between timesteps.

## Belief Update

The Bayesian belief over `hidden_courtesy` is regularized toward the uniform prior
with weight `belief_regularization={config.belief_regularization}` to prevent
unrealistically sharp collapse from correlated observations.

The target is binary: `cooperative` versus `non_cooperative`. This is a
controlled synthetic benchmark, not realistic human-driver classification.

## Success Definition

`success=true`: no collision AND ego longitudinal progress >= {config.success_min_progress_m:.1f} m.

## Validation Summary

```json
{json.dumps(summary, indent=2)}
```
"""
    output_dir.joinpath("README.md").write_text(readme, encoding="utf-8")


def save_figures(episodes: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    plot_bar(
        episodes["hidden_courtesy"].value_counts().reindex(COURTESY_TYPES).fillna(0),
        "Courtesy Distribution", "Courtesy", "Episodes",
        figures_dir / "courtesy_distribution.png",
    )
    plot_bar(
        episodes.groupby("hidden_courtesy")["collision"].mean().reindex(COURTESY_TYPES).fillna(0),
        "Collision Rate by Courtesy", "Courtesy", "Collision Rate",
        figures_dir / "collision_rate.png",
    )
    plot_bar(
        episodes.groupby("hidden_courtesy")["total_reward"].mean().reindex(COURTESY_TYPES).fillna(0),
        "Average Reward by Courtesy", "Courtesy", "Average Total Reward",
        figures_dir / "average_reward.png",
    )
    plot_bar(
        episodes.groupby("hidden_courtesy")["min_ttc"].mean().reindex(COURTESY_TYPES).fillna(0),
        "Average Minimum TTC by Courtesy", "Courtesy", "Minimum TTC (s)",
        figures_dir / "minimum_ttc.png",
    )


def plot_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2c7fb8", "#7fcdbb", "#f03b20"]
    ax.bar(series.index.astype(str), series.values, color=colors[: len(series)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_belief_calibration_figures(
    steps_with_belief: pd.DataFrame, reliability_bins: List[Dict[str, float]], output_dir: Path
) -> None:
    """Reliability diagram + Brier/accuracy/belief-in-true-class versus timestep."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reliability diagram (confidence-ECE bins)
    if reliability_bins:
        confs = [b["confidence"] for b in reliability_bins if b["count"] > 0]
        accs  = [b["accuracy"]   for b in reliability_bins if b["count"] > 0]
        cnts  = [b["count"]      for b in reliability_bins if b["count"] > 0]
        if confs:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
            sizes = 40 + 360 * np.array(cnts, dtype=float) / max(cnts)
            ax.scatter(confs, accs, s=sizes, color="#2c7fb8", alpha=0.8, edgecolors="k", label="bins (size ∝ count)")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted confidence (top-1)")
            ax.set_ylabel("Empirical accuracy")
            ax.set_title("Belief reliability diagram")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(figures_dir / "belief_reliability_diagram.png", dpi=160)
            plt.close(fig)

    # Quality over time
    quality = belief_quality_over_time(steps_with_belief)
    if not quality.empty:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(quality["timestep"], quality["mean_belief_true_class"],
                 color="#2c7fb8", linewidth=1.8, label="mean belief in true class")
        ax1.plot(quality["timestep"], quality["accuracy"],
                 color="#31a354", linewidth=1.8, label="argmax accuracy")
        ax1.axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="chance")
        ax1.set_ylim(0, 1)
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Belief in true class / accuracy")
        ax2 = ax1.twinx()
        ax2.plot(quality["timestep"], quality["mean_brier"],
                 color="#f03b20", linewidth=1.5, linestyle=":", label="mean Brier score")
        ax2.set_ylabel("Mean Brier score")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
        ax1.set_title("Belief quality over time")
        fig.tight_layout()
        fig.savefig(figures_dir / "belief_quality_over_time.png", dpi=160)
        plt.close(fig)


def generate_dataset(config: GeneratorConfig) -> None:
    load_obs_model()
    random.seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    belief_rows: List[Dict[str, Any]] = []
    courtesy_schedule = make_courtesy_schedule(config.episodes, rng)

    for episode_id in range(config.episodes):
        seed = config.seed + episode_id
        courtesy = courtesy_schedule[episode_id]
        traffic_density = (
            float(config.traffic_density)
            if config.traffic_density is not None
            else float(rng.uniform(*config.traffic_density_range))
        )
        observation_noise = float(rng.uniform(*config.observation_noise_range))
        episode, steps, beliefs = run_episode(
            episode_id=episode_id,
            seed=seed,
            courtesy=courtesy,
            traffic_density=traffic_density,
            observation_noise=observation_noise,
            config=config,
        )
        episode_rows.append(episode)
        step_rows.extend(steps)
        belief_rows.extend(beliefs)

    episodes = pd.DataFrame(episode_rows)
    steps = pd.DataFrame(step_rows)
    steps_with_belief = pd.DataFrame(belief_rows)

    episodes.to_csv(output_dir / "episodes.csv", index=False)
    steps.to_csv(output_dir / "steps.csv", index=False)
    steps_with_belief.to_csv(output_dir / "steps_with_belief.csv", index=False)

    quality_over_time = belief_quality_over_time(steps_with_belief)
    if not quality_over_time.empty:
        quality_over_time.to_csv(output_dir / "belief_quality_over_time.csv", index=False)

    summary = validation_summary(episodes, steps, steps_with_belief)
    obs_payload = _OBS_MODEL_PAYLOAD if _OBS_MODEL_PAYLOAD else _OBS_MODEL
    output_dir.joinpath("observation_model.json").write_text(json.dumps(obs_payload, indent=2), encoding="utf-8")
    config_payload = {
        **asdict(config),
        "observation_features": [name for name, _, _ in _OBS_FEATURES],
        "observation_model_used": _OBS_MODEL,
        "observation_model_source": "observation_model.json" if _OBS_MODEL_LOADED and Path("observation_model.json").exists() else "physics_informed_defaults",
        "interaction_range_m": _INTERACTION_RANGE_M,
        "highway_env_version": getattr(highway_env, "__version__", "unknown"),
        "validation_summary": summary,
    }
    output_dir.joinpath("config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    write_readme(output_dir, config, summary)
    save_figures(episodes, output_dir)
    reliability_bins = summary.get("belief_validation", {}).get("reliability_bins", [])
    save_belief_calibration_figures(steps_with_belief, reliability_bins, output_dir)

    print(f"Generated {config.dataset_name} at {output_dir.resolve()}")
    print(json.dumps(summary, indent=2))


def make_courtesy_schedule(episodes: int, rng: np.random.Generator) -> List[str]:
    schedule: List[str] = []
    while len(schedule) < episodes:
        block = list(COURTESY_TYPES)
        rng.shuffle(block)
        schedule.extend(block)
    return schedule[:episodes]


def main() -> None:
    config = parse_args()
    generate_dataset(config)


if __name__ == "__main__":
    main()
