"""
Formal POMDP model helpers for HiddenCourtesyMerge-Sim.

M = (S, A, O, T, Z, R, gamma, b0)

Hidden parameter: merge vehicle courtesy type in
{cooperative, non_cooperative}.  The simulator implements T; this module
contains only the model-facing constants, observation likelihood, belief update,
interaction-window rule, and task reward interface.  It intentionally has no
highway-env imports.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Optional, Tuple

import numpy as np


COURTESY_TYPES: Tuple[str, ...] = ("cooperative", "non_cooperative")
N_CLASSES: int = len(COURTESY_TYPES)

ACTION_NAMES: Dict[int, str] = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}
N_ACTIONS: int = len(ACTION_NAMES)

OBSERVATION_FEATURES: Tuple[str, ...] = ("ard", "mvs", "mva")

INTERACTION_FAR_M: float = 40.0
INTERACTION_NEAR_M: float = 2.0

BELIEF_LAMBDA: float = 0.08

CLOSE_CALL_TTC_THRESHOLD: float = 3.0
CLOSE_CALL_PENALTY: float = 0.25


def load_observation_model(path: str = "observation_model.json") -> Dict[str, Dict[str, float]]:
    """Load calibrated Gaussian parameters from JSON."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    model: Dict[str, Dict[str, float]] = {}
    for mode in COURTESY_TYPES:
        if mode not in raw:
            raise ValueError(f"Observation model missing class {mode!r}")
        model[mode] = {k: float(v) for k, v in raw[mode].items() if isinstance(v, (int, float))}
    return model


def _gaussian_log_likelihood(x: float, mu: float, sigma: float) -> float:
    """Return log N(x | mu, sigma)."""
    sigma = max(float(sigma), 1e-6)
    z = (float(x) - float(mu)) / sigma
    return -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * z * z


def observation_likelihood(
    obs: Dict[str, Optional[float]],
    courtesy: str,
    obs_model: Dict[str, Dict[str, float]],
) -> float:
    """Product-of-Gaussians likelihood for one courtesy class.

    Missing features, represented by None or NaN, are skipped rather than
    imputed.  If every feature is missing, the likelihood is neutral: 1.0.
    """
    if courtesy not in COURTESY_TYPES:
        raise ValueError(f"Unknown courtesy class {courtesy!r}")

    params = obs_model[courtesy]
    log_lik = 0.0
    any_valid = False

    for feat in OBSERVATION_FEATURES:
        value = obs.get(feat)
        if value is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        log_lik += _gaussian_log_likelihood(
            value,
            params[f"{feat}_mu"],
            params[f"{feat}_sigma"],
        )
        any_valid = True

    return math.exp(log_lik) if any_valid else 1.0


def normalize_belief(belief: np.ndarray) -> np.ndarray:
    """Normalize a belief vector, falling back to uniform if degenerate."""
    b = np.asarray(belief, dtype=float)
    total = float(np.sum(b))
    if not np.isfinite(total) or total < 1e-12:
        return np.ones(N_CLASSES, dtype=float) / N_CLASSES
    return b / total


def update_belief(
    prior: np.ndarray,
    obs: Dict[str, Optional[float]],
    obs_model: Dict[str, Dict[str, float]],
    lambda_reg: float = BELIEF_LAMBDA,
) -> np.ndarray:
    """Bayesian belief update with uniform regularization.

    posterior(m) is proportional to prior(m) * Z(obs | m).  The result is then
    shrunk toward the uniform belief by lambda_reg to reduce overconfidence from
    temporally correlated observations.
    """
    b = normalize_belief(np.asarray(prior, dtype=float).copy())
    for i, mode in enumerate(COURTESY_TYPES):
        b[i] *= observation_likelihood(obs, mode, obs_model)

    b = normalize_belief(b)
    uniform = np.ones(N_CLASSES, dtype=float) / N_CLASSES
    b = (1.0 - float(lambda_reg)) * b + float(lambda_reg) * uniform
    b = normalize_belief(b)

    if abs(float(np.sum(b)) - 1.0) >= 1e-6:
        raise AssertionError(f"Belief does not sum to 1: {np.sum(b)}")
    return b


def initial_belief() -> np.ndarray:
    """Return the uniform initial belief b0 = [0.5, 0.5]."""
    return np.ones(N_CLASSES, dtype=float) / N_CLASSES


def is_interaction_window(relative_distance: float, merge_ahead: bool = True) -> bool:
    """Return true when the merge vehicle is in the belief-update window."""
    if not merge_ahead:
        return False
    if relative_distance is None:
        return False
    rd = float(relative_distance)
    if math.isnan(rd):
        return False
    return INTERACTION_NEAR_M < rd < INTERACTION_FAR_M


def compute_urgency(ttc: float) -> float:
    """Return urgency = 1 / TTC clipped to [0, 1.5]."""
    ttc = float(ttc)
    if math.isnan(ttc) or math.isinf(ttc) or ttc <= 0.0:
        return 0.0
    return float(min(1.0 / ttc, 1.5))


def _optional_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def extract_observation(
    urg: float,
    ard: float,
    rs: float,
    mvs: float,
    mva: Optional[float],
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Optional[float]]:
    """Package raw features into an observation dictionary."""
    obs: Dict[str, Optional[float]] = {
        "urg": _optional_float(urg),
        "ard": _optional_float(ard),
        "rs": _optional_float(rs),
        "mvs": _optional_float(mvs),
        "mva": _optional_float(mva),
    }

    if noise_scale > 0.0 and rng is not None:
        for feat in OBSERVATION_FEATURES:
            value = obs[feat]
            if value is not None:
                obs[feat] = float(value * (1.0 + rng.normal(0.0, noise_scale)))
    return obs


def compute_task_reward(env_reward: float, min_ttc: float) -> float:
    """Return task reward: env_reward minus close-call penalty."""
    ttc = float(min_ttc)
    close_call = (not math.isnan(ttc)) and ttc < CLOSE_CALL_TTC_THRESHOLD
    return float(env_reward) - (CLOSE_CALL_PENALTY if close_call else 0.0)


def _self_test() -> None:
    """Run quick sanity checks."""
    b0 = initial_belief()
    assert abs(float(np.sum(b0)) - 1.0) < 1e-9
    print(f"Initial belief: {b0} sum={np.sum(b0):.6f} OK")

    obs_model_dummy = {
        "cooperative": {
            "urg_mu": 0.48,
            "urg_sigma": 0.37,
            "ard_mu": 27.5,
            "ard_sigma": 9.0,
            "rs_mu": -10.4,
            "rs_sigma": 3.9,
            "mvs_mu": 9.6,
            "mvs_sigma": 3.9,
            "mva_mu": -0.94,
            "mva_sigma": 1.70,
        },
        "non_cooperative": {
            "urg_mu": 0.43,
            "urg_sigma": 0.36,
            "ard_mu": 22.8,
            "ard_sigma": 10.6,
            "rs_mu": -6.77,
            "rs_sigma": 1.69,
            "mvs_mu": 13.2,
            "mvs_sigma": 1.68,
            "mva_mu": -0.04,
            "mva_sigma": 0.50,
        },
    }
    obs_nc = {"urg": 0.3, "ard": 20.0, "rs": -5.0, "mvs": 15.0, "mva": 0.0}
    b1 = update_belief(b0, obs_nc, obs_model_dummy)
    assert abs(float(np.sum(b1)) - 1.0) < 1e-6
    assert b1[1] > b1[0], f"Expected non_cooperative posterior to dominate: {b1}"
    print(f"After non_cooperative observation: {b1} sum={np.sum(b1):.6f} OK")

    obs_nan = {"urg": float("nan"), "ard": float("nan"), "rs": float("nan"), "mvs": float("nan"), "mva": None}
    lik = observation_likelihood(obs_nan, "cooperative", obs_model_dummy)
    assert lik == 1.0, f"All-missing likelihood should be 1.0, got {lik}"
    print("All-missing likelihood check OK")

    assert is_interaction_window(20.0)
    assert not is_interaction_window(1.0)
    assert not is_interaction_window(41.0)
    assert not is_interaction_window(-10.0)
    print("Interaction window checks OK")

    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
