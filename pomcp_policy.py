"""
POMCP (Partially Observable Monte Carlo Planning) solver for HiddenCourtesyMerge-Sim.

The hidden variable m ∈ {cooperative, non_cooperative} is fixed per episode
(IDM/MOBIL parameters injected once at reset, never changed).  This makes the
POMDP a hidden-parameter MDP: the particle filter never needs to propagate
particles through a mode-transition kernel — only the observation likelihood
reweights them.  The belief b_coop is therefore a sufficient statistic.

Architecture:
    GenerativeModel     – simplified analytical IDM rollout per mode
    UCTNode             – node in the action tree with N(a), Q(a) statistics
    POMCPSolver         – UCT search over (action, mode-sample) pairs
    POMCPPolicy         – stateful per-episode wrapper for evaluate_policies.py

Usage (standalone):
    solver = POMCPPolicy.build("observation_model.json")
    solver.reset()
    action = solver.plan(rel_dist=12.0, ego_speed=25.0, merge_speed=13.0)
    solver.observe({'ard': 12.0, 'mvs': 13.1, 'mva': -0.1})

Integration with evaluate_policies.py:
    Pass a POMCPPolicy instance as pomcp_instance= kwarg to select_action().
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# IDM parameter means per courtesy type (mid-points of _COURTESY_PARAMS ranges)
# ---------------------------------------------------------------------------

_IDM_MEANS: Dict[str, Dict[str, float]] = {
    "cooperative": {
        "target_speed":    26.0,   # (24+28)/2
        "distance_wanted":  7.0,   # (5+9)/2
        "time_wanted":      1.8,   # (1.4+2.2)/2
        "acc_max":          2.3,   # (1.8+2.8)/2
        "acc_min":         -2.4,   # mean |COMFORT_ACC_MIN|
    },
    "non_cooperative": {
        "target_speed":    13.0,   # (10+16)/2
        "distance_wanted":  2.75,  # (1.5+4.0)/2
        "time_wanted":      0.75,  # (0.5+1.0)/2
        "acc_max":          4.25,  # (3.0+5.5)/2
        "acc_min":         -4.25,
    },
}

_DT        = 0.2     # policy step duration (s)
_EGO_CRUISE = 25.0   # IDLE target speed (m/s)
_EGO_DEC    = 3.0    # SLOWER deceleration (m/s²)
_EGO_ACC    = 2.5    # FASTER acceleration  (m/s²)
_COLLISION_D = 1.5   # (m) threshold for counting state as collision in model
_CLOSE_CALL_TTC = 3.0
_CLOSE_CALL_PENALTY = 1.0
_COLLISION_PENALTY  = 100.0
_IDLE_IDX    = 1
_SLOWER_IDX  = 4


# ---------------------------------------------------------------------------
# Generative model
# ---------------------------------------------------------------------------

class GenerativeModel:
    """
    Analytical IDM rollout for POMCP simulations.

    State: (d, v_e, v_m)
        d   = x_m - x_e  (positive when MV is ahead, interaction window: 2–40 m)
        v_e = ego speed (m/s)
        v_m = merge vehicle speed (m/s)

    MV dynamics: free-flow IDM toward target speed (ego is behind MV, not
    acting as MV's lead vehicle — after the merge, MV follows main-road traffic).
    Ego dynamics: deterministic response to discrete action.
    """

    def __init__(self, obs_params: Dict):
        self._obs = obs_params  # {mode: {feat_mu, feat_sigma, ...}}

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    @staticmethod
    def _ego_step(v_e: float, action: int) -> float:
        if action == _SLOWER_IDX:
            return float(np.clip(v_e - _EGO_DEC * _DT, 0.0, 40.0))
        if action == 3:  # FASTER
            return float(np.clip(v_e + _EGO_ACC * _DT, 0.0, 40.0))
        # IDLE / LANE_LEFT / LANE_RIGHT — coast toward cruise speed
        return float(np.clip(v_e + 0.6 * (_EGO_CRUISE - v_e) * _DT, 0.0, 40.0))

    @staticmethod
    def _mv_step(v_m: float, mode: str) -> float:
        p = _IDM_MEANS[mode]
        # Free-flow acceleration toward target speed, bounded by comfort limits
        a = float(np.clip(
            0.5 * (p["target_speed"] - v_m),  # proportional approach
            p["acc_min"], p["acc_max"],
        ))
        return float(np.clip(v_m + a * _DT, 0.0, 40.0))

    def step(
        self, d: float, v_e: float, v_m: float, action: int, mode: str
    ) -> Tuple[float, float, float]:
        v_e2 = self._ego_step(v_e, action)
        v_m2 = self._mv_step(v_m, mode)
        d2   = d + (v_m - v_e) * _DT   # dd/dt = v_m − v_e
        return d2, v_e2, v_m2

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    @staticmethod
    def reward(d: float, v_e: float, v_m: float) -> float:
        r = v_e / 30.0  # speed reward (≈ high_speed_reward scale)
        if d < _COLLISION_D:
            return r - _COLLISION_PENALTY
        if v_e > v_m and d > 0.0:
            ttc = d / max(v_e - v_m, 1e-3)
            if ttc < _CLOSE_CALL_TTC:
                r -= _CLOSE_CALL_PENALTY
        return r

    @staticmethod
    def is_collision(d: float) -> bool:
        return d < _COLLISION_D

    # ------------------------------------------------------------------
    # Observation sampling (for belief update inside POMCP rollouts)
    # ------------------------------------------------------------------

    def sample_obs(
        self, d: float, v_m: float, prev_v_m: float, rng: np.random.Generator
    ) -> Dict[str, float]:
        mva = (v_m - prev_v_m) / _DT
        return {
            "ard": float(abs(d)   + rng.normal(0.0, 1.5)),
            "mvs": float(v_m      + rng.normal(0.0, 0.5)),
            "mva": float(mva      + rng.normal(0.0, 0.5)),
        }


# ---------------------------------------------------------------------------
# UCT node
# ---------------------------------------------------------------------------

class UCTNode:
    __slots__ = ("N", "Q", "children")

    def __init__(self) -> None:
        self.N: Dict[int, int]         = {}
        self.Q: Dict[int, float]       = {}
        self.children: Dict[int, "UCTNode"] = {}

    def ucb_score(self, action: int, c: float) -> float:
        if action not in self.N or self.N[action] == 0:
            return float("inf")
        n_tot = sum(self.N.values())
        return self.Q[action] + c * math.sqrt(math.log(max(n_tot, 1)) / self.N[action])

    def select(self, actions: List[int], c: float) -> int:
        return max(actions, key=lambda a: self.ucb_score(a, c))

    def update(self, action: int, value: float) -> None:
        n = self.N.get(action, 0) + 1
        self.N[action] = n
        self.Q[action] = self.Q.get(action, 0.0) + (value - self.Q.get(action, 0.0)) / n

    def best_action(self, actions: List[int]) -> int:
        return max(actions, key=lambda a: self.Q.get(a, -float("inf")))


# ---------------------------------------------------------------------------
# POMCP solver
# ---------------------------------------------------------------------------

class POMCPSolver:
    """
    UCT search for the binary-mode hidden-parameter POMDP.

    Because m is fixed per episode, each simulation samples m from the current
    belief and treats it as known for the entire simulated trajectory — no
    particle-filter propagation is needed across the hidden dimension.

    Planning actions are restricted to {IDLE=1, SLOWER=4} to match the
    existing heuristic policies' effective action space.  The full 5-action
    space is available by setting actions=[0,1,2,3,4].
    """

    ACTIONS = [_IDLE_IDX, _SLOWER_IDX]

    def __init__(
        self,
        gen: GenerativeModel,
        obs_model: Dict,
        n_simulations: int = 100,
        horizon: int = 10,
        c_uct: float = 1.41,
        gamma: float = 0.98,
        rollout_depth: int = 5,
        rollout_d_thresh: float = 15.0,
        rollout_ttc_thresh: float = 6.0,
    ) -> None:
        self.gen                = gen
        self.obs_model          = obs_model   # {mode: {feat_mu, feat_sigma}}
        self.n_simulations      = n_simulations
        self.horizon            = horizon
        self.c_uct              = c_uct
        self.gamma              = gamma
        self.rollout_depth      = rollout_depth
        self.rollout_d_thresh   = rollout_d_thresh
        self.rollout_ttc_thresh = rollout_ttc_thresh

        self._rng  = np.random.default_rng(0)
        self._root = UCTNode()

    def seed(self, s: int) -> None:
        self._rng = np.random.default_rng(s)

    def reset_tree(self) -> None:
        self._root = UCTNode()

    # ------------------------------------------------------------------
    # Belief helpers
    # ------------------------------------------------------------------

    def _sample_mode(self, belief: np.ndarray) -> str:
        return "cooperative" if self._rng.random() < float(belief[0]) else "non_cooperative"

    def _belief_update(
        self, belief: np.ndarray, obs: Dict[str, float]
    ) -> np.ndarray:
        from pomdp_model import update_belief, COURTESY_TYPES, BELIEF_LAMBDA
        return update_belief(belief, obs, self.obs_model, BELIEF_LAMBDA)

    # ------------------------------------------------------------------
    # UCT simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        node: UCTNode,
        d: float, v_e: float, v_m: float,
        belief: np.ndarray,
        mode: str,
        depth: int,
    ) -> float:
        if depth == 0 or self.gen.is_collision(d):
            return 0.0

        # Action selection (UCB1)
        unexplored = [a for a in self.ACTIONS if self.N_a(node, a) == 0]
        action = self._rng.choice(unexplored) if unexplored else node.select(self.ACTIONS, self.c_uct)

        # Generative step
        prev_v_m = v_m
        d2, v_e2, v_m2 = self.gen.step(d, v_e, v_m, action, mode)
        r = self.gen.reward(d2, v_e2, v_m2)

        # Belief update with sampled observation (only if in interaction window)
        b2 = belief
        if 2.0 < d2 < 40.0:
            obs = self.gen.sample_obs(d2, v_m2, prev_v_m, self._rng)
            b2  = self._belief_update(belief, obs)

        # Recurse or rollout
        if action not in node.children:
            node.children[action] = UCTNode()
        child = node.children[action]

        if depth <= self.rollout_depth:
            future = self._rollout(d2, v_e2, v_m2, mode, depth - 1)
        else:
            future = self._simulate(child, d2, v_e2, v_m2, b2, mode, depth - 1)

        value = r + self.gamma * future
        node.update(action, value)
        return value

    def _rollout(
        self, d: float, v_e: float, v_m: float, mode: str, steps: int
    ) -> float:
        total, discount = 0.0, 1.0
        for _ in range(steps):
            if self.gen.is_collision(d):
                break
            # Rollout heuristic: SLOWER when within d/TTC thresholds, IDLE otherwise.
            ttc = d / max(v_e - v_m, 1e-3) if v_e > v_m and d > 0 else float("inf")
            action = _SLOWER_IDX if (d < self.rollout_d_thresh or ttc < self.rollout_ttc_thresh) else _IDLE_IDX
            d, v_e, v_m = self.gen.step(d, v_e, v_m, action, mode)
            total    += discount * self.gen.reward(d, v_e, v_m)
            discount *= self.gamma
        return total

    @staticmethod
    def N_a(node: UCTNode, action: int) -> int:
        return node.N.get(action, 0)

    # ------------------------------------------------------------------
    # Public planning interface
    # ------------------------------------------------------------------

    def plan(
        self, d: float, v_e: float, v_m: float, belief: np.ndarray
    ) -> int:
        """Run POMCP and return best action index."""
        for _ in range(self.n_simulations):
            mode = self._sample_mode(belief)
            self._simulate(self._root, d, v_e, v_m, belief, mode, self.horizon)

        if not self._root.Q:
            return _IDLE_IDX  # no simulations ran — safe default

        best = self._root.best_action(self.ACTIONS)

        # Tree reuse: advance root to child of chosen action
        self._root = self._root.children.get(best, UCTNode())
        return best


# ---------------------------------------------------------------------------
# Per-episode stateful policy wrapper
# ---------------------------------------------------------------------------

class POMCPPolicy:
    """
    Stateful wrapper used by evaluate_policies.py.

    One instance per policy evaluation run; reset() is called at the start
    of each episode to clear the search tree and belief.

    The belief is maintained externally by evaluate_policies (using the same
    Bayesian filter as belief_policy) and passed into plan() each timestep.
    POMCP uses this externally-maintained belief so that the comparison is
    fair: the same observation model and regularisation apply to all policies.
    """

    def __init__(self, solver: POMCPSolver) -> None:
        self._solver = solver

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        obs_model_path: str = "observation_model.json",
        n_simulations: int = 100,
        horizon: int = 10,
        c_uct: float = 1.41,
        gamma: float = 0.98,
        rollout_depth: int = 5,
        rollout_d_thresh: float = 15.0,
        rollout_ttc_thresh: float = 6.0,
    ) -> "POMCPPolicy":
        raw = json.loads(Path(obs_model_path).read_text(encoding="utf-8"))
        obs_model = {k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
                     for k, v in raw.items() if isinstance(v, dict)}
        gen    = GenerativeModel(obs_model)
        solver = POMCPSolver(
            gen, obs_model,
            n_simulations=n_simulations,
            horizon=horizon,
            c_uct=c_uct,
            gamma=gamma,
            rollout_depth=rollout_depth,
            rollout_d_thresh=rollout_d_thresh,
            rollout_ttc_thresh=rollout_ttc_thresh,
        )
        return cls(solver)

    # ------------------------------------------------------------------
    # Per-episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0) -> None:
        self._solver.seed(seed)
        self._solver.reset_tree()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def plan(
        self,
        rel_dist: float,
        ego_speed: float,
        merge_speed: float,
        belief: np.ndarray,
    ) -> int:
        """
        Parameters
        ----------
        rel_dist     : x_m − x_e  (positive = MV ahead)
        ego_speed    : ego longitudinal speed (m/s)
        merge_speed  : merge vehicle longitudinal speed (m/s)
        belief       : [b_coop, b_nc] from the Bayesian filter

        Returns
        -------
        action index in {0,1,2,3,4}  (typically 1=IDLE or 4=SLOWER)
        """
        if math.isnan(rel_dist) or math.isnan(merge_speed):
            # No merge vehicle visible — coast
            return _IDLE_IDX

        d   = float(rel_dist)
        v_e = float(ego_speed)   if not math.isnan(ego_speed)   else _EGO_CRUISE
        v_m = float(merge_speed) if not math.isnan(merge_speed) else 20.0

        return self._solver.plan(d, v_e, v_m, belief)
