"""
HiddenCourtesyMerge-Sim — Interactive Pygame Visualizer
========================================================
Runs the real model (IDM/MOBIL courtesy injection, Bayesian belief update,
calibrated observation model) and shows what the paper calls the
"early-window belief convergence deficit" in real time.

Controls
--------
  B / O / R / N  : switch policy (Belief / Oracle / Rule / raNdom)
  C / X / Z      : force Courtesy (Cooperative / non-eXclusive / random-Z)
  SPACE          : pause / resume
  ENTER          : new episode immediately
  UP / DOWN      : simulation speed  (x1 / x2 / x4 / x8)
  ESC            : quit

Run from D:\\MERGE\\:
    python sim_visualizer.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pygame

# ── highway-env + project imports ───────────────────────────────────
import gymnasium as gym
import highway_env  # noqa: F401 – registers merge-v0

from generate_hidden_courtesy_merge_dataset import (
    ACTION_NAMES,
    COURTESY_TYPES,
    _POLICY_DT,
    choose_ego_action,
    compute_gaps,
    configure_courtesy_vehicle,
    estimate_ttc,
    in_interaction_window,
    load_obs_model,
    other_vehicles,
    safe_get_xy_speed,
    select_merge_vehicle,
    update_belief as gen_update_belief,
)

# ── Colour palette ───────────────────────────────────────────────────
BG        = (15,  17,  24)
PANEL_BG  = (24,  28,  40)
BORDER    = (48,  54,  72)
WHITE     = (232, 236, 245)
GRAY      = (120, 128, 148)
DIM       = (55,  60,  78)
COOP_COL  = (55,  185,  95)   # green — cooperative
NC_COL    = (215,  65,  65)   # red   — non-cooperative
YELLOW    = (232, 188,  48)
BLUE      = (75,  135, 215)
ORANGE    = (238, 148,  48)
WIN_ON    = (240, 185,  45)   # amber — interaction window active
WIN_OFF   = (58,   62,  80)

# ── Layout constants ─────────────────────────────────────────────────
SIM_W    = 880
SIM_H    = 280
PANEL_W  = 310
HUD_H    = 230
W        = SIM_W + PANEL_W
H        = SIM_H + HUD_H

MAX_STEPS   = 60
FPS         = 30
SPEED_STEPS = [1, 2, 4, 8]


# ────────────────────────────────────────────────────────────────────
_BASE_CONFIG = {
    "duration": 12,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "controlled_vehicles": 1,
    "collision_reward": -100,
    "high_speed_reward": 1,
    "merging_speed_reward": -0.5,
    "reward_speed_range": [20, 30],
    "normalize_reward": False,
    # Prevent highway-env from calling pygame.display.set_mode() and
    # opening its own "Highway-env" window that would steal our display.
    "offscreen_rendering": True,
}


def _make_env_once() -> gym.Env:
    """Create the env once; offscreen_rendering keeps it from stealing our display."""
    env = gym.make("merge-v0", render_mode="rgb_array")
    env.unwrapped.configure(_BASE_CONFIG)
    return env


def _reset_env(env: gym.Env, seed: int, density: float) -> None:
    """Reconfigure density and reset (no close — avoids pygame.quit crash)."""
    env.unwrapped.configure(
        {"vehicles_count": int(np.clip(round(5 * density), 3, 7))}
    )
    env.reset(seed=seed)


def _mva(cur: float, prev: float | None) -> float:
    if prev is None or math.isnan(cur) or math.isnan(prev):
        return float("nan")
    return (cur - prev) / _POLICY_DT


# ────────────────────────────────────────────────────────────────────
class MergeVisualizer:

    POLICY_KEYS = {
        pygame.K_b: "belief_policy",
        pygame.K_o: "oracle_policy",
        pygame.K_r: "rule_policy",
        pygame.K_n: "random_policy",
    }
    COURTESY_KEYS = {
        pygame.K_c: "cooperative",
        pygame.K_x: "non_cooperative",
        pygame.K_z: None,
    }

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("HiddenCourtesyMerge-Sim — Interactive Visualizer")
        self.clock = pygame.time.Clock()

        try:
            self.f_title = pygame.font.SysFont("Consolas", 18, bold=True)
            self.f_med   = pygame.font.SysFont("Consolas", 15)
            self.f_sm    = pygame.font.SysFont("Consolas", 13)
        except Exception:
            self.f_title = pygame.font.SysFont(None, 22, bold=True)
            self.f_med   = pygame.font.SysFont(None, 18)
            self.f_sm    = pygame.font.SysFont(None, 15)

        # ── Settings (persist across episodes) ──────────────────────
        self.policy_name     = "belief_policy"
        self.forced_courtesy: str | None = None
        self.paused          = False
        self.speed_idx       = 0          # index into SPEED_STEPS
        self.ep_seed         = 200
        self.ep_count        = 0

        # ── RNG & obs model (needed before env) ─────────────────────
        self.rng = np.random.default_rng(77)
        load_obs_model("observation_model.json")

        # ── Env: created AFTER our display; offscreen_rendering keeps
        #    highway-env from calling set_mode() and stealing our window ──
        self.env = _make_env_once()

        # ── Per-episode state ────────────────────────────────────────
        self.mv              = None
        self.true_courtesy: str | None = None
        self.belief          = np.array([0.5, 0.5])
        self.step_num        = 0
        self.total_reward    = 0.0
        self.prev_mv_spd: float | None = None
        self.in_window       = False
        self.ttc             = float("nan")
        self.front_gap       = float("nan")
        self.action_name     = "---"
        self.step_reward     = 0.0
        self.collision       = False
        self.success         = False
        self.done            = False

        # History for mini-chart (len = step_num)
        self.belief_hist: list[float] = []
        self.window_hist: list[bool]  = []
        self.window_step_count: int   = 0  # steps inside interaction window

        self._reset()

    # ── Episode lifecycle ────────────────────────────────────────────

    def _reset(self) -> None:
        density = float(self.rng.uniform(0.7, 1.1))
        _reset_env(self.env, self.ep_seed, density)   # reuse env — never close
        self.ep_seed += 1
        self.ep_count += 1

        courtesy = (
            self.forced_courtesy
            if self.forced_courtesy is not None
            else str(self.rng.choice(list(COURTESY_TYPES)))
        )
        self.true_courtesy = courtesy
        self.mv = select_merge_vehicle(self.env)
        configure_courtesy_vehicle(self.mv, courtesy, self.rng)

        self.belief        = np.array([0.5, 0.5])
        self.step_num      = 0
        self.total_reward  = 0.0
        self.prev_mv_spd   = None
        self.in_window     = False
        self.ttc           = float("nan")
        self.front_gap     = float("nan")
        self.action_name   = "---"
        self.step_reward   = 0.0
        self.collision     = False
        self.success       = False
        self.done          = False
        self.belief_hist        = []
        self.window_hist        = []
        self.window_step_count  = 0

    # ── Simulation step ──────────────────────────────────────────────

    def _step(self) -> None:
        if self.done or self.env is None:
            return

        ego = self.env.unwrapped.vehicle
        ego_x, ego_y, ego_spd = safe_get_xy_speed(ego)

        # Merge vehicle features
        if self.mv is not None:
            mv_x, mv_y, mv_spd = safe_get_xy_speed(self.mv)
        else:
            mv_x = mv_y = mv_spd = float("nan")

        rel_dist = mv_x - ego_x if not math.isnan(mv_x) else float("nan")
        rel_spd  = mv_spd - ego_spd if not math.isnan(mv_spd) else float("nan")
        ard      = abs(rel_dist) if not math.isnan(rel_dist) else float("nan")
        mva_val  = _mva(mv_spd, self.prev_mv_spd)

        # Compute TTC first so urgency uses current value
        self.ttc          = estimate_ttc(rel_dist, rel_spd)
        self.front_gap, _ = compute_gaps(self.env, ego)
        urg = (1.0 / max(self.ttc, 1e-3)
               if not math.isnan(self.ttc)
               else float("nan"))
        self.in_window  = in_interaction_window(rel_dist)

        if not math.isnan(mv_spd):
            self.prev_mv_spd = mv_spd

        # Bayesian belief update (only inside interaction window)
        if self.in_window:
            self.belief = gen_update_belief(
                self.belief, urg, ard, rel_spd, mv_spd, mva_val
            )

        # Action selection (policy-specific)
        action_idx = self._choose_action(ego_spd, mv_spd, rel_dist)
        self.action_name = ACTION_NAMES.get(action_idx, str(action_idx))

        # Environment step
        _, env_reward, terminated, truncated, _ = self.env.step(action_idx)

        self.collision = bool(getattr(ego, "crashed", False))
        close_call = (not math.isnan(self.ttc)) and self.ttc < 3.0
        self.step_reward  = float(env_reward) - (0.25 if close_call else 0.0)
        self.total_reward += self.step_reward

        ego_x_new = safe_get_xy_speed(ego)[0]
        initial_x = safe_get_xy_speed(self.env.unwrapped.vehicle)[0]
        progress = ego_x_new - (ego_x if not math.isnan(ego_x) else ego_x_new)

        self.step_num += 1
        self.belief_hist.append(float(self.belief[0]))
        self.window_hist.append(self.in_window)
        if self.in_window:
            self.window_step_count += 1

        if self.collision:
            self.success = False
            self.done = True
        elif terminated or truncated or self.step_num >= MAX_STEPS:
            self.done = True

    def _choose_action(self, ego_spd: float, mv_spd: float, rel_dist: float) -> int:
        if self.policy_name == "random_policy":
            return int(self.rng.integers(0, 5))

        b = self.belief.copy()

        if self.policy_name == "oracle_policy":
            b[:] = 0.0
            b[list(COURTESY_TYPES).index(self.true_courtesy)] = 1.0

        elif self.policy_name == "rule_policy":
            b[:] = 0.5

        # belief_policy: use actual belief as-is

        return choose_ego_action(
            b, self.ttc, self.front_gap, self.rng,
            merge_speed=mv_spd, ego_speed=ego_spd, relative_distance=rel_dist,
        )

    # ── Rendering helpers ────────────────────────────────────────────

    def _txt(self, text: str, font, color, x: int, y: int) -> int:
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))
        return surf.get_width()

    def _bar(self, x: int, y: int, w: int, h: int,
             frac: float, fg: tuple, bg: tuple = DIM) -> None:
        pygame.draw.rect(self.screen, bg, (x, y, w, h))
        fill = max(0, int(w * min(max(frac, 0.0), 1.0)))
        if fill:
            pygame.draw.rect(self.screen, fg, (x, y, fill, h))
        pygame.draw.rect(self.screen, BORDER, (x, y, w, h), 1)

    def _line(self, x0, y0, x1, y1, col=BORDER, w=1) -> None:
        pygame.draw.line(self.screen, col, (x0, y0), (x1, y1), w)

    # ── Render simulation view ───────────────────────────────────────

    def _render_sim(self) -> None:
        if self.env is None:
            return
        try:
            frame = self.env.render()
        except Exception:
            return
        if frame is None:
            return
        # frame shape: (H, W, 3)
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (SIM_W, SIM_H))
        self.screen.blit(surf, (0, 0))

        # Overlay: interaction window indicator at top of sim view
        if self.in_window:
            win_surf = self.f_med.render("[ INTERACTION WINDOW ACTIVE ]", True, WIN_ON)
            self.screen.blit(win_surf, (SIM_W // 2 - win_surf.get_width() // 2, 6))

    # ── Render HUD (bottom strip) ────────────────────────────────────

    def _render_hud(self) -> None:
        hx, hy = 0, SIM_H
        pygame.draw.rect(self.screen, PANEL_BG, (hx, hy, SIM_W, HUD_H))
        self._line(hx, hy, hx + SIM_W, hy, BORDER, 2)

        x = 14
        # Fixed absolute row positions — prevents dynamic-y overlap
        R1 = hy + 8    # hidden type + step + window
        R2 = hy + 32   # coop bar
        R3 = hy + 50   # nc bar + CORRECT badge
        R4 = hy + 70   # window timing note
        R5 = hy + 92   # metrics strip
        R7 = hy + 126  # chart (label drawn at R7-14)

        # ── R1: hidden courtesy | step | window badge ──────────
        c      = self.true_courtesy or "unknown"
        c_col  = COOP_COL if c == "cooperative" else NC_COL
        c_lbl  = "COOPERATIVE" if c == "cooperative" else "NON-COOPERATIVE"

        self._txt("Hidden type: ", self.f_title, GRAY, x, R1)
        prefix_w = self.f_title.render("Hidden type: ", True, GRAY).get_width()
        self._txt(c_lbl, self.f_title, c_col, x + prefix_w, R1)

        step_str = f"Step {self.step_num:2d}/{MAX_STEPS}   Ep #{self.ep_count:3d}"
        self._txt(step_str, self.f_med, GRAY, 440, R1 + 2)

        w_col = WIN_ON if self.in_window else WIN_OFF
        w_txt = "[ WINDOW ACTIVE ]" if self.in_window else "[  away  ]"
        self._txt(w_txt, self.f_med, w_col, 700, R1 + 2)

        # ── R2+R3: Belief bars ─────────────────────────────────
        b_coop = float(self.belief[0])
        b_nc   = float(self.belief[1])
        BAR_W  = 240

        self._txt("coop:", self.f_sm, GRAY, x, R2 + 1)
        self._bar(x + 50, R2, BAR_W, 14, b_coop, COOP_COL)
        self._txt(f" {b_coop*100:4.1f}%", self.f_sm, COOP_COL, x + 50 + BAR_W, R2 + 1)

        self._txt("  nc:", self.f_sm, GRAY, x, R3 + 1)
        self._bar(x + 50, R3, BAR_W, 14, b_nc, NC_COL)
        self._txt(f" {b_nc*100:4.1f}%", self.f_sm, NC_COL, x + 50 + BAR_W, R3 + 1)

        correct = (
            (b_coop > 0.5 and c == "cooperative") or
            (b_nc   > 0.5 and c == "non_cooperative")
        )
        acc_col = COOP_COL if correct else NC_COL
        acc_txt = "CORRECT ✓" if correct else "WRONG  ✗"
        self._txt(acc_txt, self.f_title, acc_col, 440, R3)

        # ── R4: Window timing note (non-overlapping) ───────────
        if self.in_window and self.window_step_count <= 5:
            note = (f"early window (win-step {self.window_step_count}/5): "
                    f"belief unreliable — paper: 71.3% acc steps 1-5")
            self._txt(note, self.f_sm, YELLOW, x, R4)
        elif self.in_window and self.window_step_count > 20:
            note = (f"late window (win-step {self.window_step_count}): "
                    f"belief converged — paper: 90.3% acc steps 21+")
            self._txt(note, self.f_sm, COOP_COL, x, R4)
        elif self.in_window:
            note = (f"mid window (win-step {self.window_step_count}): "
                    f"belief accumulating evidence...")
            self._txt(note, self.f_sm, GRAY, x, R4)

        # ── R5: Metrics strip ──────────────────────────────────
        pol_short = self.policy_name.replace("_policy", "").upper()
        pol_col   = {"BELIEF": BLUE, "ORACLE": COOP_COL,
                     "RULE": YELLOW, "RANDOM": GRAY}.get(pol_short, WHITE)

        metrics = [
            ("Policy: ",   pol_short,                                          pol_col),
            ("  Action: ", self.action_name,
             NC_COL if self.action_name == "SLOWER" else BLUE),
            ("  TTC: ",    f"{self.ttc:.1f}s" if not math.isnan(self.ttc) else "---",
             YELLOW),
            ("  StepR: ",  f"{self.step_reward:+.2f}",  GRAY),
            ("  TotalR: ", f"{self.total_reward:6.1f}", WHITE),
        ]
        cx = x
        for lbl, val, col in metrics:
            lw = self._txt(lbl, self.f_med, GRAY, cx, R5)
            vw = self._txt(val, self.f_med, col,  cx + lw, R5)
            cx += lw + vw

        if self.collision:
            self._txt("  COLLISION!", self.f_title, NC_COL, cx, R5)
        elif self.done:
            self._txt("  EPISODE END — press ENTER", self.f_title, YELLOW, cx, R5)

        # ── Chart ──────────────────────────────────────────────
        chart_h = HUD_H - (R7 - hy) - 6
        self._render_chart(x, R7, SIM_W - x - 14, chart_h)

    def _render_chart(self, cx: int, cy: int, cw: int, ch: int) -> None:
        """b_coop over episode; amber shading = interaction window active."""
        pygame.draw.rect(self.screen, BG, (cx, cy, cw, ch))
        pygame.draw.rect(self.screen, BORDER, (cx, cy, cw, ch), 1)

        # 0.5 reference line
        mid = cy + ch // 2
        self._line(cx, mid, cx + cw, mid, DIM)

        label = "b_coop over episode  (amber = interaction window active)"
        self._txt(label, self.f_sm, GRAY, cx, cy - 15)

        n = len(self.belief_hist)
        if n < 2:
            return

        slot = cw / MAX_STEPS

        # Amber shading for window steps
        for i, win in enumerate(self.window_hist):
            if win:
                sx = int(cx + i * slot)
                sw = max(1, int(slot))
                pygame.draw.rect(self.screen, (70, 58, 10), (sx, cy + 1, sw, ch - 2))

        # Belief line
        pts = []
        for i, b in enumerate(self.belief_hist):
            px = int(cx + i * slot)
            py = int(cy + ch - b * ch)
            py = max(cy + 1, min(cy + ch - 1, py))
            pts.append((px, py))

        for i in range(len(pts) - 1):
            pygame.draw.line(self.screen, COOP_COL, pts[i], pts[i + 1], 2)

        # 0.5 line on top
        self._line(cx, mid, cx + cw, mid, (90, 95, 115))

    # ── Render right panel ───────────────────────────────────────────

    def _render_panel(self) -> None:
        px = SIM_W
        pygame.draw.rect(self.screen, PANEL_BG, (px, 0, PANEL_W, H))
        self._line(px, 0, px, H, BORDER, 2)

        x, y = px + 14, 10

        def title(t: str, col=WHITE) -> None:
            nonlocal y
            w = self._txt(t, self.f_title, col, x, y)
            y += 22

        def sep() -> None:
            nonlocal y
            pygame.draw.line(self.screen, BORDER,
                             (x, y + 3), (x + PANEL_W - 28, y + 3))
            y += 12

        def row(key: str, label: str, active: bool, col=YELLOW) -> None:
            nonlocal y
            c = col if active else GRAY
            self._txt(f"[{key}] {label}", self.f_med, c, x + 4, y)
            y += 19

        # ── Policy ──────────────────────────────────────────────
        title("== POLICY ==")
        row("B", "Belief    (Bayesian)",  self.policy_name == "belief_policy",  BLUE)
        row("O", "Oracle    (cheat!)",    self.policy_name == "oracle_policy",  COOP_COL)
        row("R", "Rule      (no belief)", self.policy_name == "rule_policy",    YELLOW)
        row("N", "raNdom    (chaos)",     self.policy_name == "random_policy",  GRAY)
        sep()

        # ── Courtesy ─────────────────────────────────────────────
        title("== COURTESY ==")
        row("C", "force Cooperative",     self.forced_courtesy == "cooperative",   COOP_COL)
        row("X", "force Non-cooperative", self.forced_courtesy == "non_cooperative", NC_COL)
        row("Z", "random each episode",   self.forced_courtesy is None,            GRAY)
        sep()

        # ── Controls ─────────────────────────────────────────────
        title("== CONTROLS ==")
        spd = SPEED_STEPS[self.speed_idx]
        row("↑/↓", f"Speed  x{spd}", True, WHITE)
        row("SPC", "Pause / Resume", self.paused, YELLOW)
        row("ENT", "New episode",    False, WHITE)
        row("ESC", "Quit",           False, WHITE)
        sep()

        # ── Model stats ──────────────────────────────────────────
        title("== OBS MODEL ==")
        stats = [
            ("Features",    "ard, mvs, mva"),
            ("Step acc.",   "98.3%"),
            ("NLL",         "0.168 nats (4.1×)"),
            ("Early win.",  "71.3% (steps 1-5)"),
            ("Late win.",   "90.3% (steps 21+)"),
            ("λ-reg",       "0.08"),
        ]
        for k, v in stats:
            kw = self._txt(f"{k}: ", self.f_sm, GRAY,  x + 4, y)
            self._txt(v, self.f_sm, WHITE, x + 4 + kw, y)
            y += 17
        sep()

        # ── Key finding callout ───────────────────────────────────
        title("== KEY FINDING ==", ORANGE)
        finding = [
            "Belief policy does NOT",
            "significantly beat rule",
            "(p_adj=1.0, n=300).",
            "",
            "Reason: belief is only",
            "71.3% accurate in the",
            "early window — when the",
            "ego must act.",
            "",
            "Oracle proves courtesy",
            "matters (p=0.013), but",
            "the filter is too slow.",
        ]
        for line in finding:
            if line:
                self._txt(line, self.f_sm, DIM, x + 4, y)
            y += 16

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        auto_reset_timer = 0  # ms to wait before auto-resetting after episode end

        while True:
            dt_ms = self.clock.tick(FPS)

            # ── Events ──────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._quit()

                if event.type == pygame.KEYDOWN:
                    k = event.key

                    if k == pygame.K_ESCAPE:
                        self._quit()
                    elif k == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif k == pygame.K_RETURN:
                        self._reset()
                        auto_reset_timer = 0
                    elif k in self.POLICY_KEYS:
                        self.policy_name = self.POLICY_KEYS[k]
                    elif k in self.COURTESY_KEYS:
                        self.forced_courtesy = self.COURTESY_KEYS[k]
                    elif k == pygame.K_UP:
                        self.speed_idx = min(self.speed_idx + 1, len(SPEED_STEPS) - 1)
                    elif k == pygame.K_DOWN:
                        self.speed_idx = max(self.speed_idx - 1, 0)

            # ── Simulation ───────────────────────────────────────
            if self.done:
                auto_reset_timer += dt_ms
                if auto_reset_timer > 2000:   # auto-reset after 2 s
                    self._reset()
                    auto_reset_timer = 0
            elif not self.paused:
                for _ in range(SPEED_STEPS[self.speed_idx]):
                    if not self.done:
                        self._step()

            # ── Draw ─────────────────────────────────────────────
            # Re-assert our title each frame — highway-env may overwrite it
            pygame.display.set_caption("HiddenCourtesyMerge-Sim — Interactive Visualizer")
            self.screen.fill(BG)
            self._render_sim()
            self._render_panel()
            self._render_hud()
            pygame.display.flip()

    def _quit(self) -> None:
        pygame.quit()
        sys.exit()


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    viz = MergeVisualizer()
    viz.run()
