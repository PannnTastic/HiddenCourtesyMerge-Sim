"""
Record one GIF per policy × courtesy combination (8 GIFs total).

Output: gifs/
  belief_cooperative.gif
  belief_non_cooperative.gif
  oracle_cooperative.gif
  oracle_non_cooperative.gif
  rule_cooperative.gif
  rule_non_cooperative.gif
  random_cooperative.gif
  random_non_cooperative.gif

Run from D:\\MERGE\\:
    python record_gifs.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
from PIL import Image, ImageDraw, ImageFont

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
    safe_get_xy_speed,
    select_merge_vehicle,
    update_belief as gen_update_belief,
)

# ── Config ───────────────────────────────────────────────────────────
POLICIES   = ["belief_policy", "oracle_policy", "rule_policy", "random_policy"]
COURTESIES = ["cooperative", "non_cooperative"]

SEED       = 42        # episode seed (same for all combos — matched)
DENSITY    = 0.9       # traffic density
FPS_GIF    = 6         # frames per second in output GIF
SCALE      = 2         # upscale factor for sim frame (150×600 → 300×1200)
HUD_H      = 110       # pixel height of HUD strip below sim frame
OUT_DIR    = Path("gifs")

# ── Colours (RGB for PIL) ────────────────────────────────────────────
BG         = (20,  22,  32)
PANEL_BG   = (28,  33,  46)
WHITE      = (230, 234, 245)
GRAY       = (130, 138, 158)
DIM        = (60,   65,  85)
COOP_COL   = (55,  185,  95)
NC_COL     = (215,  65,  65)
YELLOW     = (232, 188,  48)
BLUE       = (80,  140, 220)
ORANGE     = (238, 148,  48)
WIN_ON     = (240, 185,  45)

# ── Fonts ────────────────────────────────────────────────────────────
def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ["consola.ttf", "consolab.ttf", "DejaVuSansMono.ttf",
                 "cour.ttf", "courbd.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        path = "C:/Windows/Fonts/consola.ttf"
        return ImageFont.truetype(path, size)
    except Exception:
        pass
    return ImageFont.load_default()


F_TITLE = _load_font(17, bold=True)
F_MED   = _load_font(14)
F_SM    = _load_font(12)


# ── Environment ──────────────────────────────────────────────────────
_ENV_CFG = {
    "duration": 12,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "controlled_vehicles": 1,
    "collision_reward": -100,
    "high_speed_reward": 1,
    "merging_speed_reward": -0.5,
    "reward_speed_range": [20, 30],
    "normalize_reward": False,
    "offscreen_rendering": True,
}


def _mva(cur: float, prev: float | None) -> float:
    if prev is None or math.isnan(cur) or math.isnan(prev):
        return float("nan")
    return (cur - prev) / _POLICY_DT


def _belief_for_policy(policy: str, belief: np.ndarray,
                        true_courtesy: str) -> np.ndarray:
    b = belief.copy()
    if policy == "oracle_policy":
        b[:] = 0.0
        b[list(COURTESY_TYPES).index(true_courtesy)] = 1.0
    elif policy == "rule_policy":
        b[:] = 0.5
    # belief_policy: use as-is; random_policy: doesn't use belief
    return b


# ── PIL drawing helpers ───────────────────────────────────────────────

def _bar(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int,
         frac: float, fg: tuple, bg: tuple = DIM) -> None:
    draw.rectangle([x, y, x + w, y + h], fill=bg)
    fill_w = max(0, int(w * min(max(frac, 0.0), 1.0)))
    if fill_w:
        draw.rectangle([x, y, x + fill_w, y + h], fill=fg)
    draw.rectangle([x, y, x + w, y + h], outline=DIM, width=1)


def _txt(draw: ImageDraw.Draw, text: str, font, color: tuple,
         x: int, y: int) -> int:
    draw.text((x, y), text, font=font, fill=color)
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0]
    except Exception:
        return len(text) * 8


# ── Frame composer ───────────────────────────────────────────────────

def _compose_frame(
    sim_frame: np.ndarray,       # (H, W, 3) from env.render()
    policy: str,
    courtesy: str,
    belief: np.ndarray,
    action_name: str,
    ttc: float,
    total_reward: float,
    step_num: int,
    in_window: bool,
    window_step: int,
    collision: bool,
    done: bool,
) -> Image.Image:
    """Build one GIF frame: scaled sim + HUD strip."""
    sim_h, sim_w = sim_frame.shape[:2]
    new_w = sim_w * SCALE
    new_h = sim_h * SCALE

    sim_img = Image.fromarray(sim_frame).resize(
        (new_w, new_h), Image.NEAREST
    )

    # Optional: amber overlay bar at top when window active
    if in_window:
        top = Image.new("RGBA", (new_w, 24), (240, 185, 45, 180))
        sim_rgba = sim_img.convert("RGBA")
        sim_rgba.paste(top, (0, 0), top)
        sim_img = sim_rgba.convert("RGB")
        draw_top = ImageDraw.Draw(sim_img)
        label = "[ INTERACTION WINDOW ACTIVE ]"
        try:
            bb = F_MED.getbbox(label)
            lw = bb[2] - bb[0]
        except Exception:
            lw = len(label) * 8
        draw_top.text(((new_w - lw) // 2, 4), label, font=F_MED, fill=(20, 22, 32))

    # HUD strip
    hud = Image.new("RGB", (new_w, HUD_H), PANEL_BG)
    draw = ImageDraw.Draw(hud)
    draw.line([(0, 0), (new_w, 0)], fill=DIM, width=2)

    x = 12

    # Row 1: policy | courtesy | step
    pol_short = policy.replace("_policy", "").upper()
    pol_col   = {"BELIEF": BLUE, "ORACLE": COOP_COL,
                 "RULE": YELLOW, "RANDOM": GRAY}.get(pol_short, WHITE)
    c_col = COOP_COL if courtesy == "cooperative" else NC_COL
    c_lbl = "COOPERATIVE" if courtesy == "cooperative" else "NON-COOPERATIVE"

    cx = x
    cx += _txt(draw, "Policy: ", F_MED, GRAY, cx, 8) + 2
    cx += _txt(draw, pol_short, F_MED, pol_col, cx, 8) + 14
    cx += _txt(draw, "Hidden: ", F_MED, GRAY, cx, 8) + 2
    _txt(draw, c_lbl, F_MED, c_col, cx, 8)

    step_txt = f"Step {step_num:2d}/60"
    try:
        bb = F_MED.getbbox(step_txt)
        tw = bb[2] - bb[0]
    except Exception:
        tw = len(step_txt) * 8
    _txt(draw, step_txt, F_MED, GRAY, new_w - tw - 12, 8)

    # Row 2: Belief bars
    b_coop = float(belief[0])
    b_nc   = float(belief[1])
    BAR_W  = 200

    _txt(draw, "coop:", F_SM, GRAY, x, 30)
    _bar(draw, x + 46, 30, BAR_W, 12, b_coop, COOP_COL)
    _txt(draw, f" {b_coop*100:4.1f}%", F_SM, COOP_COL, x + 46 + BAR_W, 30)

    _txt(draw, "  nc:", F_SM, GRAY, x, 46)
    _bar(draw, x + 46, 46, BAR_W, 12, b_nc, NC_COL)
    _txt(draw, f" {b_nc*100:4.1f}%", F_SM, NC_COL, x + 46 + BAR_W, 46)

    correct = (
        (b_coop > 0.5 and courtesy == "cooperative") or
        (b_nc   > 0.5 and courtesy == "non_cooperative")
    )
    acc_col = COOP_COL if correct else NC_COL
    acc_txt = "CORRECT ✓" if correct else "WRONG  ✗"
    try:
        bb = F_TITLE.getbbox(acc_txt)
        aw = bb[2] - bb[0]
    except Exception:
        aw = len(acc_txt) * 10
    _txt(draw, acc_txt, F_TITLE, acc_col, new_w - aw - 12, 36)

    # Row 3: window note
    if in_window and window_step <= 5:
        note = f"early window (win-step {window_step}/5): belief unreliable — paper: 71.3% acc"
        _txt(draw, note, F_SM, YELLOW, x, 64)
    elif in_window and window_step > 20:
        note = f"late window (win-step {window_step}): belief converged — paper: 90.3% acc"
        _txt(draw, note, F_SM, COOP_COL, x, 64)
    elif in_window:
        note = f"mid window (win-step {window_step}): belief accumulating evidence..."
        _txt(draw, note, F_SM, GRAY, x, 64)

    # Row 4: metrics
    action_col = NC_COL if action_name == "SLOWER" else BLUE
    ttc_str = f"{ttc:.1f}s" if not math.isnan(ttc) else "---"

    parts = [
        ("Action: ", action_name, action_col),
        ("  TTC: ",  ttc_str,     YELLOW),
        ("  TotalR: ", f"{total_reward:.1f}", WHITE),
    ]
    cx = x
    for lbl, val, col in parts:
        cx += _txt(draw, lbl, F_MED, GRAY, cx, 82) + 2
        cx += _txt(draw, val, F_MED, col, cx, 82) + 2

    if collision:
        _txt(draw, "  COLLISION!", F_TITLE, NC_COL, cx, 82)
    elif done:
        _txt(draw, "  EPISODE END", F_TITLE, YELLOW, cx, 82)

    # Combine sim + hud
    combined = Image.new("RGB", (new_w, new_h + HUD_H))
    combined.paste(sim_img, (0, 0))
    combined.paste(hud, (0, new_h))
    return combined


# ── Episode runner ────────────────────────────────────────────────────

def run_episode(
    env: gym.Env,
    policy: str,
    courtesy: str,
    rng: np.random.Generator,
) -> list[Image.Image]:
    """Run one episode, return list of PIL frames."""
    env.unwrapped.configure({"vehicles_count": int(np.clip(round(5 * DENSITY), 3, 7))})
    env.reset(seed=SEED)

    mv = select_merge_vehicle(env)
    configure_courtesy_vehicle(mv, courtesy, rng)

    belief     = np.array([0.5, 0.5])
    prev_spd   = None
    total_r    = 0.0
    win_step   = 0
    frames: list[Image.Image] = []

    for step in range(60):
        ego = env.unwrapped.vehicle
        ex, _, espd = safe_get_xy_speed(ego)
        mx, _, mspd = safe_get_xy_speed(mv) if mv else (float("nan"),) * 3

        rd   = mx - ex if not math.isnan(mx) else float("nan")
        rs   = mspd - espd if not math.isnan(mspd) else float("nan")
        ard  = abs(rd) if not math.isnan(rd) else float("nan")
        mva  = _mva(mspd, prev_spd)
        if not math.isnan(mspd):
            prev_spd = mspd

        ttc      = estimate_ttc(rd, rs)
        fg, _    = compute_gaps(env, ego)
        in_win   = in_interaction_window(rd)

        if in_win:
            urg    = 1.0 / max(ttc, 1e-3) if not math.isnan(ttc) else float("nan")
            belief = gen_update_belief(belief, urg, ard, rs, mspd, mva)
            win_step += 1

        if policy == "random_policy":
            action_idx = int(rng.integers(0, 5))
        else:
            b_act = _belief_for_policy(policy, belief, courtesy)
            action_idx = choose_ego_action(b_act, ttc, fg, rng, mspd, espd, rd)

        action_name = ACTION_NAMES.get(action_idx, str(action_idx))

        sim_frame = env.render()   # (150, 600, 3)
        collision  = bool(getattr(ego, "crashed", False))

        _, env_r, terminated, truncated, _ = env.step(action_idx)
        close_call = (not math.isnan(ttc)) and ttc < 3.0
        total_r += float(env_r) - (0.25 if close_call else 0.0)

        done = terminated or truncated or collision or step == 59

        frame = _compose_frame(
            sim_frame, policy, courtesy, belief, action_name,
            ttc, total_r, step + 1, in_win, win_step, collision, done,
        )
        frames.append(frame)

        if done:
            # Freeze last frame for 2 extra seconds
            for _ in range(FPS_GIF * 2):
                frames.append(frame)
            break

    return frames


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    load_obs_model("observation_model.json")

    env = gym.make("merge-v0", render_mode="rgb_array")
    env.unwrapped.configure(_ENV_CFG)
    env.reset(seed=SEED)

    total = len(POLICIES) * len(COURTESIES)
    done_count = 0

    for policy in POLICIES:
        for courtesy in COURTESIES:
            done_count += 1
            slug = f"{policy.replace('_policy', '')}_{courtesy}"
            out_path = OUT_DIR / f"{slug}.gif"
            print(f"[{done_count}/{total}] Recording {slug} ...", end=" ", flush=True)

            rng = np.random.default_rng(99)
            frames = run_episode(env, policy, courtesy, rng)

            duration_ms = int(1000 / FPS_GIF)
            frames[0].save(
                out_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
            print(f"{len(frames)} frames -> {out_path}  ({out_path.stat().st_size // 1024} KB)")

    print(f"\nDone. {total} GIFs saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
