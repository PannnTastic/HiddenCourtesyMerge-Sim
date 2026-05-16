"""Environment schematic v2 — clean two-section layout."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.2)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Road scene  (y: 3.4 → 7.2)
# ══════════════════════════════════════════════════════════════════════════════
ax.text(6.5, 7.05, '(a) Highway Merge Scenario — top-down view',
        fontsize=11, ha='center', va='center', fontweight='bold', color='#111')

# Road geometry
lane_top = 6.30
lane_mid = 5.45   # divider between ego lane (upper) and merge lane (lower)
lane_bot = 4.60
road_left  = 0.45
road_right = 10.6

# Road fill + borders
ax.fill_between([road_left, road_right], lane_bot, lane_top,
                color='#e0e0e0', alpha=0.55, zorder=1)
for y in [lane_top, lane_bot]:
    ax.plot([road_left, road_right], [y, y], 'k-', lw=2.4, zorder=2)

# Lane-divider dashes
for x in np.arange(road_left + 0.3, road_right - 0.1, 0.55):
    ax.plot([x, x + 0.28], [lane_mid, lane_mid],
            color='#999', lw=1.2, zorder=2)

# Traffic flow arrow
ax.annotate('', xy=(road_right + 0.85, (lane_top + lane_bot) / 2),
            xytext=(road_right + 0.1, (lane_top + lane_bot) / 2),
            arrowprops=dict(arrowstyle='->', color='#444', lw=1.6), zorder=3)
ax.text(road_right + 0.92, (lane_top + lane_bot) / 2,
        'flow', fontsize=8.5, va='center', color='#444')

# On-ramp (enters from lower-right, merges into merge lane at road_right)
rx, ry = 12.2, 3.7   # ramp far tip
ax.plot([rx, road_right], [ry + 0.38, lane_bot], 'k-', lw=2.4, zorder=2)
ax.plot([rx, road_right], [ry - 0.38, lane_bot], 'k-', lw=2.4, zorder=2)
ax.fill([rx, road_right, road_right, rx],
        [ry - 0.38, lane_bot, lane_bot, ry + 0.38],
        color='#e0e0e0', alpha=0.55, zorder=1)
ax.text(12.45, ry, 'On-ramp', fontsize=8.5, ha='left', va='center',
        color='#444', fontweight='bold')

# Lane labels (left margin)
ax.text(road_left - 0.12, (lane_mid + lane_top) / 2,
        'Ego\nlane', fontsize=8, ha='right', va='center', color='#444')
ax.text(road_left - 0.12, (lane_mid + lane_bot) / 2,
        'Merge\nlane', fontsize=8, ha='right', va='center', color='#444')

# ── Ego vehicle ────────────────────────────────────────────────────────────────
ego_cx = 3.6
ego_cy = (lane_mid + lane_top) / 2    # upper lane centre
ego_w, ego_h = 1.25, 0.58
ax.add_patch(FancyBboxPatch(
    (ego_cx - ego_w/2, ego_cy - ego_h/2), ego_w, ego_h,
    boxstyle='round,pad=0.07',
    facecolor='#2166AC', edgecolor='#0d3d6b', lw=2.0, zorder=5))
ax.text(ego_cx, ego_cy, 'Ego', fontsize=10.5, ha='center', va='center',
        color='white', fontweight='bold', zorder=6)

# ── Merge vehicle ──────────────────────────────────────────────────────────────
mv_cx = 8.1
mv_cy = (lane_mid + lane_bot) / 2     # lower lane centre (coming off ramp)
mv_w, mv_h = 1.35, 0.58
ax.add_patch(FancyBboxPatch(
    (mv_cx - mv_w/2, mv_cy - mv_h/2), mv_w, mv_h,
    boxstyle='round,pad=0.07',
    facecolor='#D6604D', edgecolor='#8b1a0a', lw=2.0, zorder=5))
ax.text(mv_cx, mv_cy + 0.09, 'Merge', fontsize=9, ha='center', va='center',
        color='white', fontweight='bold', zorder=6)
ax.text(mv_cx, mv_cy - 0.12, 'vehicle', fontsize=8.5, ha='center', va='center',
        color='white', zorder=6)

# Hidden-state callout above merge vehicle
ax.annotate(r'$m \in$ {coop, non-coop}   [hidden]',
            xy=(mv_cx, mv_cy + mv_h/2 + 0.02),
            xytext=(mv_cx, lane_top + 0.12),
            fontsize=9, ha='center', va='bottom',
            color='#8b1a0a', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.28', facecolor='#FFF0EF',
                      edgecolor='#D6604D', lw=1.3),
            arrowprops=dict(arrowstyle='->', color='#D6604D', lw=1.2),
            zorder=7)

# ── Interaction window ─────────────────────────────────────────────────────────
iw_l = ego_cx + ego_w / 2 + 0.06
iw_r = mv_cx  - mv_w / 2 - 0.06

# Yellow highlight across both lanes
ax.add_patch(mpatches.Rectangle(
    (iw_l, lane_bot), iw_r - iw_l, lane_top - lane_bot,
    facecolor='#FFD700', alpha=0.22,
    edgecolor='#b8960a', lw=1.4, linestyle='--', zorder=2))

# Bracket below road
bkt_y = lane_bot - 0.25
ax.annotate('', xy=(iw_l, bkt_y), xytext=(iw_r, bkt_y),
            arrowprops=dict(arrowstyle='<->', color='#7a6000', lw=1.6), zorder=3)
ax.text((iw_l + iw_r) / 2, bkt_y - 0.13,
        'Interaction window: 2 m < d < 40 m  (belief updates active here)',
        fontsize=8, ha='center', va='top', color='#5a4000',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFDE7',
                  edgecolor='#b8960a', lw=0.8), zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — POMDP decision loop  (y: 0.1 → 3.3)
# ══════════════════════════════════════════════════════════════════════════════
ax.axhline(y=3.4, color='#cccccc', lw=0.9, linestyle=':')
ax.text(6.5, 3.28, '(b) POMDP Decision Loop (per timestep)',
        fontsize=11, ha='center', va='center', fontweight='bold', color='#111')

# Five boxes, left-to-right
# Centres:  1.1   3.6   6.35   9.0   11.5
BH  = 1.40   # box height
BY  = 1.75   # box y-centre

def add_box(cx, w, title, lines, fc, ec, dashed=False):
    ls = '--' if dashed else '-'
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, BY - BH/2), w, BH,
        boxstyle='round,pad=0.12', facecolor=fc, edgecolor=ec,
        lw=1.7, linestyle=ls, zorder=4))
    ax.text(cx, BY + BH/2 - 0.24, title,
            fontsize=8.8, ha='center', va='center',
            fontweight='bold', color=ec, zorder=5)
    for k, line in enumerate(lines):
        ax.text(cx, BY + 0.12 - k * 0.35, line,
                fontsize=8.2, ha='center', va='center', color='#222', zorder=5)

# Box specs: (cx, width, title, lines, facecolor, edgecolor, dashed)
add_box(1.15, 1.90,
        'Hidden state',
        [r'$m$ ∈ {coop, nc}', '(latent, unknown)'],
        '#FFF0EF', '#D6604D', dashed=True)

add_box(3.65, 2.15,
        r'Observations $o_t$',
        ['urg, ard, rs', 'mvs, mva  (noisy)'],
        '#EEF2FF', '#4C78A8')

add_box(6.40, 2.50,
        'Bayesian belief update',
        [r'$b_{t+1}(m) \propto b_t(m) \cdot Z(o_t|m)$',
         r'$\lambda{=}0.08$ prior shrinkage'],
        '#F0FFF0', '#2ca02c')

add_box(9.15, 2.10,
        r'Policy $\pi(b_t)$',
        ['V-shaped caution rule', 'action ∈ {IDLE, SLOWER…}'],
        '#FFF8F0', '#FF7F0E')

add_box(11.75, 1.90,
        r'Action $a_t$',
        ['→ Ego vehicle', '(executes)'],
        '#E8F0FF', '#2166AC')

# Arrows between boxes
def box_arrow(x1, x2, label, color):
    ax.annotate('', xy=(x2, BY), xytext=(x1, BY),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.6), zorder=3)
    if label:
        ax.text((x1 + x2) / 2, BY + 0.22, label,
                fontsize=8.5, ha='center', va='center',
                color=color, fontstyle='italic', zorder=3)

# m → o_t  (dashed: hidden causes observations)
ax.annotate('', xy=(3.65 - 2.15/2, BY), xytext=(1.15 + 1.90/2, BY),
            arrowprops=dict(arrowstyle='->', color='#D6604D', lw=1.6,
                            linestyle='dashed'), zorder=3)
ax.text((1.15 + 1.90/2 + 3.65 - 2.15/2) / 2, BY + 0.22,
        'drives behaviour', fontsize=7.8, ha='center', va='center',
        color='#D6604D', fontstyle='italic', zorder=3)

box_arrow(3.65 + 2.15/2, 6.40 - 2.50/2, r'$o_t$', '#4C78A8')
box_arrow(6.40 + 2.50/2, 9.15 - 2.10/2, r'$b_t$', '#2ca02c')
box_arrow(9.15 + 2.10/2, 11.75 - 1.90/2, r'$a_t$', '#FF7F0E')

# ── Legend ─────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor='#2166AC', edgecolor='#0d3d6b',
                   label='Ego vehicle (controlled agent)'),
    mpatches.Patch(facecolor='#D6604D', edgecolor='#8b1a0a',
                   label='Merge vehicle (IDM/MOBIL, hidden courtesy)'),
    mpatches.Patch(facecolor='#FFD700', edgecolor='#b8960a', alpha=0.5,
                   label='Interaction window (belief updates)'),
]
ax.legend(handles=handles, loc='upper left',
          bbox_to_anchor=(0.005, 0.995), fontsize=8.8,
          framealpha=0.96, edgecolor='#aaa', handlelength=1.6)

fig.tight_layout(pad=0.3)
out = Path(r'D:\MERGE\paper_figures\environment_schematic.png')
out.parent.mkdir(exist_ok=True)
fig.savefig(str(out), dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')
