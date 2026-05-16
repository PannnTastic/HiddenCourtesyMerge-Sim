"""Generate environment schematic for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

fig, ax = plt.subplots(figsize=(11, 5.8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')

# ── Road geometry ──────────────────────────────────────────────────────────────
road_y_bot = 1.3
road_y_top = 3.1
lane_y = 2.2
ax.plot([0.1, 10.9], [road_y_bot, road_y_bot], 'k-', lw=1.8)
ax.plot([0.1, 10.9], [road_y_top, road_y_top], 'k-', lw=1.8)
for x in np.arange(0.4, 10.8, 0.6):
    ax.plot([x, x + 0.3], [lane_y, lane_y], 'k--', lw=0.8)

# Ramp: upper edge converges to main road bottom at x~7.2
ramp_merge_x = 7.2
ramp_start_x = 10.6
ax.plot([ramp_start_x, ramp_merge_x], [road_y_bot - 1.0, road_y_bot], 'k-', lw=1.8)
ax.plot([ramp_start_x, ramp_merge_x + 0.5], [0.1, road_y_bot - 0.0], 'k-', lw=1.8)

# Road direction arrow
ax.annotate('', xy=(10.85, 2.6), xytext=(10.2, 2.6),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))
ax.text(10.5, 2.85, 'flow', fontsize=7.5, ha='center', color='#555555')

# Road / ramp labels
ax.text(0.25, 3.25, 'Main road', fontsize=9, color='#333333')
ax.text(9.8, 0.05, 'Ramp', fontsize=9, color='#333333', ha='center')

# ── Ego vehicle ────────────────────────────────────────────────────────────────
ego_x, ego_y = 3.4, 2.58
ego_w, ego_h = 0.9, 0.52
ego_rect = FancyBboxPatch((ego_x - ego_w/2, ego_y - ego_h/2), ego_w, ego_h,
    boxstyle='round,pad=0.05', facecolor='#4C78A8', edgecolor='black', lw=1.6, zorder=5)
ax.add_patch(ego_rect)
ax.text(ego_x, ego_y, 'Ego', fontsize=8.5, ha='center', va='center',
        color='white', fontweight='bold', zorder=6)

# ── Merge vehicle ──────────────────────────────────────────────────────────────
mv_x, mv_y = 7.6, 1.62
mv_w, mv_h = 0.9, 0.52
mv_rect = FancyBboxPatch((mv_x - mv_w/2, mv_y - mv_h/2), mv_w, mv_h,
    boxstyle='round,pad=0.05', facecolor='#E45756', edgecolor='black', lw=1.6, zorder=5)
ax.add_patch(mv_rect)
ax.text(mv_x, mv_y + 0.07, 'Merge', fontsize=7.5, ha='center', va='center',
        color='white', fontweight='bold', zorder=6)
ax.text(mv_x, mv_y - 0.13, 'vehicle', fontsize=7, ha='center', va='center',
        color='white', zorder=6)

# Hidden courtesy label under merge vehicle
ax.text(mv_x, mv_y - 0.52,
        r'$m \in \{$coop, non-coop$\}$',
        fontsize=7.5, ha='center', va='top', color='#B03030', style='italic',
        bbox=dict(boxstyle='round,pad=0.18', facecolor='#FFF0EF',
                  edgecolor='#E45756', lw=0.7))

# ── Interaction window ─────────────────────────────────────────────────────────
win_start_x = ego_x + ego_w/2 + 0.12
win_end_x = win_start_x + 3.85   # proportional to 38 m
win_y_bot = road_y_bot - 0.04
win_y_top = road_y_top + 0.04
win_rect = mpatches.Rectangle((win_start_x, win_y_bot),
                               win_end_x - win_start_x,
                               win_y_top - win_y_bot,
                               facecolor='gold', alpha=0.20,
                               edgecolor='goldenrod', lw=1.2,
                               linestyle='--', zorder=2)
ax.add_patch(win_rect)
ax.annotate('', xy=(win_start_x, road_y_top + 0.12),
            xytext=(win_end_x, road_y_top + 0.12),
            arrowprops=dict(arrowstyle='<->', color='goldenrod', lw=1.0))
ax.text((win_start_x + win_end_x)/2, road_y_top + 0.35,
        'Interaction window (2–40 m ahead of ego)',
        fontsize=7.5, ha='center', va='bottom', color='#7a5a00',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFF0',
                  edgecolor='goldenrod', lw=0.6))

# ── Observation box ────────────────────────────────────────────────────────────
obs_x, obs_y = 8.55, 4.65
obs_w, obs_h = 2.25, 1.20
obs_box = FancyBboxPatch((obs_x, obs_y - obs_h/2), obs_w, obs_h,
                          boxstyle='round,pad=0.1',
                          facecolor='#EEF2FF', edgecolor='#4C78A8', lw=1.0, zorder=4)
ax.add_patch(obs_box)
ax.text(obs_x + obs_w/2, obs_y + 0.53,
        r'Observation $o_t$',
        fontsize=8, ha='center', va='top', fontweight='bold', color='#2a4e7a')
obs_lines = [
    r'urg $= 1/\mathrm{TTC}$',
    r'ard $= |x_m - x_e|$',
    r'rs $= v_m - v_e$',
    r'mvs $= v_m$,  mva $= \dot{v}_m$',
]
for k, line in enumerate(obs_lines):
    ax.text(obs_x + 0.12, obs_y + 0.30 - k*0.22, line,
            fontsize=6.8, va='top', color='#222')

# Arrow: merge vehicle → obs box
ax.annotate('', xy=(obs_x + 0.4, obs_y - obs_h/2),
            xytext=(mv_x + 0.1, mv_y + mv_h/2),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.1,
                            connectionstyle='arc3,rad=-0.28'))
ax.text(8.35, 3.35, 'observe', fontsize=7, color='#555', rotation=60)

# ── Belief box ─────────────────────────────────────────────────────────────────
bel_x, bel_y = 4.85, 5.08
bel_w, bel_h = 2.35, 0.88
bel_box = FancyBboxPatch((bel_x, bel_y - bel_h/2), bel_w, bel_h,
                          boxstyle='round,pad=0.1',
                          facecolor='#F0FFF0', edgecolor='#2ca02c', lw=1.0, zorder=4)
ax.add_patch(bel_box)
ax.text(bel_x + bel_w/2, bel_y + 0.18,
        'Bayesian belief update',
        fontsize=7.8, ha='center', va='center', fontweight='bold', color='#1a6e1a')
ax.text(bel_x + bel_w/2, bel_y - 0.13,
        r'$b_{t+1}(m)\!\propto\!b_t(m)\!\cdot\!Z(o_t|m)$',
        fontsize=7.5, ha='center', va='center', color='#1a6e1a')

# Arrow: obs → belief
ax.annotate('', xy=(bel_x + bel_w, bel_y + 0.05),
            xytext=(obs_x, obs_y + 0.05),
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))
ax.text(7.6, 5.38, r'$Z(o_t|m)$', fontsize=7.8, ha='center', color='#2ca02c')

# ── Policy / action box ────────────────────────────────────────────────────────
act_x, act_y = 1.55, 5.08
act_w, act_h = 2.15, 0.88
act_box = FancyBboxPatch((act_x, act_y - act_h/2), act_w, act_h,
                          boxstyle='round,pad=0.1',
                          facecolor='#FFF8F0', edgecolor='#FF7F0E', lw=1.0, zorder=4)
ax.add_patch(act_box)
ax.text(act_x + act_w/2, act_y + 0.18,
        'Heuristic policy',
        fontsize=7.8, ha='center', va='center', fontweight='bold', color='#a05000')
ax.text(act_x + act_w/2, act_y - 0.13,
        r'$a_t = \pi(b_t)\!\in\!\{$IDLE, SLOWER, ...$\}$',
        fontsize=6.8, ha='center', va='center', color='#a05000')

# Arrow: belief → policy (reverse: policy reads belief)
ax.annotate('', xy=(act_x + act_w, act_y),
            xytext=(bel_x, act_y),
            arrowprops=dict(arrowstyle='<-', color='#FF7F0E', lw=1.2))
ax.text(3.90, 5.28, r'$b_t$', fontsize=9, ha='center', color='#FF7F0E')

# Arrow: policy → ego
ax.annotate('', xy=(ego_x + 0.05, ego_y + ego_h/2 + 0.02),
            xytext=(act_x + act_w/2, act_y - act_h/2),
            arrowprops=dict(arrowstyle='->', color='#FF7F0E', lw=1.1,
                            connectionstyle='arc3,rad=0.18'))
ax.text(2.4, 3.95, r'$a_t$', fontsize=9, color='#FF7F0E')

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor='#4C78A8', edgecolor='black', label='Ego vehicle'),
    mpatches.Patch(facecolor='#E45756', edgecolor='black', label='Merge vehicle'),
    mpatches.Patch(facecolor='gold', edgecolor='goldenrod', alpha=0.4,
                   linestyle='--', label='Interaction window'),
]
ax.legend(handles=legend_handles, loc='lower left',
          bbox_to_anchor=(0.0, 0.0), fontsize=8,
          framealpha=0.9, edgecolor='#888', handlelength=1.3)

fig.tight_layout(pad=0.4)
Path('paper_figures').mkdir(exist_ok=True)
fig.savefig('paper_figures/environment_schematic.png', dpi=180,
            bbox_inches='tight', facecolor='white')
print('Saved paper_figures/environment_schematic.png')
