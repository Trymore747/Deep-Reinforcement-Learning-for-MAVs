#!/usr/bin/env python3
"""
R54 Comprehensive Training Dashboard — 12 panels
Parses training_history_r54.log and generates a full performance plot including
Q-values, actor/critic losses, and all mission metrics.
Run: python3 src/tunnel_drl/plot_r54.py
"""

import re, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter

LOG = os.path.join(os.path.dirname(__file__), "results_r54", "training_history_r54.log")
OUT = os.path.join(os.path.dirname(__file__), "results_r54", "r54_dashboard.png")

# ── Parse log ─────────────────────────────────────────────────────────────────
pat_base = re.compile(
    r"(OK|FAIL)\s+M(\d+)\s+rew=([+-]?\d+(?:\.\d+)?)\s+dist=([\d.]+)m\s+spd=([\d.]+)m/s\s+"
    r"col=(\d+)\s+avd=(\d+)\s+dur=(\d+)s\s+sr=([\d.]+)%\s+sr20=([\d.]+)%\s+"
    r"stage=([\d.]+)m/s\s+event=(\w+)"
)
pat_ext = re.compile(
    r"critic_loss=([+-]?[\d.]+)\s+actor_loss=([+-]?[\d.]+)\s+q_mean=([+-]?[\d.]+)\s+noise=([\d.]+)"
)

records = []
with open(LOG) as f:
    for line in f:
        line = line.strip()
        mb = pat_base.search(line)
        if not mb:
            continue
        rec = dict(
            ok      = 1 if mb.group(1) == "OK" else 0,
            mission = int(mb.group(2)),
            rew     = float(mb.group(3)),
            dist    = float(mb.group(4)),
            spd     = float(mb.group(5)),
            col     = int(mb.group(6)),
            avd     = int(mb.group(7)),
            dur     = int(mb.group(8)),
            sr      = float(mb.group(9)),
            sr20    = float(mb.group(10)),
            stage   = float(mb.group(11)),
            event   = mb.group(12),
            critic_loss = None, actor_loss = None,
            q_mean  = None, noise = None,
        )
        me = pat_ext.search(line)
        if me:
            rec['critic_loss'] = float(me.group(1))
            rec['actor_loss']  = float(me.group(2))
            rec['q_mean']      = float(me.group(3))
            rec['noise']       = float(me.group(4))
        records.append(rec)

if not records:
    print(f"No records found in {LOG}"); sys.exit(1)

n   = len(records)
eps = list(range(1, n + 1))   # absolute log index (1-based)

ok    = [r['ok']    for r in records]
rew   = [r['rew']   for r in records]
dist  = [r['dist']  for r in records]
spd   = [r['spd']   for r in records]
col   = [r['col']   for r in records]
avd   = [r['avd']   for r in records]
dur   = [r['dur']   for r in records]
sr    = [r['sr']    for r in records]
sr20  = [r['sr20']  for r in records]
stg   = [r['stage'] for r in records]
ev    = [r['event'] for r in records]

avoid_rate = [avd[i]/(col[i]+avd[i])*100 if (col[i]+avd[i])>0 else 100 for i in range(n)]

# ── Collision distribution ────────────────────────────────────────────────────
from collections import Counter as _Counter
col_counts = _Counter(col)
max_col = max(col) if col else 0
col_bins  = list(range(0, min(max_col, 25) + 1))
col_freq  = [col_counts.get(b, 0) for b in col_bins]

# ── Stuck location analysis ───────────────────────────────────────────────────
stuck_dists = [dist[i] for i in range(n) if ev[i] == 'stuck']

# ── Extended metrics: only where data exists ───────────────────────────────────
# Use absolute episode index (position in combined log) as x-axis
ext_idx   = [i+1 for i,r in enumerate(records) if r['critic_loss'] is not None]
ext_cl    = [r['critic_loss'] for r in records if r['critic_loss'] is not None]
ext_al    = [r['actor_loss']  for r in records if r['actor_loss']  is not None]
ext_q     = [r['q_mean']      for r in records if r['q_mean']      is not None]
ext_noise = [r['noise']       for r in records if r['noise']       is not None]
ext_ok    = [ok[i-1]          for i in ext_idx]
has_ext   = len(ext_idx) > 0
n_ext     = len(ext_idx)
ext_start = ext_idx[0] if has_ext else n   # episode index where extended data starts

# Separate actor loss: filter out zero entries (those are from the early bug)
ext_al_valid_idx = [ext_idx[i] for i,v in enumerate(ext_al) if v != 0.0]
ext_al_valid     = [v for v in ext_al if v != 0.0]
has_actor        = len(ext_al_valid) > 0

# ── Rolling helpers ────────────────────────────────────────────────────────────
def roll(data, w=20):
    return [np.mean(data[max(0,i-w+1):i+1]) for i in range(len(data))]

def roll_ext(idx, data, w=20):
    """Rolling average where x-values are absolute episode indices."""
    result = []
    for i in range(len(data)):
        window = data[max(0, i-w+1):i+1]
        result.append(np.mean(window))
    return result

def roll_sum(data, w=20):
    return [np.sum(data[max(0,i-w+1):i+1]) for i in range(len(data))]

# ── Stage colour mapping ───────────────────────────────────────────────────────
stage_colors = {2.0:'#90CAF9', 3.0:'#A5D6A7', 4.0:'#FFE082', 4.5:'#EF9A9A'}
bar_colors = [stage_colors.get(s,'#BDBDBD') for s in stg]
ok_colors  = ['#2196F3' if o else '#F44336' for o in ok]

ext_bar_colors = ['#2196F3' if v == 1 else '#F44336' for v in ext_ok]

ev_counts = Counter(ev)

# ── Figure: 4 rows × 3 cols = 12 panels ───────────────────────────────────────
fig = plt.figure(figsize=(24, 26))
gs  = GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.33)
fig.patch.set_facecolor('white')

TITLE_FS = 11; LABEL_FS = 9; TICK_FS = 8

def style(ax, title, xlabel='Episode (log index)', ylabel=''):
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.spines[['top','right']].set_visible(False)

def no_data_yet(ax, msg="No data for\npre-R54+ runs.\nWill fill as\ntraining continues."):
    ax.set_facecolor('#F5F5F5')
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='#616161',
            style='italic', multialignment='center')

# ── Panel 1: SR all-time + SR-20 ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(eps, sr,   color='#1565C0', lw=1.5, label='SR all-time', zorder=3)
ax.plot(eps, sr20, color='#43A047', lw=1.5, ls='--', label='SR last-20', zorder=3)
ax.fill_between(eps, sr,   alpha=0.12, color='#1565C0')
ax.fill_between(eps, sr20, alpha=0.10, color='#43A047')
ax.axhline(54.3, color='gray',   ls=':', lw=1.2, label='v5r baseline 54.3%')
ax.axhline(80,   color='orange', ls=':', lw=1.2, label='80% target')
if has_ext:
    ax.axvline(ext_start, color='purple', ls=':', lw=1.2, alpha=0.7, label=f'R54+ metrics start')
ax.set_ylim(0, 105)
ax.legend(fontsize=7, loc='lower right')
style(ax, 'Success Rate (%)', ylabel='%')

# ── Panel 2: Episode Reward ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
rew_roll = roll(rew, 20)
ax.bar(eps, rew, color=ok_colors, alpha=0.55, width=1.0, zorder=2)
ax.plot(eps, rew_roll, color='#E65100', lw=2.0, label='Rolling-20', zorder=3)
ax.axhline(0, color='black', lw=0.8)
if has_ext:
    ax.axvline(ext_start, color='purple', ls=':', lw=1.2, alpha=0.7)
ax.legend(fontsize=7)
style(ax, 'Episode Reward', ylabel='Reward')

# ── Panel 3: Mission Duration ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
dur_roll = roll(dur, 20)
ax.bar(eps, dur, color=bar_colors, alpha=0.7, width=1.0, zorder=2)
ax.plot(eps, dur_roll, color='#4A148C', lw=1.8, label='Rolling-20', zorder=3)
patches = [mpatches.Patch(color=c, label=f'{s} m/s') for s,c in stage_colors.items()]
ax.legend(handles=patches, fontsize=7, title='Stage', title_fontsize=7)
style(ax, 'Mission Duration (s)', ylabel='Seconds')

# ── Panel 4: Max Distance Reached ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.bar(eps, dist, color=bar_colors, alpha=0.75, width=1.0)
ax.axhline(95.0, color='green', ls='--', lw=1.3, label='Turn point 95m')
ax.set_ylim(0, 105)
ax.legend(fontsize=7)
style(ax, 'Max Distance Reached (m)', ylabel='Metres')

# ── Panel 5: Peak Speed per Mission ───────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
spd_roll = roll(spd, 20)
ax.bar(eps, spd, color=bar_colors, alpha=0.65, width=1.0, zorder=2)
ax.plot(eps, spd_roll, color='#BF360C', lw=1.8, label='Rolling-20', zorder=3)
ax.plot(eps, stg, color='black', lw=1.2, ls=':', label='Speed stage', zorder=4)
ax.legend(fontsize=7)
style(ax, 'Peak Speed (m/s)', ylabel='m/s')

# ── Panel 6: Collisions + Avoidance Rate ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
col_roll = roll(col, 20)
ar_roll  = roll(avoid_rate, 20)
ax2 = ax.twinx()
ax.bar(eps,  col,     color='#EF5350', alpha=0.7, width=1.0, label='Collisions', zorder=2)
ax.plot(eps, col_roll, color='#B71C1C', lw=2.0, label='Collisions rolling-20', zorder=3)
ax2.plot(eps, ar_roll, color='#1B5E20', lw=1.8, ls='--', label='Avoidance % rolling-20', zorder=4)
ax2.axhline(90, color='green', ls=':', lw=1.0, alpha=0.6)
ax2.set_ylim(0, 110); ax2.set_ylabel('Avoidance %', fontsize=LABEL_FS, color='#1B5E20')
ax2.tick_params(labelsize=TICK_FS, colors='#1B5E20')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=7, loc='upper right')
style(ax, 'Collisions & Avoidance Rate', ylabel='Collisions')

# ── Panel 7: Failure Mode Breakdown (all actual events from log) ──────────────
ax = fig.add_subplot(gs[2, 0])
win = 20
# Discover all fail event types actually present in the log (excludes success/turn)
all_fail_types = sorted(set(e for e in ev if e not in ('success', 'turn')))
fail_palette   = ['#EF5350','#FF7043','#AB47BC','#29B6F6','#FFCA28','#A5D6A7','#CE93D8']
bottoms = np.zeros(n)
for et, ec in zip(all_fail_types, fail_palette):
    counts = [1 if ev[i]==et else 0 for i in range(n)]
    roll_c = roll_sum(counts, win)
    total  = sum(counts)
    ax.bar(eps, roll_c, bottom=bottoms, color=ec, alpha=0.82, width=1.0,
           label=f'{et} (n={total})')
    bottoms += np.array(roll_c)
# Add note: left_tunnel = exits tunnel bounds (either side — no separate right-wall event)
ax.text(0.02, 0.97,
        "Note: 'left_tunnel' = exits tunnel\nbounds either side (left OR right).\n"
        "No right-wall-only event — both\nare captured under same label.",
        transform=ax.transAxes, fontsize=7, va='top', color='#4A148C',
        bbox=dict(fc='#EDE7F6', ec='#7B1FA2', alpha=0.85, boxstyle='round,pad=0.3'))
ax.legend(fontsize=7, loc='upper right', ncol=2)
style(ax, f'Failure Modes — rolling-{win} (all types)', ylabel='Count per 20 missions')

# ── Panel 8: Q-Value Mean (extended data only) ────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
if has_ext and any(v != 0 for v in ext_q):
    q_roll = roll_ext(ext_idx, ext_q, 20)
    ax.bar(ext_idx, ext_q, color=ext_bar_colors, alpha=0.5, width=1.0, zorder=2)
    ax.plot(ext_idx, q_roll, color='#6A1B9A', lw=2.0, label='Rolling-20', zorder=3)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlim(max(1, ext_start - 5), n + 5)
    ax.annotate(f'← R54+ metrics\n   start ep {ext_start}',
                xy=(ext_start, ax.get_ylim()[0]), xytext=(ext_start + 3, np.mean(ext_q) * 0.5),
                fontsize=8, color='purple', ha='left',
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.2))
    ax.legend(fontsize=7)
    label = f'Mean Q-value — episodes {ext_start}–{n}'
else:
    no_data_yet(ax)
    label = 'Mean Q-value (Q1)'
style(ax, label, ylabel='Q-value')

# ── Panel 9: Critic (TD) Loss (extended data only) ────────────────────────────
ax = fig.add_subplot(gs[2, 2])
if has_ext and any(v > 0 for v in ext_cl):
    cl_roll = roll_ext(ext_idx, ext_cl, 20)
    ax.bar(ext_idx, ext_cl, color='#FF7043', alpha=0.5, width=1.0, zorder=2)
    ax.plot(ext_idx, cl_roll, color='#BF360C', lw=2.0, label='Rolling-20', zorder=3)
    ax.set_xlim(max(1, ext_start - 5), n + 5)
    ax.legend(fontsize=7)
    label = f'Critic (TD) Loss — episodes {ext_start}–{n}'
else:
    no_data_yet(ax)
    label = 'Critic Loss'
style(ax, label, ylabel='MSE Loss')

# ── Panel 10: Actor Policy Loss (extended data only) ──────────────────────────
ax = fig.add_subplot(gs[3, 0])
if has_actor:
    al_roll = roll_ext(ext_al_valid_idx, ext_al_valid, 10)
    ax.bar(ext_al_valid_idx, ext_al_valid, color='#1976D2', alpha=0.5, width=1.0, zorder=2)
    ax.plot(ext_al_valid_idx, al_roll, color='#0D47A1', lw=2.0, label='Rolling-10', zorder=3)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlim(max(1, ext_start - 5), n + 5)
    ax.legend(fontsize=7)
    label = f'Actor Loss (−Q) — episodes {ext_al_valid_idx[0]}–{n}'
elif has_ext:
    no_data_yet(ax,
        "Actor loss = 0 in early run4\n(bug: was filtered incorrectly).\n"
        "Correct values appear from\ncurrent run onward.")
    label = 'Actor Policy Loss (−Q)'
else:
    no_data_yet(ax)
    label = 'Actor Policy Loss (−Q)'
style(ax, label, ylabel='Policy Loss (negative Q)')

# ── Panel 11: Exploration Noise (extended data only) ──────────────────────────
ax = fig.add_subplot(gs[3, 1])
if has_ext and any(v > 0 for v in ext_noise):
    ax.plot(ext_idx, ext_noise, color='#FF6F00', lw=1.8, label='Noise σ', zorder=3)
    ax.fill_between(ext_idx, ext_noise, alpha=0.18, color='#FF6F00')
    ax.set_xlim(max(1, ext_start - 5), n + 5)
    ax.set_ylim(0, max(ext_noise) * 1.2)
    ax.legend(fontsize=7)
    label = f'Exploration Noise σ — ep {ext_start}–{n}'
else:
    no_data_yet(ax)
    label = 'Exploration Noise'
style(ax, label, ylabel='Noise σ')

# ── Panel 12: Collision Count Distribution ────────────────────────────────────
ax = fig.add_subplot(gs[3, 2])
if col_bins:
    bar_c = ['#43A047' if b == 0 else '#EF5350' for b in col_bins]
    bars = ax.bar(col_bins, col_freq, color=bar_c, edgecolor='white', linewidth=0.8, zorder=3)
    for bar, freq in zip(bars, col_freq):
        if freq > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(freq), ha='center', va='bottom', fontsize=7, fontweight='bold')
    zero_pct = col_freq[0] / n * 100 if n > 0 else 0
    ax.text(0.97, 0.97,
            f'Zero-collision missions:\n{col_freq[0]} / {n}  ({zero_pct:.0f}%)',
            transform=ax.transAxes, fontsize=8, ha='right', va='top', color='#1B5E20',
            bbox=dict(fc='#E8F5E9', ec='#388E3C', alpha=0.88, boxstyle='round,pad=0.3'))
    ax.set_xlim(-0.5, max(col_bins) + 0.5)
    ax.set_xticks(col_bins)
else:
    no_data_yet(ax, "No collision data yet")
style(ax, 'Collision Count Distribution\n(# missions by collision count)',
      xlabel='Collisions per Mission', ylabel='# Missions')

# ── Panel 13: Stuck Location Histogram (wide: spans 2 cols) ───────────────────
ax = fig.add_subplot(gs[4, 0:2])
if stuck_dists:
    n_stuck = len(stuck_dists)
    ax.hist(stuck_dists, bins=40, color='#FF7043', edgecolor='white', linewidth=0.6,
            alpha=0.85, zorder=3, label=f'Stuck events  (n={n_stuck})')
    ax.axvspan(10, 18, alpha=0.12, color='#B71C1C', zorder=2,
               label='Dynamic obstacle patrol zone (≈10–18m)')
    med_s = float(np.median(stuck_dists))
    mea_s = float(np.mean(stuck_dists))
    ax.axvline(med_s, color='#4A148C', lw=2.0, ls='--',
               label=f'Median: {med_s:.1f} m')
    ax.axvline(mea_s, color='#0D47A1', lw=1.5, ls=':',
               label=f'Mean: {mea_s:.1f} m')
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8, loc='upper right')
    ax.text(0.01, 0.93,
            f"Stuck events: {n_stuck} / {n} missions ({n_stuck/n*100:.1f}%)\n"
            f"Dynamic pedestrian patrols at X ≈ 13–14m — main hotspot\n"
            f"Peak stuck distance: {float(np.median(stuck_dists)):.1f} m  "
            f"(agent blocked before reaching halfway turn)",
            transform=ax.transAxes, fontsize=8.5, va='top', color='#B71C1C',
            bbox=dict(fc='#FFF8F6', ec='#BF360C', alpha=0.92, boxstyle='round,pad=0.35'))
else:
    no_data_yet(ax, "No stuck events recorded yet")
style(ax, 'Stuck Event Location — Where in the Tunnel Does the Agent Get Stuck?',
      xlabel='Distance from Start (m)', ylabel='# Stuck Events')

# ── Panel 14: Summary Stats ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[4, 2])
ax.axis('off')

total_ok  = sum(ok)
total_col = sum(col)
total_avd = sum(avd)
oar        = total_avd / max(total_col + total_avd, 1) * 100
recent_sr  = np.mean(ok[-20:])*100  if n >= 20 else np.mean(ok)*100
recent_avd = np.mean(avoid_rate[-20:]) if n >= 20 else np.mean(avoid_rate)
recent_col = np.mean(col[-20:]) if n >= 20 else np.mean(col)

q_str  = f"{np.mean(ext_q):.1f}"        if ext_q        else "— (pre-R54+ runs)"
cl_str = f"{np.mean(ext_cl):.1f}"       if ext_cl       else "— (pre-R54+ runs)"
al_str = f"{np.mean(ext_al_valid):.1f}" if ext_al_valid else "— (early bug, fixed)"
noise_last = ext_noise[-1] if ext_noise else 0.0

ev_str = '\n'.join([f'  {k:15s}: {v}'
                    for k, v in sorted(ev_counts.items(), key=lambda x: -x[1])])

n_stuck_tot = sum(1 for e in ev if e == 'stuck')
txt = (
    f"TUNNEL TD3 R54 — SUMMARY\n"
    f"{'═'*34}\n"
    f"Total log entries  : {n}\n"
    f"  Pre-R54+         : {ext_start - 1}  (no Q/loss)\n"
    f"  R54+ extended    : {n_ext}  (ep {ext_start}–{n})\n"
    f"{'─'*34}\n"
    f"Successes          : {total_ok} / {n}  ({total_ok/n*100:.1f}%)\n"
    f"SR last-20         : {recent_sr:.1f}%\n"
    f"{'─'*34}\n"
    f"Avg peak speed     : {np.mean(spd):.2f} m/s\n"
    f"Avg mission time   : {np.mean(dur):.0f} s\n"
    f"Avg dist reached   : {np.mean(dist):.1f} m\n"
    f"{'─'*34}\n"
    f"Total collisions   : {total_col}\n"
    f"Total avoided      : {total_avd}\n"
    f"Avoidance rate     : {oar:.1f}%\n"
    f"Recent col/mission : {recent_col:.1f}\n"
    f"Recent avoid rate  : {recent_avd:.1f}%\n"
    f"{'─'*34}\n"
    f"Stuck failures     : {n_stuck_tot}  ({n_stuck_tot/n*100:.1f}%)\n"
    f"  Hotspot: X≈13–14m (pedestrian)\n"
    f"{'─'*34}\n"
    f"[R54+ metrics]\n"
    f"Mean Q-value (Q1)  : {q_str}\n"
    f"Mean critic loss   : {cl_str}\n"
    f"Mean actor loss    : {al_str}\n"
    f"Current noise σ    : {noise_last:.4f}\n"
    f"{'─'*34}\n"
    f"Event breakdown:\n{ev_str}\n"
    f"{'─'*34}\n"
    f"Arch : 512→512→256+LN+residual\n"
    f"Prox : exp(−d/1.0) × 150\n"
    f"Avoid: proportional, 5m radius\n"
)
ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=7.8,
        family='monospace', va='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='#E3F2FD', ec='#1565C0', alpha=0.9))

fig.suptitle(
    f'Tunnel TD3 R54 — Full Dashboard  '
    f'({n} total log entries · R54+ metrics from ep {ext_start} · '
    f'20-obstacle round-trip · 14 panels)',
    fontsize=14, fontweight='bold', color='#0D47A1', y=1.003
)

plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Dashboard saved : {OUT}")
print(f"Total: {n}  Pre-R54+: {ext_start-1}  Extended: {n_ext}")
print(f"SR: {np.mean(ok)*100:.1f}%  SR-20: {recent_sr:.1f}%  AvdRate: {oar:.1f}%  Q={q_str}  CL={cl_str}  AL={al_str}")
