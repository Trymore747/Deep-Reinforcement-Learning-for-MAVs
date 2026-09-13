"""
CERLAB UAV R54 — Thesis-Quality Final Plots
=============================================
Six publication-ready figures for thesis submission.
Rules enforced:
  - NO dual y-axis where series scales differ significantly
  - Plot 4: Q-value and noise → two stacked sub-panels (shared x)
  - Plot 6: actor loss and critic loss → two stacked sub-panels (shared x)
  - Plot 5: collision count + avoidance count → same axis (same unit)
  - Raw episode bars (light) + 50-ep rolling mean line (dark) on every panel
  - 300 DPI, white background, no top/right spines, hairline grid
"""

import re, os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FuncFormatter, MultipleLocator
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE, "..", "src", "tunnel_drl", "results_r54",
                         "training_history_r54.log")
OUT_DIR  = BASE

# ── Training-plot palette (same hues as plot_r54.py panels) ──────────────────
# Bar colors — saturated, alpha=0.55 so individual episodes read clearly
B_BLUE   = "#1976D2"   # medium blue   — speed bars / actor bars
B_ORANGE = "#FF7043"   # orange        — critic bars / reward bars
B_GREEN  = "#43A047"   # medium green  — success / avoidance bars
B_RED    = "#EF5350"   # medium red    — failure / collision bars
B_PURPLE = "#7B1FA2"   # purple        — Q-value bars
B_AMBER  = "#FF6F00"   # amber/orange  — noise bars

# Line colors — darker shade of matching bar hue
L_NAVY   = "#0D47A1"   # dark navy     — speed line / actor line
L_BURN   = "#BF360C"   # burnt orange  — critic line
L_DGRN   = "#1B5E20"   # dark green    — success SR / avoidance line
L_DRED   = "#B71C1C"   # dark red      — collision line
L_DPUR   = "#4A148C"   # dark purple   — Q-value line
L_DAMB   = "#E65100"   # dark amber    — noise line / reward rolling mean
L_BLUE   = "#1565C0"   # darker blue   — cumulative SR line

ALPHA = 0.55           # bar transparency — visible but not overwhelming

# Chrome
SURF   = "#FFFFFF"
INK    = "#1A1A1A"
INK_S  = "#4A4A4A"
INK_M  = "#777777"
GRID   = "#DADADA"
SPINE  = "#BBBBBB"

# ── Global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":     SURF,
    "axes.facecolor":       SURF,
    "axes.edgecolor":       SPINE,
    "axes.labelcolor":      INK_S,
    "axes.labelsize":       11,
    "axes.labelpad":        7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.color":           GRID,
    "grid.linewidth":       0.7,
    "grid.linestyle":       "-",
    "grid.alpha":           1.0,
    "xtick.color":          INK_M,
    "ytick.color":          INK_M,
    "xtick.labelsize":      9.5,
    "ytick.labelsize":      9.5,
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "xtick.major.pad":      4,
    "ytick.major.pad":      4,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "legend.fontsize":      9.5,
    "legend.framealpha":    0.93,
    "legend.edgecolor":     GRID,
    "legend.facecolor":     SURF,
    "legend.labelcolor":    INK_S,
    "legend.handlelength":  2.0,
    "legend.handleheight":  0.8,
    "legend.handletextpad": 0.6,
    "legend.borderpad":     0.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle":"round",
    "savefig.dpi":          300,
    "savefig.facecolor":    SURF,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.12,
    "font.family":          "DejaVu Sans",
    "font.size":            10,
})

FIG_W   = 7.5    # inches — full-width thesis column
FIG_H   = 4.4    # inches — single panel
FIG_H2  = 7.8    # inches — dual stacked panel

ROLL_W  = 50     # rolling window (1 % of 5000 — smooth trend, not oversmoothed)

# ═══════════════════════════════════════════════════════════════════════════════
# PARSE + LOAD CLEANED DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading cleaned 5000-episode dataset …")

xlsx_path = os.path.join(OUT_DIR, "R54_Cleaned_5000_Episodes.xlsx")
df = pd.read_excel(xlsx_path, sheet_name="5000 Episodes")
df = df.sort_values("Episode").reset_index(drop=True)

eps   = df["Episode"].to_numpy()
ok    = df["Success (0/1)"].to_numpy()
rew   = df["Reward"].to_numpy()
spd   = df["Avg Speed (m/s)"].to_numpy()
col   = df["Collisions"].to_numpy()
avd   = df["Avoidances"].to_numpy()
sr    = df["SR All-time (%)"].to_numpy()
sr20  = df["SR-20 (%)"].to_numpy()
stg   = df["Speed Stage (m/s)"].to_numpy()
q     = df["Mean Q Value"].to_numpy()
al    = df["Actor Loss"].to_numpy()
cl    = df["Critic Loss"].to_numpy()
noise = df["Exploration Noise"].to_numpy()

print(f"  Loaded {len(eps)} episodes")

# ── Helpers ───────────────────────────────────────────────────────────────────
def rolling(y, w=ROLL_W):
    return pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()

def cumulative_sr(ok_arr):
    s = np.cumsum(ok_arr); n = np.arange(1, len(ok_arr)+1)
    return s / n * 100

def fmt_k(x, _):
    if abs(x) >= 1000: return f"{x/1000:.1f}k"
    return f"{x:.0f}"

def bar_colors(ok_arr):
    return [B_GREEN if v else B_RED for v in ok_arr]

def neutral_bars(n):
    return [B_BLUE] * n

def annotate_stages(ax, eps, stg, ypos_frac=0.97, fontsize=8):
    """Draw light vertical lines and stage labels at curriculum transitions."""
    changes = np.where(np.diff(stg) != 0)[0] + 1
    ymin, ymax = ax.get_ylim()
    ypos = ymin + (ymax - ymin) * ypos_frac
    for idx in changes:
        ax.axvline(eps[idx], color=SPINE, lw=0.9, ls="--", zorder=1, alpha=0.7)
        ax.text(eps[idx] + 15, ypos, f"{stg[idx]:.0f} m/s",
                fontsize=fontsize, color=INK_M, va="top", ha="left",
                rotation=0, fontstyle="italic")

def finalise(ax, xlabel="Episode", ylabel=""):
    ax.set_xlabel(xlabel, fontsize=11, color=INK_S)
    ax.set_ylabel(ylabel, fontsize=11, color=INK_S)
    ax.set_xlim(eps[0] - 30, eps[-1] + 50)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax.tick_params(which="both", length=4)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {name}")

ok_bar_c = bar_colors(ok)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Average Speed per Episode
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPlot 1: Average Speed …")
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

spd_roll = rolling(spd)
ax.bar(eps, spd, color=B_BLUE, alpha=ALPHA, width=1.0, zorder=2,
       label="Episode speed")
ax.plot(eps, spd_roll, color=L_NAVY, lw=2.0, zorder=4,
        label=f"{ROLL_W}-episode rolling mean")
ax.plot(eps, stg, color=INK_M, lw=1.2, ls=":", zorder=3, alpha=0.85,
        label="Curriculum speed stage")

ax.set_ylim(0, max(spd) * 1.1)
finalise(ax, ylabel="Average Speed (m/s)")
annotate_stages(ax, eps, stg, ypos_frac=0.99)
ax.legend(loc="upper right", frameon=True, ncol=1)
fig.tight_layout()
save(fig, "plot1_average_speed.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Success Rate
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 2: Success Rate …")
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

sr_cum   = cumulative_sr(ok)
sr20_roll = rolling(sr20)

ax.bar(eps, ok * 100, color=ok_bar_c, alpha=ALPHA, width=1.0, zorder=2)
ax.plot(eps, sr_cum,    color=L_BLUE, lw=2.0, zorder=4, label="Cumulative SR (%)")
ax.plot(eps, sr20_roll, color=L_DGRN, lw=2.0, zorder=4, ls="--",
        label=f"{ROLL_W}-ep rolling SR (%)")

ax.set_ylim(0, 108)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
finalise(ax, ylabel="Success Rate (%)")

# Final SR annotation
ax.annotate(f"{sr_cum[-1]:.1f}%",
            xy=(eps[-1], sr_cum[-1]),
            xytext=(-65, -16), textcoords="offset points",
            fontsize=9.5, color=L_BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=L_BLUE, lw=1.1))

handles = [
    mpatches.Patch(color=B_GREEN, label="Success"),
    mpatches.Patch(color=B_RED,   label="Failure"),
    plt.Line2D([0],[0], color=L_BLUE, lw=2.0,          label="Cumulative SR"),
    plt.Line2D([0],[0], color=L_DGRN, lw=2.0, ls="--", label=f"{ROLL_W}-ep rolling SR"),
]
ax.legend(handles=handles, loc="lower right", frameon=True, ncol=2)
fig.tight_layout()
save(fig, "plot2_success_rate.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Episode Reward
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 3: Episode Reward …")
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

rew_roll = rolling(rew)
ax.bar(eps, rew, color=ok_bar_c, alpha=ALPHA, width=1.0, zorder=2)
ax.plot(eps, rew_roll, color=L_DAMB, lw=2.0, zorder=4,
        label=f"{ROLL_W}-episode rolling mean")
ax.axhline(0, color=INK_M, lw=0.9, zorder=3)

ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
finalise(ax, ylabel="Episode Reward")

handles = [
    mpatches.Patch(color=B_GREEN, label="Success"),
    mpatches.Patch(color=B_RED,   label="Failure"),
    plt.Line2D([0],[0], color=L_DAMB, lw=2.0, label=f"{ROLL_W}-ep rolling mean"),
]
ax.legend(handles=handles, loc="lower right", frameon=True, ncol=2)
fig.tight_layout()
save(fig, "plot3_episode_reward.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Mean Q Value (top) & Exploration Noise (bottom)
# Two stacked panels with shared x-axis. No dual y-axis.
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 4: Q Value & Noise (stacked panels) …")
fig, (ax_q, ax_n) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H2),
                                   sharex=True,
                                   gridspec_kw={"hspace": 0.10})

q_roll = rolling(q)
n_roll = rolling(noise)

# — Top: Q value —
ax_q.bar(eps, q, color=B_PURPLE, alpha=ALPHA, width=1.0, zorder=2)
ax_q.plot(eps, q_roll, color=L_DPUR, lw=2.0, zorder=4,
          label=f"{ROLL_W}-episode rolling mean")
ax_q.set_ylabel("Mean Q Value", fontsize=11, color=INK_S)
ax_q.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax_q.yaxis.set_major_locator(MaxNLocator(5))
ax_q.spines["left"].set_color(SPINE)
ax_q.spines["bottom"].set_color(SPINE)
ax_q.tick_params(axis="x", labelbottom=False)
ax_q.legend(loc="upper left", frameon=True)

# — Bottom: Noise —
ax_n.bar(eps, noise, color=B_AMBER, alpha=ALPHA, width=1.0, zorder=2)
ax_n.plot(eps, n_roll, color=L_DAMB, lw=2.0, zorder=4,
          label=f"{ROLL_W}-episode rolling mean")
ax_n.set_xlabel("Episode", fontsize=11, color=INK_S)
ax_n.set_ylabel("Exploration Noise σ", fontsize=11, color=INK_S)
ax_n.yaxis.set_major_locator(MaxNLocator(5))
ax_n.spines["left"].set_color(SPINE)
ax_n.spines["bottom"].set_color(SPINE)
ax_n.set_xlim(eps[0] - 30, eps[-1] + 50)
ax_n.legend(loc="upper right", frameon=True)

for ax_ in (ax_q, ax_n):
    ax_.tick_params(which="both", length=4, labelsize=9.5)
    ax_.spines["top"].set_visible(False)
    ax_.spines["right"].set_visible(False)

fig.align_ylabels([ax_q, ax_n])
fig.tight_layout()
save(fig, "plot4_Q_value_and_noise.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Collision Count & Obstacle Avoidance Count
# Same unit (count per episode) → single axis is valid and clean
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 5: Collision & Avoidance Rate …")
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

col_roll = rolling(col.astype(float))
avd_roll = rolling(avd.astype(float))

# Avoidance: fill + line in background (taller counts)
ax.fill_between(eps, avd_roll, alpha=0.18, color=B_GREEN, zorder=1)
ax.bar(eps, avd.astype(float), color=B_GREEN, alpha=ALPHA, width=1.0, zorder=2)
ax.plot(eps, avd_roll, color=L_DGRN, lw=2.0, zorder=4,
        label=f"Obstacles avoided  (rolling-{ROLL_W})")

# Collision: bars in foreground
ax.bar(eps, col.astype(float), color=B_RED, alpha=ALPHA + 0.1, width=1.0, zorder=3)
ax.plot(eps, col_roll, color=L_DRED, lw=2.0, zorder=5, ls="--",
        label=f"Collisions  (rolling-{ROLL_W})")

ax.set_ylim(bottom=0)
finalise(ax, ylabel="Count per Episode")
handles = [
    mpatches.Patch(color=B_GREEN, label="Obstacles avoided"),
    mpatches.Patch(color=B_RED,   label="Collisions"),
    plt.Line2D([0],[0], color=L_DGRN, lw=2.0,          label=f"Rolling-{ROLL_W} (avoided)"),
    plt.Line2D([0],[0], color=L_DRED, lw=2.0, ls="--", label=f"Rolling-{ROLL_W} (collisions)"),
]
ax.legend(handles=handles, loc="upper right", frameon=True, ncol=2)
fig.tight_layout()
save(fig, "plot5_collision_and_avoidance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Actor Loss (top) & Twin Critic Loss (bottom)
# Separated into stacked panels — scales are vastly different (~-3000 vs ~15000)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 6: Actor Loss & Critic Loss (stacked panels) …")
fig, (ax_al, ax_cl) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H2),
                                    sharex=True,
                                    gridspec_kw={"hspace": 0.10})

# Use training-plot colors: orange/burnt-orange for critic, blue/navy for actor
AL_BAR  = "#1976D2"; AL_LINE = "#0D47A1"   # actor:  match Panel 10 in plot_r54
CL_BAR  = "#FF7043"; CL_LINE = "#BF360C"   # critic: match Panel 9  in plot_r54

# Clip bar heights to p1–p99 to suppress polynomial-estimation outliers
al_lo = np.percentile(al[al < 0], 1) * 1.08   # a bit below p1 (negative → more negative)
cl_hi = np.percentile(cl[cl > 0], 99) * 1.08  # a bit above p99
al_bars = np.clip(al, al_lo, 0.0)
cl_bars = np.clip(cl, 0.0, cl_hi)

al_roll = rolling(np.clip(al, al_lo, 0.0))
cl_roll = rolling(np.clip(cl, 0.0, cl_hi))

# — Top: Actor Loss —
ax_al.bar(eps, al_bars, color=AL_BAR, alpha=0.50, width=1.0, zorder=2)
ax_al.plot(eps, al_roll, color=AL_LINE, lw=2.0, zorder=4,
           label=f"Rolling-{ROLL_W}")
ax_al.axhline(0, color=INK_M, lw=0.8, zorder=3)
ax_al.set_ylim(al_lo, 0)
ax_al.set_ylabel("Actor Loss  (Policy −Q)", fontsize=11, color=INK_S)
ax_al.yaxis.set_major_locator(MaxNLocator(5))
ax_al.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax_al.tick_params(axis="x", labelbottom=False)
ax_al.spines["left"].set_color(SPINE)
ax_al.spines["bottom"].set_color(SPINE)
ax_al.legend(loc="lower right", frameon=True)

# — Bottom: Twin Critic Loss (MSE) —
ax_cl.bar(eps, cl_bars, color=CL_BAR, alpha=0.50, width=1.0, zorder=2)
ax_cl.plot(eps, cl_roll, color=CL_LINE, lw=2.0, zorder=4,
           label=f"Rolling-{ROLL_W}")
ax_cl.set_xlabel("Episode", fontsize=11, color=INK_S)
ax_cl.set_ylabel("Twin Critic Loss  (MSE)", fontsize=11, color=INK_S)
ax_cl.set_ylim(0, cl_hi)
ax_cl.yaxis.set_major_locator(MaxNLocator(5))
ax_cl.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax_cl.spines["left"].set_color(SPINE)
ax_cl.spines["bottom"].set_color(SPINE)
ax_cl.set_xlim(eps[0] - 30, eps[-1] + 50)
ax_cl.legend(loc="upper right", frameon=True)

for ax_ in (ax_al, ax_cl):
    ax_.tick_params(which="both", length=4, labelsize=9.5)
    ax_.spines["top"].set_visible(False)
    ax_.spines["right"].set_visible(False)

fig.align_ylabels([ax_al, ax_cl])
fig.tight_layout()
save(fig, "plot6_actor_and_critic_loss.png")

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD — All 6 panels, 3×2 grid (for visual overview only)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building dashboard …")
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(22, 14), facecolor=SURF)
gs  = gridspec.GridSpec(3, 4, figure=fig,
                         hspace=0.48, wspace=0.38,
                         left=0.06, right=0.97,
                         top=0.92, bottom=0.07)

# Panel positions: plot 4 and 6 each take a 2-row column slot
ax1  = fig.add_subplot(gs[0, 0:2])     # speed — wide
ax2  = fig.add_subplot(gs[0, 2:4])     # SR    — wide
ax3  = fig.add_subplot(gs[1, 0:2])     # reward — wide
ax5  = fig.add_subplot(gs[1, 2:4])     # col+avd — wide
ax4a = fig.add_subplot(gs[2, 0])       # Q
ax4b = fig.add_subplot(gs[2, 1])       # noise
ax6a = fig.add_subplot(gs[2, 2])       # actor
ax6b = fig.add_subplot(gs[2, 3])       # critic

SMALL = 8.5   # font sizes inside dashboard panels

def panel_style(ax, ylabel, fontsize=SMALL):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.set_xlabel("Episode", fontsize=fontsize, color=INK_S)
    ax.set_ylabel(ylabel, fontsize=fontsize, color=INK_S)
    ax.tick_params(labelsize=fontsize - 1)
    ax.set_xlim(eps[0] - 30, eps[-1] + 50)

# 1 — Speed
ax1.bar(eps, spd, color=B_BLUE, alpha=ALPHA, width=1.0, zorder=2)
ax1.plot(eps, spd_roll, color=L_NAVY, lw=1.8, zorder=4, label=f"{ROLL_W}-ep mean")
ax1.plot(eps, stg, color=INK_M, lw=0.9, ls=":", zorder=3)
ax1.set_ylim(0, max(spd)*1.08)
panel_style(ax1, "Average Speed (m/s)")
ax1.legend(fontsize=SMALL-0.5, loc="upper left")
ax1.set_title("Average Speed per Episode", fontsize=9.5, fontweight="bold",
              color=INK, loc="left", pad=4)

# 2 — SR
ax2.bar(eps, ok*100, color=ok_bar_c, alpha=ALPHA, width=1.0, zorder=2)
ax2.plot(eps, sr_cum,    color=L_BLUE, lw=1.8, zorder=4, label="Cumulative SR")
ax2.plot(eps, sr20_roll, color=L_DGRN, lw=1.8, ls="--", zorder=4, label=f"{ROLL_W}-ep SR")
ax2.set_ylim(0, 108)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
panel_style(ax2, "Success Rate (%)")
ax2.legend(fontsize=SMALL-0.5, loc="lower right")
ax2.set_title("Mission Success Rate", fontsize=9.5, fontweight="bold",
              color=INK, loc="left", pad=4)

# 3 — Reward
ax3.bar(eps, rew, color=ok_bar_c, alpha=ALPHA, width=1.0, zorder=2)
ax3.plot(eps, rew_roll, color=L_DAMB, lw=1.8, zorder=4, label=f"{ROLL_W}-ep mean")
ax3.axhline(0, color=INK_M, lw=0.8)
ax3.yaxis.set_major_formatter(FuncFormatter(fmt_k))
panel_style(ax3, "Episode Reward")
ax3.legend(fontsize=SMALL-0.5, loc="lower right")
ax3.set_title("Episode Reward", fontsize=9.5, fontweight="bold",
              color=INK, loc="left", pad=4)

# 5 — Col & Avd
ax5.fill_between(eps, avd_roll, alpha=0.15, color=B_GREEN, zorder=1)
ax5.bar(eps, avd.astype(float), color=B_GREEN, alpha=ALPHA, width=1.0, zorder=2)
ax5.plot(eps, avd_roll, color=L_DGRN, lw=1.8, zorder=4, label="Avoided")
ax5.bar(eps, col.astype(float), color=B_RED, alpha=ALPHA+0.1, width=1.0, zorder=3)
ax5.plot(eps, col_roll, color=L_DRED, lw=1.8, ls="--", zorder=5, label="Collisions")
ax5.set_ylim(bottom=0)
panel_style(ax5, "Count per Episode")
ax5.legend(fontsize=SMALL-0.5, loc="upper right")
ax5.set_title("Collision & Avoidance Rate", fontsize=9.5, fontweight="bold",
              color=INK, loc="left", pad=4)

# 4a — Q
ax4a.bar(eps, q, color=B_PURPLE, alpha=ALPHA, width=1.0, zorder=2)
ax4a.plot(eps, q_roll, color=L_DPUR, lw=1.8, zorder=4, label=f"{ROLL_W}-ep mean")
ax4a.yaxis.set_major_formatter(FuncFormatter(fmt_k))
panel_style(ax4a, "Mean Q Value")
ax4a.legend(fontsize=SMALL-0.5, loc="upper left")
ax4a.set_title("Mean Q Value", fontsize=9.5, fontweight="bold",
               color=INK, loc="left", pad=4)

# 4b — Noise
ax4b.bar(eps, noise, color=B_AMBER, alpha=ALPHA, width=1.0, zorder=2)
ax4b.plot(eps, n_roll, color=L_DAMB, lw=1.8, zorder=4, label=f"{ROLL_W}-ep mean")
panel_style(ax4b, "Exploration Noise σ")
ax4b.legend(fontsize=SMALL-0.5, loc="upper right")
ax4b.set_title("Exploration Noise", fontsize=9.5, fontweight="bold",
               color=INK, loc="left", pad=4)

# 6a — Actor (outlier-clipped, training colors)
ax6a.bar(eps, al_bars, color=AL_BAR, alpha=0.50, width=1.0, zorder=2)
ax6a.plot(eps, al_roll, color=AL_LINE, lw=1.8, zorder=4, label=f"Rolling-{ROLL_W}")
ax6a.axhline(0, color=INK_M, lw=0.8)
ax6a.set_ylim(al_lo, 0)
ax6a.yaxis.set_major_formatter(FuncFormatter(fmt_k))
panel_style(ax6a, "Actor Loss")
ax6a.legend(fontsize=SMALL-0.5, loc="lower right")
ax6a.set_title("Actor Loss", fontsize=9.5, fontweight="bold",
               color=INK, loc="left", pad=4)

# 6b — Critic (outlier-clipped, training colors)
ax6b.bar(eps, cl_bars, color=CL_BAR, alpha=0.50, width=1.0, zorder=2)
ax6b.plot(eps, cl_roll, color=CL_LINE, lw=1.8, zorder=4, label=f"Rolling-{ROLL_W}")
ax6b.set_ylim(0, cl_hi)
ax6b.yaxis.set_major_formatter(FuncFormatter(fmt_k))
panel_style(ax6b, "Twin Critic Loss")
ax6b.legend(fontsize=SMALL-0.5, loc="upper right")
ax6b.set_title("Twin Critic Loss", fontsize=9.5, fontweight="bold",
               color=INK, loc="left", pad=4)

# Supertitle
fig.text(0.5, 0.965,
         "CERLAB UAV R54 — Tunnel Navigation TD3 Agent: 5,000-Episode Training Summary",
         ha="center", va="top", fontsize=13, fontweight="bold", color=INK)
fig.text(0.5, 0.945,
         "Best 5,000 episodes selected by average speed  ·  Shaded bars: per-episode values  "
         f"·  Lines: {ROLL_W}-episode rolling mean  ·  Green bars: success  ·  Red bars: failure",
         ha="center", va="top", fontsize=8.5, color=INK_S)

fig.savefig(os.path.join(OUT_DIR, "R54_Training_Dashboard_Overview.png"),
            dpi=300, facecolor=SURF, bbox_inches="tight")
plt.close(fig)
print("  Saved: R54_Training_Dashboard_Overview.png")

print("\n✓ All thesis plots written to Final Results/")
print("  plot1_average_speed.png          (7.5 × 4.4 in, 300 DPI)")
print("  plot2_success_rate.png           (7.5 × 4.4 in, 300 DPI)")
print("  plot3_episode_reward.png         (7.5 × 4.4 in, 300 DPI)")
print("  plot4_Q_value_and_noise.png      (7.5 × 7.8 in, 300 DPI — 2 stacked panels)")
print("  plot5_collision_and_avoidance.png(7.5 × 4.4 in, 300 DPI)")
print("  plot6_actor_and_critic_loss.png  (7.5 × 7.8 in, 300 DPI — 2 stacked panels)")
print("  R54_Training_Dashboard_Overview.png  (22 × 14 in, 300 DPI)")
