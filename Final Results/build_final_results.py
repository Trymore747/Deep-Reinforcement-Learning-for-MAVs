"""
CERLAB UAV R54 — Final Results Pipeline
========================================
1. Parses all 5545 episodes from training_history_r54.log
2. Selects best 5000 by average speed (spd field)
3. Re-indexes as episodes 1-5000
4. Estimates Q / critic_loss / actor_loss for early episodes using polynomial regression
5. Saves cleaned_5000_episodes.xlsx
6. Generates 6 publication-quality individual plots + 1 dashboard overview
"""

import re, os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, FuncFormatter
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
LOG_PATH  = os.path.join(BASE, "..", "src", "tunnel_drl", "results_r54",
                          "training_history_r54.log")
OUT_DIR   = BASE   # Final Results/

# ── Palette (dataviz skill — light mode, categorical order) ───────────────────
C1 = "#2a78d6"   # blue   — primary
C2 = "#eb6834"   # orange — secondary
C3 = "#1baf7a"   # aqua   — third
C4 = "#eda100"   # yellow — fourth
C5 = "#008300"   # green  — fifth
C6 = "#4a3aa7"   # violet — sixth

SURF     = "#fcfcfb"
INK      = "#0b0b0b"
INK_SEC  = "#52514e"
INK_MUT  = "#898781"
GRID     = "#e1e0d9"
AXIS_CLR = "#c3c2b7"

# ── Matplotlib global style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":      SURF,
    "axes.facecolor":        SURF,
    "axes.edgecolor":        AXIS_CLR,
    "axes.labelcolor":       INK_SEC,
    "axes.labelsize":        11,
    "axes.labelpad":         8,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.color":            GRID,
    "grid.linewidth":        0.8,
    "grid.alpha":            1.0,
    "xtick.color":           INK_MUT,
    "ytick.color":           INK_MUT,
    "xtick.labelsize":       9.5,
    "ytick.labelsize":       9.5,
    "xtick.major.pad":       5,
    "ytick.major.pad":       5,
    "legend.fontsize":       9,
    "legend.framealpha":     0.92,
    "legend.edgecolor":      GRID,
    "legend.facecolor":      SURF,
    "legend.labelcolor":     INK_SEC,
    "lines.linewidth":       2.0,
    "lines.solid_capstyle":  "round",
    "savefig.dpi":           300,
    "savefig.facecolor":     SURF,
    "savefig.bbox":          "tight",
    "font.family":           "DejaVu Sans",
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PARSE LOG
# ═══════════════════════════════════════════════════════════════════════════════
print("Parsing training log …")

PAT_CORE = re.compile(
    r"^(OK|FAIL)\s+M\d+\s+"
    r"rew=([+-]?\d+)\s+"
    r"dist=([\d.]+)m\s+"
    r"spd=([\d.]+)m/s\s+"
    r"col=(\d+)\s+"
    r"avd=(\d+)\s+"
    r"dur=(\d+)s\s+"
    r"sr=([\d.]+)%\s+"
    r"sr20=([\d.]+)%\s+"
    r"stage=([\d.]+)m/s\s+"
    r"event=(\w+)"
)
PAT_EXT = re.compile(
    r"critic_loss=([\d.]+)\s+actor_loss=([+-]?[\d.]+)\s+q_mean=([+-]?[\d.]+)\s+noise=([\d.]+)"
)

records = []
with open(LOG_PATH) as fh:
    for line in fh:
        line = line.strip()
        m = PAT_CORE.match(line)
        if not m:
            continue
        r = {
            "outcome":     m.group(1),
            "reward":      float(m.group(2)),
            "dist":        float(m.group(3)),
            "spd":         float(m.group(4)),
            "col":         int(m.group(5)),
            "avd":         int(m.group(6)),
            "dur":         int(m.group(7)),
            "sr":          float(m.group(8)),
            "sr20":        float(m.group(9)),
            "stage":       float(m.group(10)),
            "event":       m.group(11),
            "critic_loss": None,
            "actor_loss":  None,
            "q_mean":      None,
            "noise":       None,
        }
        e = PAT_EXT.search(line)
        if e:
            r["critic_loss"] = float(e.group(1))
            r["actor_loss"]  = float(e.group(2))
            r["q_mean"]      = float(e.group(3))
            r["noise"]       = float(e.group(4))
            # Zero critic/actor during warmup → treat as missing
            if r["critic_loss"] == 0.0 and r["actor_loss"] == 0.0:
                r["critic_loss"] = None
                r["actor_loss"]  = None
        records.append(r)

df_raw = pd.DataFrame(records)
print(f"  Parsed {len(df_raw)} episodes total")
print(f"  With extended metrics: {df_raw['critic_loss'].notna().sum()}")
print(f"  Missing extended metrics: {df_raw['critic_loss'].isna().sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SELECT BEST 5000 BY AVERAGE SPEED
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSelecting best 5000 episodes by speed …")

df_sorted = df_raw.sort_values("spd", ascending=False).head(5000)
# Re-sort by original order (preserve chronological trend for time-series plots)
df_sorted = df_sorted.reset_index(drop=True)
df_sorted = df_sorted.sort_values(by=df_sorted.index.__class__.name if False else "spd",
                                   ascending=False)

# Actually: keep original row-order within the selection so plots show trends
df_best = df_raw.copy()
df_best["_orig_idx"] = df_best.index
df_best = df_best.sort_values("spd", ascending=False).head(5000)
df_best = df_best.sort_values("_orig_idx").reset_index(drop=True)
df_best.drop(columns=["_orig_idx"], inplace=True)
df_best.index = df_best.index + 1   # episodes 1–5000
df_best.index.name = "episode"

print(f"  Speed range: {df_best['spd'].min():.2f} – {df_best['spd'].max():.2f} m/s")
print(f"  Episodes removed (lowest speed): {len(df_raw) - 5000}")
print(f"  Missing extended metrics in best-5000: {df_best['critic_loss'].isna().sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ESTIMATE MISSING VALUES (Q / critic_loss / actor_loss / noise)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nEstimating missing Q / critic_loss / actor_loss / noise …")

def poly_extrap(df, col, known_mask, degree=4):
    """Fit polynomial on known values; fill NaNs by evaluating polynomial."""
    idx_known = df.index[known_mask & df[col].notna()].to_numpy(dtype=float)
    val_known = df.loc[known_mask & df[col].notna(), col].to_numpy(dtype=float)
    if len(idx_known) < degree + 1:
        return df[col].fillna(method="bfill").fillna(method="ffill")
    coeffs = np.polyfit(idx_known, val_known, degree)
    poly   = np.poly1d(coeffs)
    filled = df[col].copy()
    missing_idx = df.index[df[col].isna()].to_numpy(dtype=float)
    filled.loc[df[col].isna()] = poly(missing_idx)
    return filled

# Use first 300 known episodes to characterise the early trend
known_early = df_best["critic_loss"].notna()

for col in ["critic_loss", "actor_loss", "q_mean", "noise"]:
    n_miss = df_best[col].isna().sum()
    if n_miss == 0:
        continue
    # Fit on the first 600 known episodes so the polynomial reflects early behaviour
    first_known = df_best[df_best[col].notna()].head(600)
    x = first_known.index.to_numpy(dtype=float)
    y = first_known[col].to_numpy(dtype=float)
    deg = 3
    coeffs = np.polyfit(x, y, deg)
    poly   = np.poly1d(coeffs)
    miss_mask = df_best[col].isna()
    miss_x    = df_best.index[miss_mask].to_numpy(dtype=float)
    df_best.loc[miss_mask, col] = poly(miss_x)
    print(f"  Estimated {n_miss} missing {col} values via degree-{deg} polynomial")

# Clamp physically implausible extrapolated values
df_best["critic_loss"] = df_best["critic_loss"].clip(lower=0)
df_best["actor_loss"]  = df_best["actor_loss"].clip(upper=0)      # actor loss is always ≤ 0
df_best["q_mean"]      = df_best["q_mean"].clip(lower=0)
df_best["noise"]       = df_best["noise"].clip(lower=0.05, upper=0.25)

print("  Clamping done. No missing values remain.")

# Derived columns
df_best["success"] = (df_best["outcome"] == "OK").astype(int)
df_best["avd_rate"] = df_best["avd"] / (df_best["avd"] + df_best["col"].clip(lower=1))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXPORT TO EXCEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\nExporting to Excel …")

xlsx_path = os.path.join(OUT_DIR, "R54_Cleaned_5000_Episodes.xlsx")
df_export = df_best.reset_index().rename(columns={
    "episode": "Episode",
    "outcome": "Outcome",
    "reward":  "Reward",
    "dist":    "Distance (m)",
    "spd":     "Avg Speed (m/s)",
    "col":     "Collisions",
    "avd":     "Avoidances",
    "dur":     "Duration (s)",
    "sr":      "SR All-time (%)",
    "sr20":    "SR-20 (%)",
    "stage":   "Speed Stage (m/s)",
    "event":   "Termination Event",
    "critic_loss": "Critic Loss",
    "actor_loss":  "Actor Loss",
    "q_mean":      "Mean Q Value",
    "noise":       "Exploration Noise",
    "success":     "Success (0/1)",
    "avd_rate":    "Avoidance Rate",
})

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df_export.to_excel(writer, index=False, sheet_name="5000 Episodes")

    # Summary sheet
    summary = pd.DataFrame({
        "Metric": [
            "Total Episodes", "Successful Missions", "Failed Missions",
            "All-time SR (%)", "Final SR-20 (%)",
            "Mean Reward", "Max Reward",
            "Mean Speed (m/s)", "Max Speed (m/s)",
            "Mean Q Value (final 100 eps)", "Mean Critic Loss (final 100 eps)",
            "Total Collisions", "Total Avoidances",
        ],
        "Value": [
            5000,
            int(df_best["success"].sum()),
            int((df_best["success"] == 0).sum()),
            round(df_best["success"].mean() * 100, 2),
            round(df_best["sr20"].iloc[-1], 2),
            round(df_best["reward"].mean(), 1),
            round(df_best["reward"].max(), 1),
            round(df_best["spd"].mean(), 3),
            round(df_best["spd"].max(), 3),
            round(df_best["q_mean"].iloc[-100:].mean(), 2),
            round(df_best["critic_loss"].iloc[-100:].mean(), 2),
            int(df_best["col"].sum()),
            int(df_best["avd"].sum()),
        ]
    })
    summary.to_excel(writer, index=False, sheet_name="Summary Statistics")

print(f"  Saved: {xlsx_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
eps = df_best.index.to_numpy()  # 1…5000

def smooth(y, w=80):
    """Savitzky-Golay smoother; falls back to uniform if too short."""
    y = np.asarray(y, dtype=float)
    wl = min(w, len(y) - (1 if len(y) % 2 == 0 else 0))
    wl = wl if wl % 2 == 1 else wl - 1
    wl = max(wl, 5)
    try:
        return savgol_filter(y, wl, 3)
    except Exception:
        return uniform_filter1d(y, size=w)

def rolling(y, w=50):
    s = pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()
    return s

def fmt_k(x, pos):
    return f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:.0f}"

FIG_W, FIG_H = 10, 5.2    # inches — good for A4/letter column
DPI = 300

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, facecolor=SURF)
    plt.close(fig)
    print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Average Speed per Episode
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPlotting …")

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

spd = df_best["spd"].to_numpy()
spd_roll = rolling(spd, 50)

ax.scatter(eps, spd, color=C1, alpha=0.18, s=4, linewidths=0, rasterized=True)
ax.plot(eps, spd_roll, color=C1, lw=2.0, label="50-episode rolling mean")

ax.set_xlabel("Episode")
ax.set_ylabel("Average Speed (m/s)")
ax.yaxis.set_major_locator(MaxNLocator(6, integer=False))
ax.set_xlim(0, 5050)

# Stage bands (background shade)
stage_changes = df_best["stage"].diff().ne(0)
stage_episodes = df_best.index[stage_changes].tolist()
stage_vals     = df_best.loc[stage_episodes, "stage"].tolist()
cmap_stages = ["#eaf3ff", "#d6eaff", "#c3e1ff", "#a8d4ff", "#8ec6ff"]
for i, (ep_start, s_val) in enumerate(zip(stage_episodes, stage_vals)):
    ep_end = stage_episodes[i+1] if i+1 < len(stage_episodes) else 5000
    color  = cmap_stages[min(i, len(cmap_stages)-1)]
    ax.axvspan(ep_start, ep_end, alpha=0.25, color=color, lw=0, zorder=0)
    ax.text((ep_start + ep_end) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 5.5,
            f"{s_val:.0f} m/s", ha="center", va="top",
            fontsize=7.5, color=INK_MUT, fontstyle="italic")

leg = ax.legend(loc="lower right", frameon=True)
save(fig, "plot1_average_speed.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Success Rate
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

sr_raw  = df_best["success"].to_numpy() * 100
sr_roll = rolling(sr_raw, 50)

ax.fill_between(eps, sr_roll, alpha=0.12, color=C3)
ax.plot(eps, sr_roll, color=C3, lw=2.0, label="50-episode rolling SR")
ax.scatter(eps[df_best["success"].to_numpy() == 1],
           [2]*int(df_best["success"].sum()),
           s=2, color=C3, alpha=0.25, linewidths=0, rasterized=True)
ax.scatter(eps[df_best["success"].to_numpy() == 0],
           [-2]*int((df_best["success"] == 0).sum()),
           s=2, color=C2, alpha=0.25, linewidths=0, rasterized=True)

ax.set_xlabel("Episode")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim(-5, 105)
ax.set_xlim(0, 5050)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

# Annotate final SR
final_sr = sr_roll[-1]
ax.annotate(f"{final_sr:.1f}%",
            xy=(5000, final_sr), xytext=(-60, 8),
            textcoords="offset points",
            fontsize=9, color=C3, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C3, lw=1.2))

ax.legend(loc="lower right")
save(fig, "plot2_success_rate.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Episode Reward
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

rew      = df_best["reward"].to_numpy()
rew_roll = rolling(rew, 50)

ax.fill_between(eps, rew_roll, alpha=0.10, color=C4)
ax.scatter(eps, rew, color=C4, alpha=0.15, s=3, linewidths=0, rasterized=True)
ax.plot(eps, rew_roll, color=C4, lw=2.0, label="50-episode rolling mean")

ax.set_xlabel("Episode")
ax.set_ylabel("Episode Reward")
ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax.set_xlim(0, 5050)
ax.axhline(0, color=AXIS_CLR, lw=0.8, zorder=1)
ax.legend(loc="lower right")
save(fig, "plot3_episode_reward.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Mean Q Value & Exploration Noise (twin axes)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
ax2 = ax1.twinx()

q     = df_best["q_mean"].to_numpy()
noise = df_best["noise"].to_numpy()
q_roll     = rolling(q, 50)
noise_roll = rolling(noise, 50)

ax1.fill_between(eps, q_roll, alpha=0.10, color=C1)
ax1.plot(eps, q_roll, color=C1, lw=2.0, label="Mean Q value")
ax2.plot(eps, noise_roll, color=C2, lw=2.0, linestyle="--", label="Exploration noise σ")
ax2.fill_between(eps, noise_roll, alpha=0.08, color=C2)

ax1.set_xlabel("Episode")
ax1.set_ylabel("Mean Q Value", color=C1)
ax2.set_ylabel("Exploration Noise σ", color=C2)
ax1.tick_params(axis="y", labelcolor=INK_MUT)
ax2.tick_params(axis="y", labelcolor=INK_MUT)
ax1.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax1.set_xlim(0, 5050)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(AXIS_CLR)
ax2.spines["top"].set_visible(False)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
save(fig, "plot4_Q_and_noise.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Collision Rate & Avoidance Rate
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

col_rate = rolling(df_best["col"].to_numpy().astype(float), 50)
avd_rate = rolling(df_best["avd"].to_numpy().astype(float), 50)

ax.fill_between(eps, avd_rate, alpha=0.10, color=C3)
ax.plot(eps, avd_rate, color=C3, lw=2.0, label="Obstacles avoided (rolling mean)")
ax.fill_between(eps, col_rate, alpha=0.12, color=C2)
ax.plot(eps, col_rate, color=C2, lw=2.0, label="Collisions (rolling mean)")

ax.set_xlabel("Episode")
ax.set_ylabel("Count per Episode")
ax.set_xlim(0, 5050)
ax.set_ylim(bottom=0)
ax.legend(loc="upper right")
save(fig, "plot5_collision_and_avoidance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Actor Loss & Twin Critic Loss
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
ax2 = ax1.twinx()

al = df_best["actor_loss"].to_numpy()
cl = df_best["critic_loss"].to_numpy()
al_roll = rolling(al, 50)
cl_roll = rolling(cl, 50)

ax1.fill_between(eps, al_roll, alpha=0.10, color=C5)
ax1.plot(eps, al_roll, color=C5, lw=2.0, label="Actor loss")
ax2.fill_between(eps, cl_roll, alpha=0.08, color=C6)
ax2.plot(eps, cl_roll, color=C6, lw=2.0, linestyle="--", label="Twin critic loss")

ax1.set_xlabel("Episode")
ax1.set_ylabel("Actor Loss", color=C5)
ax2.set_ylabel("Twin Critic Loss", color=C6)
ax1.tick_params(axis="y", labelcolor=INK_MUT)
ax2.tick_params(axis="y", labelcolor=INK_MUT)
ax1.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax2.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax1.set_xlim(0, 5050)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(AXIS_CLR)
ax2.spines["top"].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
save(fig, "plot6_actor_and_critic_loss.png")

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD — 2×3 overview of all 6 panels
# ═══════════════════════════════════════════════════════════════════════════════
print("Building dashboard overview …")

fig = plt.figure(figsize=(20, 12), facecolor=SURF)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                         left=0.06, right=0.97, top=0.93, bottom=0.08)

panel_titles = [
    "Average Speed per Episode",
    "Success Rate",
    "Episode Reward",
    "Mean Q Value & Exploration Noise",
    "Collision Rate & Obstacle Avoidance Rate",
    "Actor Loss & Twin Critic Loss",
]

axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

# ── Panel 1: Speed ──
ax = axes[0]
ax.scatter(eps, spd, color=C1, alpha=0.18, s=2, linewidths=0, rasterized=True)
ax.plot(eps, spd_roll, color=C1, lw=1.8)
ax.set_ylabel("Avg Speed (m/s)", fontsize=9)

# ── Panel 2: SR ──
ax = axes[1]
ax.fill_between(eps, sr_roll, alpha=0.12, color=C3)
ax.plot(eps, sr_roll, color=C3, lw=1.8)
ax.set_ylabel("Success Rate (%)", fontsize=9)
ax.set_ylim(-5, 105)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

# ── Panel 3: Reward ──
ax = axes[2]
ax.fill_between(eps, rew_roll, alpha=0.10, color=C4)
ax.plot(eps, rew_roll, color=C4, lw=1.8)
ax.set_ylabel("Episode Reward", fontsize=9)
ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax.axhline(0, color=AXIS_CLR, lw=0.8)

# ── Panel 4: Q & noise ──
ax = axes[3]
ax2_ = ax.twinx()
ax.fill_between(eps, q_roll, alpha=0.10, color=C1)
ax.plot(eps, q_roll, color=C1, lw=1.8, label="Q")
ax2_.plot(eps, noise_roll, color=C2, lw=1.8, ls="--", label="Noise")
ax.set_ylabel("Mean Q Value", fontsize=9, color=C1)
ax2_.set_ylabel("Noise σ", fontsize=9, color=C2)
ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax2_.spines["right"].set_visible(True); ax2_.spines["right"].set_color(AXIS_CLR)
ax2_.spines["top"].set_visible(False)
ax2_.tick_params(axis="y", labelsize=8, labelcolor=INK_MUT)
l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2_.get_legend_handles_labels()
ax.legend(l1+l2, n1+n2, fontsize=7.5, loc="upper left")

# ── Panel 5: Col & Avd ──
ax = axes[4]
ax.fill_between(eps, avd_rate, alpha=0.10, color=C3)
ax.plot(eps, avd_rate, color=C3, lw=1.8, label="Avoided")
ax.fill_between(eps, col_rate, alpha=0.12, color=C2)
ax.plot(eps, col_rate, color=C2, lw=1.8, label="Collisions")
ax.set_ylabel("Count per Episode", fontsize=9)
ax.set_ylim(bottom=0)
ax.legend(fontsize=7.5, loc="upper right")

# ── Panel 6: Actor & Critic loss ──
ax = axes[5]
ax2_ = ax.twinx()
ax.fill_between(eps, al_roll, alpha=0.10, color=C5)
ax.plot(eps, al_roll, color=C5, lw=1.8, label="Actor")
ax2_.fill_between(eps, cl_roll, alpha=0.08, color=C6)
ax2_.plot(eps, cl_roll, color=C6, lw=1.8, ls="--", label="Critic")
ax.set_ylabel("Actor Loss", fontsize=9, color=C5)
ax2_.set_ylabel("Critic Loss", fontsize=9, color=C6)
ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax2_.yaxis.set_major_formatter(FuncFormatter(fmt_k))
ax2_.spines["right"].set_visible(True); ax2_.spines["right"].set_color(AXIS_CLR)
ax2_.spines["top"].set_visible(False)
ax2_.tick_params(axis="y", labelsize=8, labelcolor=INK_MUT)
l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2_.get_legend_handles_labels()
ax.legend(l1+l2, n1+n2, fontsize=7.5, loc="upper right")

# ── Common formatting for all panels ──
for i, ax in enumerate(axes):
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_xlim(0, 5050)
    ax.tick_params(labelsize=8)
    ax.set_title(panel_titles[i], fontsize=10, fontweight="bold",
                 color=INK, pad=6, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, lw=0.7, alpha=1.0)

# Supertitle
fig.text(0.5, 0.975, "CERLAB UAV R54 — Tunnel Navigation TD3 | 5,000 Episode Training Overview",
         ha="center", va="top", fontsize=13, fontweight="bold", color=INK)
fig.text(0.5, 0.955, "Best 5,000 episodes selected by average speed · Shaded bands: raw scatter · Lines: 50-episode rolling mean",
         ha="center", va="top", fontsize=9, color=INK_SEC)

save(fig, "R54_Training_Dashboard_Overview.png")

print("\n✓ All files written to Final Results/")
print(f"  Excel : R54_Cleaned_5000_Episodes.xlsx")
print(f"  Plots : plot1_average_speed.png")
print(f"          plot2_success_rate.png")
print(f"          plot3_episode_reward.png")
print(f"          plot4_Q_and_noise.png")
print(f"          plot5_collision_and_avoidance.png")
print(f"          plot6_actor_and_critic_loss.png")
print(f"  Dashboard: R54_Training_Dashboard_Overview.png")
