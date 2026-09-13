#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/home/makhosazana/Project/CERLAB-UAV-Autonomy/Final Results/testing")
OUT  = BASE / "plots" / "Actual"
OUT.mkdir(parents=True, exist_ok=True)

A = pd.read_csv(BASE / "test_r54_attempt2_20260823_195520.csv")
B = pd.read_csv(BASE / "test_px4_attempt1_20260824_083738.csv")
C = pd.read_csv(BASE / "test_px4_training_attempt1_20260824_114608.csv")
for df in (A, B, C):
    df["tunnel_pos"] = df["x_m"] - df["x_m"].iloc[0]

SA = dict(duration=99.7, avg_spd=1.739, peak_spd=3.923, min_dep=0.279, col=2, nm=38)
SB = dict(duration=76.7, avg_spd=2.449, peak_spd=4.849, min_dep=0.851, col=0, nm=0)
SC = dict(duration=80.6, avg_spd=2.195, peak_spd=6.531, min_dep=0.856, col=0, nm=0)
SAFE_A = 1.5; SAFE_PX4 = 2.0; COL_THR = 0.35; NM_THR = 0.80
COL_A = "#1565C0"; COL_B = "#2E7D32"; COL_C = "#BF360C"


def sty():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f5f5f5",
        "grid.color": "white", "grid.linewidth": 1.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 14, "axes.labelweight": "bold",
        "axes.titlesize": 15, "axes.titleweight": "bold",
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11, "legend.framealpha": 0.93,
        "legend.edgecolor": "#BBBBBB", "lines.linewidth": 2.5,
        "font.family": "DejaVu Sans", "figure.dpi": 200,
    })


def mk(main, sub):
    sty()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    # Panel label in bold, context subtitle smaller below
    fig.text(0.5, 0.98, main, ha="center", va="top",
             fontsize=15, fontweight="bold")
    fig.text(0.5, 0.92, sub, ha="center", va="top",
             fontsize=10.5, color="#444444")
    fig.subplots_adjust(top=0.82)
    return fig, ax


def hl(ax, y, c, ls, lw, lbl):
    ax.axhline(y, color=c, linestyle=ls, linewidth=lw, label=lbl, zorder=2)


def sv(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def sb(ax, txt, color, loc="upper right"):
    side = loc.split()[1]
    x = 0.98 if side == "right" else 0.02
    ha = "right" if side == "right" else "left"
    ax.text(x, 0.97, txt, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha=ha,
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec=color, alpha=0.92, linewidth=1.3))


def tm(ax, df, color, yf=0.06):
    ret = df[df["phase"] == "RETURN"]
    if ret.empty:
        return
    t0 = ret["t_s"].iloc[0]
    ax.axvline(t0, color=color, ls=":", lw=1.4, alpha=0.60, zorder=2)
    yl = ax.get_ylim()
    ax.text(t0 + 0.6, yl[0] + (yl[1] - yl[0]) * yf, "↩ Return",
            fontsize=10, color=color, alpha=0.85, fontstyle="italic")


TA  = "Domain Randomisation — R54 Policy"
SA_ = "Native CERLAB Drone  |  Testing World — 20 Dynamic Obstacles  |  ✓ SUCCESS (99.7 s)"
TB  = "Software-in-Loop (SIL) Validation — R54 Policy"
SB_ = ("PX4 MAVROS OFFBOARD  |  "
       "Test B: Testing World 20 obs ✓ (76.7 s)  |  "
       "Test C: Training World 25 obs ✓ (80.6 s)")

PB = mpatches.Patch(color=COL_B,
                    label=f"Test B — Testing World 20 obs  "
                          f"({SB['avg_spd']:.2f} m/s avg)")
PC = mpatches.Patch(color=COL_C,
                    label=f"Test C — Training World 25 obs  "
                          f"({SC['avg_spd']:.2f} m/s avg)")

# ── A-a ───────────────────────────────────────────────────────────────────────
fig, ax = mk("(a)  Mission Progress — Round Trip", f"{TA}\n{SA_}")
t, pos = A["t_s"].values, A["tunnel_pos"].values
ax.plot(t, pos, color=COL_A, lw=2.8, label="Tunnel position (m)", zorder=3)
ax.fill_between(t, 0, pos, color=COL_A, alpha=0.09)
hl(ax, 95, "#555555", "--", 1.6, "Turn-point (95 m)")
ax.set_ylim(0, 108); ax.set_xlim(left=0)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Tunnel position (m)")
pi = int(np.argmax(pos))
ax.annotate("Outbound\ncomplete", xy=(t[pi], pos[pi]),
            xytext=(t[pi] - 16, pos[pi] - 20),
            arrowprops=dict(arrowstyle="->", color=COL_A, lw=1.5),
            fontsize=10, color=COL_A, fontweight="bold")
tm(ax, A, COL_A, 0.05)
ax.legend(loc="lower right", frameon=True)
sv(fig, "A_a_mission_progress.png")

# ── A-b ───────────────────────────────────────────────────────────────────────
fig, ax = mk("(b)  Flight Altitude", f"{TA}\n{SA_}")
t, alt = A["t_s"].values, A["z_m"].values
ax.plot(t, alt, color=COL_A, lw=2.5, label="Altitude", zorder=3)
ax.fill_between(t, alt, SAFE_A, where=alt < SAFE_A,
                color="#E53935", alpha=0.22, label="Below safe altitude")
hl(ax, SAFE_A, "#1976D2", "--", 1.9, f"Safe altitude ({SAFE_A} m)")
hl(ax, 2.8, "#FF8F00", ":", 1.5, "Max altitude (2.8 m)")
ax.set_xlim(left=0); ax.set_ylim(1.15, 3.0)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)")
ax.legend(loc="upper right", frameon=True)
sv(fig, "A_b_altitude.png")

# ── A-c ───────────────────────────────────────────────────────────────────────
fig, ax = mk("(c)  Flight Speed", f"{TA}\n{SA_}")
t, spd = A["t_s"].values, A["speed_ms"].values
ax.plot(t, spd, color=COL_A, lw=2.0, alpha=0.88, label="Speed (m/s)", zorder=3)
ax.fill_between(t, 0, spd, color=COL_A, alpha=0.10)
hl(ax, SA["avg_spd"], "#1976D2", "--", 1.9,
   f"Mean speed  {SA['avg_spd']:.3f} m/s")
hl(ax, SA["peak_spd"], "#E53935", ":", 1.6,
   f"Peak speed  {SA['peak_spd']:.3f} m/s")
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
ax.legend(loc="upper right", frameon=True)
sv(fig, "A_c_speed.png")

# ── A-d ───────────────────────────────────────────────────────────────────────
fig, ax = mk("(d)  Minimum Obstacle Distance", f"{TA}\n{SA_}")
t, dep = A["t_s"].values, A["min_depth_m"].values
ax.plot(t, dep, color=COL_A, lw=2.0, label="Min obstacle distance", zorder=3)
ax.fill_between(t, 0, dep, where=dep < COL_THR,
                color="#E53935", alpha=0.30,
                label=f"Collision zone (< {COL_THR} m)")
ax.fill_between(t, COL_THR, np.minimum(dep, NM_THR),
                where=(dep >= COL_THR) & (dep < NM_THR),
                color="#FF8F00", alpha=0.15,
                label=f"Near-miss zone (< {NM_THR} m)")
hl(ax, COL_THR, "#E53935", "--", 1.9, f"Collision threshold ({COL_THR} m)")
hl(ax, NM_THR, "#FF8F00", ":", 1.6, f"Near-miss threshold ({NM_THR} m)")
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
sb(ax, f"Min depth:   {SA['min_dep']:.3f} m\n"
       f"Collisions:  {SA['col']}\n"
       f"Near-misses: {SA['nm']}", COL_A, "upper right")
ax.legend(loc="upper left", frameon=True)
sv(fig, "A_d_obstacle_distance.png")

# ── BC-a ──────────────────────────────────────────────────────────────────────
fig, ax = mk("(a)  Mission Progress — Round Trip", f"{TB}\n{SB_}")
for df, col in [(B, COL_B), (C, COL_C)]:
    t, pos = df["t_s"].values, df["tunnel_pos"].values
    ax.plot(t, pos, color=col, lw=2.8, zorder=3)
    ax.fill_between(t, 0, pos, color=col, alpha=0.07)
hl(ax, 95, "#555555", "--", 1.6, "Turn-point (95 m)")
ax.set_ylim(0, 108); ax.set_xlim(left=0)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Tunnel position (m)")
ref = plt.Line2D([0], [0], color="#555555", ls="--", lw=1.6,
                 label="Turn-point (95 m)")
ax.legend(handles=[PB, PC, ref], loc="lower right", frameon=True)
sv(fig, "BC_a_mission_progress.png")

# ── BC-b ──────────────────────────────────────────────────────────────────────
fig, ax = mk("(b)  Flight Altitude", f"{TB}\n{SB_}")
for df, col in [(B, COL_B), (C, COL_C)]:
    t, alt = df["t_s"].values, df["z_m"].values
    ax.plot(t, alt, color=col, lw=2.5, zorder=3)
    ax.fill_between(t, alt, SAFE_PX4, where=alt < SAFE_PX4, color=col, alpha=0.12)
hl(ax, SAFE_PX4, "#1976D2", "--", 1.9, f"Safe altitude ({SAFE_PX4} m)")
hl(ax, 3.2, "#FF8F00", ":", 1.5, "Max altitude (3.2 m)")
ax.set_xlim(left=0)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)")
refs = [plt.Line2D([0],[0], color="#1976D2", ls="--", lw=1.9,
                   label=f"Safe altitude ({SAFE_PX4} m)"),
        plt.Line2D([0],[0], color="#FF8F00", ls=":", lw=1.5,
                   label="Max altitude (3.2 m)")]
ax.legend(handles=[PB, PC] + refs, loc="upper right", frameon=True)
sv(fig, "BC_b_altitude.png")

# ── BC-c ──────────────────────────────────────────────────────────────────────
fig, ax = mk("(c)  Flight Speed", f"{TB}\n{SB_}")
ml = []
for df, col, s, lbl in [(B, COL_B, SB, "B"), (C, COL_C, SC, "C")]:
    t, spd = df["t_s"].values, df["speed_ms"].values
    ax.plot(t, spd, color=col, lw=2.0, alpha=0.88, zorder=3)
    ax.fill_between(t, 0, spd, color=col, alpha=0.07)
    ax.axhline(s["avg_spd"], color=col, ls="--", lw=1.6, alpha=0.75, zorder=2)
    ml.append(plt.Line2D([0],[0], color=col, ls="--", lw=1.6,
                         label=f"Test {lbl} mean ({s['avg_spd']:.2f} m/s)"))
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
ax.legend(handles=[PB, PC] + ml, loc="upper right", frameon=True)
sv(fig, "BC_c_speed.png")

# ── BC-d ──────────────────────────────────────────────────────────────────────
fig, ax = mk("(d)  Minimum Obstacle Distance", f"{TB}\n{SB_}")
for df, col in [(B, COL_B), (C, COL_C)]:
    t, dep = df["t_s"].values, df["min_depth_m"].values
    ax.plot(t, dep, color=col, lw=2.0, alpha=0.88, zorder=3)
ax.axhspan(0, COL_THR, color="#E53935", alpha=0.07, zorder=0)
hl(ax, COL_THR, "#E53935", "--", 1.9, f"Collision threshold ({COL_THR} m)")
hl(ax, NM_THR, "#FF8F00", ":", 1.5, f"Near-miss threshold ({NM_THR} m)")
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
sb(ax, f"Test B — min: {SB['min_dep']:.3f} m  col: {SB['col']}  NM: {SB['nm']}\n"
       f"Test C — min: {SC['min_dep']:.3f} m  col: {SC['col']}  NM: {SC['nm']}",
   "#555555", "upper right")
refs2 = [plt.Line2D([0],[0], color="#E53935", ls="--", lw=1.9,
                    label=f"Collision threshold ({COL_THR} m)"),
         plt.Line2D([0],[0], color="#FF8F00", ls=":", lw=1.5,
                    label=f"Near-miss threshold ({NM_THR} m)")]
ax.legend(handles=[PB, PC] + refs2, loc="upper left", frameon=True)
sv(fig, "BC_d_obstacle_distance.png")

print("DONE — 8 figures saved.")
