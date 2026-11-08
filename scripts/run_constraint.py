"""
run_constraint.py  [Hedi]  -- MPC v1 (clip) vs v2 (OSQP, constraint in the QP)
Skidpad, start 0.5 m off, tighter and tighter steering limits.
Prints a table and saves results/constraint_v1_vs_v2.png.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ca3.params import Params
from ca3.vehicle import LateralErrorPlant
from ca3.reference import Reference
from ca3.controllers.mpc import MPCController
from run_sim import run_sim          # same folder, so a plain import works

LIMITS_DEG = [28.6, 14.3, 11.5]     # 0.50, 0.25, 0.20 rad
PLOT_DEG = 14.3                     # the limit shown in the figure
T = 10.0

P = Params()
vx = P.vx_op
ref = Reference(kind="skidpad")
logs = {}

print(f"{'dmax[deg]':>9} | {'v1 rms':>8} {'v1 over':>8} | {'v2 rms':>8} {'v2 over':>8}   [mm]")
for dmax_deg in LIMITS_DEG:
    P.delta_max = np.radians(dmax_deg)          # override the limit for this run
    row = []
    for name, cons in [("v1", False), ("v2", True)]:
        plant = LateralErrorPlant(P, vx)
        ctrl = MPCController(P, vx, constrained=cons)
        log = run_sim(plant, ctrl, ref, dt=P.dt, vx=vx, T=T, x0=[0.5, 0, 0, 0])
        ey = log["x"][:, 0]
        rms = np.sqrt(np.mean(ey**2)) * 1e3
        overshoot = max(0.0, -ey.min()) * 1e3   # starts at +0.5 m, so overshoot = how far it crosses below 0
        row += [rms, overshoot]
        logs[(dmax_deg, name)] = log
    print(f"{dmax_deg:9.1f} | {row[0]:8.1f} {row[1]:8.1f} | {row[2]:8.1f} {row[3]:8.1f}")

# --- figure: e_y(t) and delta(t) at PLOT_DEG ---
C1, C2 = "#eb6834", "#2a78d6"   # v1 orange, v2 blue
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for name, col, lab in [("v1", C1, "v1: unconstrained QP + clip"),
                       ("v2", C2, "v2: constraint inside QP (OSQP)")]:
    log = logs[(PLOT_DEG, name)]
    ax[0].plot(log["t"], log["x"][:, 0], color=col, lw=2, label=lab)
    ax[1].plot(log["t"], np.degrees(log["u"]), color=col, lw=2)
for s in (+1, -1):
    ax[1].axhline(s * PLOT_DEG, ls="--", color="0.4", lw=1)
ax[0].axhline(0, color="0.6", lw=0.8)
ax[0].set_ylabel("cross-track error e_y [m]")
ax[0].legend(frameon=False)
ax[0].set_title(f"Skidpad, vx={vx:g} m/s, start 0.5 m off, steering limit ±{PLOT_DEG}°")
ax[1].set_ylabel("steering δ [deg]")
ax[1].set_xlabel("time [s]")
ax[1].set_xlim(0, 3)                 # all the action is in the first 3 s
for a in ax:
    a.grid(True, alpha=0.3)
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "constraint_v1_vs_v2.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")