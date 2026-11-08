"""
run_q1.py  [Hedi]  -- Q1: MPC vs Stanley vs PID on both tracks
(a) skidpad, start 0.5 m off  -> convergence + steady circle
(b) double lane change, start on the path -> pure tracking
Prints a table and saves results/q1_comparison.png.
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
from ca3.controllers.stanley import StanleyController
from ca3.controllers.pid import PIDController
from run_sim import run_sim

P = Params()
vx = P.vx_op

CTRLS = {   # name: (builder, colour)
    "MPC":     (lambda: MPCController(P, vx),     "#2a78d6"),
    "Stanley": (lambda: StanleyController(P, vx), "#eb6834"),
    "PID":     (lambda: PIDController(P, vx),     "#1baf7a"),
}
CASES = {   # track: (ey0, T)
    "skidpad": (0.5, 10.0),
    "dlc":     (0.0, 80.0 / vx),   # 80 m covers the whole manoeuvre
}

logs = {}
print(f"{'track':8} {'ctrl':8} | {'rms e_y':>8} {'max|e_y|':>8} {'final e_y':>9} [mm] | {'mean solve':>10} [us]")
for track, (ey0, T) in CASES.items():
    ref = Reference(kind=track)
    for name, (build, _) in CTRLS.items():
        plant = LateralErrorPlant(P, vx)
        log = run_sim(plant, build(), ref, dt=P.dt, vx=vx, T=T, x0=[ey0, 0, 0, 0])
        logs[(track, name)] = log
        ey = log["x"][:, 0]
        final = np.mean(ey[-int(1.0 / P.dt):])         # mean over the last 1 s = steady-state error
        print(f"{track:8} {name:8} | {np.sqrt(np.mean(ey**2))*1e3:8.1f} {np.max(np.abs(ey))*1e3:8.1f}"
              f" {final*1e3:9.1f}      | {np.mean(log['solve_time'])*1e6:10.1f}")

# --- figure: two panels, each with its own y-axis (different scales, NOT a dual axis) ---
fig, ax = plt.subplots(2, 1, figsize=(8, 7))
panels = [("skidpad", 1.0,  "m",  "(a) Skidpad, start 0.5 m off", 4.0),
          ("dlc",     1e3,  "mm", "(b) Double lane change, start on path", CASES["dlc"][1])]
for a, (track, scale, unit, title, xmax) in zip(ax, panels):
    for name, (_, col) in CTRLS.items():
        log = logs[(track, name)]
        a.plot(log["t"], log["x"][:, 0] * scale, color=col, lw=2, label=name)
    a.axhline(0, color="0.6", lw=0.8)
    a.set_title(title, loc="left")
    a.set_ylabel(f"cross-track error e_y [{unit}]")
    a.set_xlabel("time [s]")
    a.set_xlim(0, xmax)
    a.grid(True, alpha=0.3)
    a.spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False)
fig.suptitle(f"Q1: lateral tracking at vx = {vx:g} m/s", x=0.02, ha="left")
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "q1_comparison.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")