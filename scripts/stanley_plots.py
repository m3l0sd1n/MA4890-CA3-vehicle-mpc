"""stanley_plots.py -- Stanley gain-tuning figures (k = 1, 3, 10) on the skidpad."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ca3.params import Params
from ca3.vehicle import LateralErrorPlant
from ca3.reference import Reference
from ca3.controllers.stanley import StanleyController
from run_sim import run_sim

P = Params(); vx = P.vx_op
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(out_dir, exist_ok=True)

for k in [1, 3, 10, 13]:
    log = run_sim(LateralErrorPlant(P, vx), StanleyController(P, vx, k=k), Reference("skidpad"),
                  dt=P.dt, vx=vx, T=10, x0=[0.5, 0, 0, 0])
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax[0].plot(log["t"], log["x"][:, 0], color="#2a78d6", lw=2)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_ylabel("cross-track error e_y [m]")
    ax[0].set_title(f"Stanley, k = {k}, skidpad, vx = {vx:g} m/s")
    ax[1].plot(log["t"], np.degrees(log["u"]), color="#eb6834", lw=2)
    for s in (1, -1):
        ax[1].axhline(s * np.degrees(P.delta_max), ls="--", color="r", lw=0.8)
    ax[1].set_ylabel("steering δ [deg]"); ax[1].set_xlabel("time [s]")
    for a in ax:
        a.grid(True, alpha=0.3); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(out_dir, f"stanley_k{k}.png")
    fig.savefig(out, dpi=150)
    print(f"k={k}: final e_y = {log['x'][-1, 0]*1e3:.1f} mm, saved {out}")