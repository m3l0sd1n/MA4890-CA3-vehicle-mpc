"""
run_sweep.py  [Hedi]  -- Q3: effect of speed mismatch on MPC (double lane change)
For each true speed vx, run:
  - MPC with its model at the true speed     (matched)
  - MPC with its model frozen at vx_op       (mismatched)
  - Stanley and PID as baselines
All runs start on the path (e_y0 = 0) and cover the same distance of track.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ca3.params import Params
from ca3.vehicle import LateralErrorPlant
from ca3.reference import Reference
from ca3.controllers.mpc import MPCController
from ca3.controllers.stanley import StanleyController
from ca3.controllers.pid import PIDController
from ca3.metrics import metrics
from scripts.run_sim import run_sim

SPEEDS = [8.0, 10.0, 12.0, 14.0]   # true vehicle speeds [m/s]
DIST = 80.0                        # track distance covered by every run [m]
N = 20                             # MPC horizon


def main():
    P = Params()
    ref = Reference(kind="double_lane_change")
    vx_op = P.vx_op                # speed the mismatched MPC is frozen at

    # one entry per controller variant: name -> function that builds it for a given vx
    variants = {
        "MPC (model at true vx)":  lambda vx: MPCController(P, vx, N=N),
        f"MPC (model at {vx_op:g} m/s)": lambda vx: MPCController(P, vx, N=N, vx_lin=vx_op),
        "Stanley":                 lambda vx: StanleyController(P, vx),
        "PID":                     lambda vx: PIDController(P, vx),
    }

    results = {name: [] for name in variants}       # name -> list of max|e_y| [m], one per speed
    for vx in SPEEDS:
        T = DIST / vx                                # same distance at every speed
        for name, build in variants.items():
            plant = LateralErrorPlant(P, vx)         # the real car always runs at the true vx
            log = run_sim(plant, build(vx), ref, dt=P.dt, vx=vx, T=T, x0=[0.0, 0.0, 0.0, 0.0])
            results[name].append(metrics(log, P.delta_max)["max_abs_ey"])

    # print the table in millimetres
    print(f"{'vx [m/s]':>10}" + "".join(f"{name:>28}" for name in variants))
    for i, vx in enumerate(SPEEDS):
        print(f"{vx:>10g}" + "".join(f"{results[name][i]*1000:>25.1f} mm" for name in variants))

    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    plot(results, vx_op, os.path.join(out_dir, "q3_speed_mismatch.png"))
    return results

def plot(results, vx_op, out_path):
    """max |e_y| vs true speed, one line per controller (log y-axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    # fixed categorical order: blue, orange, aqua, yellow (colour-blind safe)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    ink, ink_muted, grid = "#1f1f1e", "#6b6a64", "#e4e3dc"

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (name, vals), c in zip(results.items(), colors):
        y_mm = [v * 1000 for v in vals]
        ax.plot(SPEEDS, y_mm, color=c, lw=2, marker="o", ms=7,
                markeredgecolor="white", markeredgewidth=1.5, label=name)
        # direct label at the left end, so a line is never identified by colour alone
        ax.annotate(name, (SPEEDS[0], y_mm[0]), xytext=(-10, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, color=ink)

    # the speed the mismatched MPC model is frozen at
    ax.axvline(vx_op, color=ink_muted, lw=1, ls="--", zorder=0)
    ax.annotate(f"MPC model linearised at {vx_op:g} m/s", (vx_op, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, -4),
                textcoords="offset points", va="top", fontsize=8, color=ink_muted)

    # log y-axis with plain-number ticks
    ax.set_yscale("log")
    ticks = [3, 10, 30, 100, 300]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(2.5, 250)
    ax.set_xlim(SPEEDS[0] - 2.6, SPEEDS[-1] + 0.4)
    ax.set_xticks(SPEEDS)

    ax.set_xlabel("True vehicle speed $v_x$ [m/s]", color=ink)
    ax.set_ylabel("Max lateral error $|e_y|$ [mm] (log scale)", color=ink)
    ax.set_title("Double lane change: effect of speed mismatch on tracking error", color=ink, fontsize=11, loc="left")

    # light grid and axes so the data stands out
    ax.grid(True, which="major", color=grid, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink_muted)
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=ink)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")

if __name__ == "__main__":
    main()