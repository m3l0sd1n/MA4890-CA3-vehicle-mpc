"""
run_sim.py  [Hedi]  -- the driver everyone runs
Wires plant + controller + reference into one closed-loop simulation and saves
a plot. This is the STUB handover: clone, `python scripts/run_sim.py`, get a
figure. Each owner then fills in their file (see the [Name] tags) against the
interface below -- nothing here needs to change for that.

Interface contract (frozen -- build your part against these):
    x       = plant.step(u, kappa)        # x = [e_y, e_y_dot, e_psi, e_psi_dot], u = delta [rad]
    u, info = ctrl.solve(x, kappa_ahead)  # info["solve_time"] [s]
    log     = run_sim(plant, ctrl, ref)   # dict of numpy arrays: t, x, u, kappa, solve_time
    scores  = metrics(log, delta_max)     # dict of scalars

Run:
    python scripts/run_sim.py                 # MPC (stub), skidpad
    python scripts/run_sim.py --ctrl stanley  # Stanley baseline
    python scripts/run_sim.py --vx 16 --N 40  # Q2/Q3 knobs
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on a headless machine; saves to file
import matplotlib.pyplot as plt

# make `import ca3...` work no matter where you run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ca3.params import Params
from ca3.vehicle import LateralErrorPlant
from ca3.reference import Reference
from ca3.controllers.mpc import MPCController
from ca3.controllers.stanley import StanleyController
from ca3.controllers.pid import PIDController
from ca3.metrics import metrics


def run_sim(plant, ctrl, ref, dt, vx, T=20.0, x0=None):
    """Closed-loop simulation. Returns a log dict of numpy arrays."""
    n_steps = int(T / dt)
    ds = vx * dt            # arc length advanced per step (constant speed)
    N = getattr(ctrl, "N", 20)

    plant.reset(x0 if x0 is not None else [0.5, 0.0, 0.0, 0.0])  # start 0.5 m off
    s = 0.0

    t_log, x_log, u_log, k_log, st_log = [], [], [], [], []
    for k in range(n_steps):
        kappa_now = ref.curvature(s)
        kappa_ahead = ref.curvature_preview(s, N, ds)

        u, info = ctrl.solve(plant.x.copy(), kappa_ahead)

        t_log.append(k * dt)
        x_log.append(plant.x.copy())
        u_log.append(u)
        k_log.append(kappa_now)
        st_log.append(info["solve_time"])

        plant.step(u, kappa_now)
        s += ds

    return {
        "t": np.array(t_log),
        "x": np.array(x_log),
        "u": np.array(u_log),
        "kappa": np.array(k_log),
        "solve_time": np.array(st_log),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", choices=["mpc", "stanley", "pid"], default="mpc")
    ap.add_argument("--vx", type=float, default=None, help="actual speed [m/s]")
    ap.add_argument("--N", type=int, default=20, help="MPC horizon")
    ap.add_argument("--track", default="skidpad")
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--vx_lin", type=float, default=None,
                    help="speed the MPC model is linearised at [m/s]; default = vx (no mismatch)")
    args = ap.parse_args()

    P = Params()
    vxl = args.vx_lin if args.vx_lin is not None else vx
    print(f"controller={args.ctrl}  track={args.track}  vx={vx} m/s  vx_lin={vxl} m/s  N={args.N}")

    # plant is built at the ACTUAL speed vx; the MPC is linearised at vx too in
    # the stub. Q3 (model mismatch) = build the controller's model at P.vx_op
    # but run the plant at a different vx -- wire that in when the real MPC lands.
    plant = LateralErrorPlant(P, vx)
    ref = Reference(kind=args.track)

    if args.ctrl == "mpc":
        ctrl = MPCController(P, vx, N=args.N, vx_lin=args.vx_lin)
    elif args.ctrl == "stanley":
        ctrl = StanleyController(P, vx)
    else:
        ctrl = PIDController(P, vx)

    log = run_sim(plant, ctrl, ref, dt=P.dt, vx=vx, T=args.T)
    scores = metrics(log, delta_max=P.delta_max)

    print(f"controller={args.ctrl}  vx={vx} m/s  N={args.N}")
    for key, val in scores.items():
        print(f"  {key:18s} = {val:.5g}")

    # --- plot: e_y(t) and delta(t) ---
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(log["t"], log["x"][:, 0])
    ax[0].set_ylabel("cross-track error e_y [m]")
    ax[0].grid(True)
    ax[0].set_title(f"{args.ctrl} | {args.track} | vx={vx} | vx_lin={vxl} | N={args.N}")
    ax[1].plot(log["t"], np.degrees(log["u"]))
    ax[1].axhline(np.degrees(P.delta_max), ls="--", color="r", lw=0.8)
    ax[1].axhline(-np.degrees(P.delta_max), ls="--", color="r", lw=0.8)
    ax[1].set_ylabel("steering delta [deg]")
    ax[1].set_xlabel("time [s]")
    ax[1].grid(True)
    fig.tight_layout()
    out = os.path.join(out_dir, f"run_{args.ctrl}_{args.track}_vx{vx:g}_lin{vxl:g}_N{args.N}.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
