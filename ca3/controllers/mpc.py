"""
mpc.py  [Hedi]  -- the constrained linear MPC
Interface:  u, info = ctrl.solve(x, kappa_ahead)
            info["solve_time"]  single-step solve time [s]

STUB: this placeholder is a simple stabilizing full-state feedback, clipped to
delta_max, just so the closed loop runs and produces a sensible plot today.
TODO [Hedi]: replace with the condensed-QP MPC from the FYP legacy code
  (prediction matrices M, C and H, E; U* = -H^{-1} E x; take U*[0]; add the
  steering-angle constraint via OSQP or clipping). Keep this exact signature.
"""
import time
import numpy as np


class MPCController:
    def __init__(self, P, vx, N=20):
        self.P = P
        self.vx = vx
        self.N = N
        self.delta_max = P.delta_max
        # STUB gains (full-state feedback). Real version solves a QP instead.
        self._K = np.array([0.8, 0.15, 1.2, 0.1])

    def solve(self, x, kappa_ahead):
        t0 = time.perf_counter()

        # --- STUB control law (replace with condensed-QP MPC) ---------------
        u = -float(self._K @ np.asarray(x))
        # feed-forward for steady curvature so the car does not sit at an offset
        u += self.vx**2 * kappa_ahead[0] / 1.0 * 0.0   # placeholder, real FF in MPC
        # steering-angle constraint
        u = float(np.clip(u, -self.delta_max, self.delta_max))
        # -------------------------------------------------------------------

        solve_time = time.perf_counter() - t0
        return u, {"solve_time": solve_time}
