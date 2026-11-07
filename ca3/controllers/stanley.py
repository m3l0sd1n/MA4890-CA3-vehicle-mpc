"""
stanley.py -- classical baseline
Interface (same as the MPC):  u, info = ctrl.solve(x, kappa_ahead)

Stanley law on the error state:  delta = e_psi + atan(k * e_y / vx)
  e_y   = x[0]   (cross-track error)
  e_psi = x[2]   (heading error)

STUB: a first version is written so comparisons run. TODO: tune k,
decide the cl: clip to delta_max to match the MPC's constraint for a fair
comparison, and confirm the sign convention against the plant.
"""
import time
import numpy as np


class StanleyController:
    def __init__(self, P, vx, k=1.0):
        self.P = P
        self.vx = vx
        self.k = k
        self.delta_max = P.delta_max

    def solve(self, x, kappa_ahead):
        t0 = time.perf_counter()
        e_y, e_psi = float(x[0]), float(x[2])
        # sign: in this error model positive e_y needs NEGATIVE steer, so the
        # classical Stanley law is negated to match the plant convention.
        u = -(e_psi + np.arctan(self.k * e_y / max(self.vx, 1e-3)))
        u = float(np.clip(u, -self.delta_max, self.delta_max))
        return u, {"solve_time": time.perf_counter() - t0}
