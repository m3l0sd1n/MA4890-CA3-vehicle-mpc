"""
pid.py  [Member ?]  -- naive baseline
Interface (same as all controllers):  u, info = ctrl.solve(x, kappa_ahead)

A plain PID on the cross-track error e_y only. This is the "what a team often
starts with" baseline: single-error feedback, no vehicle model, no preview of
the road ahead, no constraint handling. Expect it to struggle on sustained
curvature (steady-state error / lag) -- that is the point of including it.

  e_y     = x[0]   (cross-track error)
  e_y_dot = x[1]
  delta = -(Kp*e_y + Ki*∫e_y dt + Kd*e_y_dot)   (negative to match plant sign)

STUB: starter gains are set. TODO [owner]: tune Kp/Ki/Kd, decide whether to
clip to delta_max for a fair comparison, and discuss WHY it lags on curves.
"""
import time
import numpy as np


class PIDController:
    def __init__(self, P, vx, Kp=0.6, Ki=0.0, Kd=0.15):
        self.P = P
        self.vx = vx
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.delta_max = P.delta_max
        self.dt = P.dt
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def solve(self, x, kappa_ahead):
        t0 = time.perf_counter()
        e_y, e_y_dot = float(x[0]), float(x[1])
        self._integral += e_y * self.dt
        u = -(self.Kp * e_y + self.Ki * self._integral + self.Kd * e_y_dot)
        u = float(np.clip(u, -self.delta_max, self.delta_max))
        return u, {"solve_time": time.perf_counter() - t0}
