"""
pid.py  [Yash -- Part B]  -- classical PID baseline
Interface (same as all controllers):  u, info = ctrl.solve(x, kappa_ahead)

A PID on the cross-track error e_y only. This is the "what a team usually
starts with" baseline: one measured error, no vehicle model, no preview of the
road. kappa_ahead is accepted so the interface matches, but the baseline does
not use it.

  e_y     = x[0]   cross-track error [m]
  e_y_dot = x[1]   its rate [m/s], read from the state (no numerical d/dt)
  delta   = -(Kp*e_y + Ki*integral(e_y dt) + Kd*e_y_dot)        [rad]
  (minus sign: in this error model a positive e_y needs negative steer)

Design decisions (reproduce with `python scripts/pid_analysis.py`)
  1. Gains Kp = 1.05 rad/m, Ki = 2.0 rad/(m s), Kd = 0.19 rad s/m, tuned at
     vx_op = 10 m/s by a constrained search: minimise RMS e_y on the skidpad +
     RMS e_y on the double lane change (the two Q1 runs), subject to
       - delay margin >= 100 ms at the steering input (the plant has no delay,
         a real car does; without this the search just pushes the gains up), and
       - closed-loop damping ratio >= 0.6.
     The cost is flat near its optimum, which lies on the delay-margin limit
     (any further error reduction costs robustness). Of the gain sets within
     1 % of the minimum cost, the best-damped one is used (damping 0.66 vs 0.61
     at the optimum). Gains are fixed, not scheduled with speed, as a plain PID
     would be. Tuned on params SOURCE_COMMIT 6500013; if the yml changes, rerun
     scripts/pid_analysis.py -- it warns if these defaults are no longer the pick.
  2. Output clipped to +/- delta_max: the MPC enforces the same limit inside its
     QP, and Stanley clips too, so all three see the same actuator.
  3. Anti-windup by conditional integration: while the output is saturated,
     the integrator stops accumulating error that would push it further into
     saturation (otherwise it overshoots after the 0.5 m skidpad start).

Why it still lags on curves (structural, not tuning)
  To drive a curve of curvature kappa the car needs a steady steering angle
  delta_ss = (L + K_v*vx^2)*kappa (10.3 deg on the skidpad at 10 m/s). A PID
  can only produce steering from error, so:
    - P/PD: needs a standing error e_ss = delta_ss / Kp (171 mm at Kp = 1.05);
    - PID: the integrator removes that offset on a constant curve, but only
      after the error has built up and been integrated away, and on a lane
      change it is still holding steering for the curve that just ended.
  Raising the gains shrinks the error but runs out of delay margin first.
  What is missing is information (the road curvature, which kappa_ahead
  already carries), not gain. `feedforward=True` adds
  delta_ff = (L + K_v*vx^2)*kappa_ahead[0] to show this; it is an ablation for
  the analysis only and is OFF in the baseline used by run_q1 / run_sweep.
"""
import time
import numpy as np


class PIDController:
    def __init__(self, P, vx, Kp=1.05, Ki=2.0, Kd=0.19, feedforward=False):
        self.P = P
        self.vx = vx
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.delta_max = P.delta_max
        self.dt = P.dt
        self.feedforward = feedforward
        # steady-state steering per unit curvature, delta_ss = (L + K_v vx^2) kappa
        # (Rajamani Ch. 3 steady-state cornering, no heading-error gain; Cf, Cr
        # are axle values). Current params: K_v = 3.4e-4 rad s^2/m (mild
        # understeer), so the speed term adds ~2 % to L at 10 m/s.
        K_v = (P.m / P.L) * (P.lr / P.Cf - P.lf / P.Cr)   # understeer gradient [rad/(m/s^2)]
        self.ff_gain = P.L + K_v * vx**2                   # [rad per 1/m]
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def _law(self, e_y, e_y_dot, integral, delta_ff):
        return delta_ff - (self.Kp * e_y + self.Ki * integral + self.Kd * e_y_dot)

    def solve(self, x, kappa_ahead):
        t0 = time.perf_counter()
        e_y, e_y_dot = float(x[0]), float(x[1])
        delta_ff = self.ff_gain * float(kappa_ahead[0]) if self.feedforward else 0.0

        candidate = self._integral + e_y * self.dt
        u_raw = self._law(e_y, e_y_dot, candidate, delta_ff)
        u = float(np.clip(u_raw, -self.delta_max, self.delta_max))

        # conditional integration: when saturated, do not integrate an error
        # whose integral term (-Ki*e_y) pushes further into the same limit
        winding_up = (u != u_raw) and (np.sign(u_raw) == np.sign(-self.Ki * e_y))
        if winding_up:
            u = float(np.clip(self._law(e_y, e_y_dot, self._integral, delta_ff),
                              -self.delta_max, self.delta_max))
        else:
            self._integral = candidate

        return u, {"solve_time": time.perf_counter() - t0}
