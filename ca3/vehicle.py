"""
vehicle.py  [Hedi]  -- the PLANT
Linear lateral error-dynamics model (Rajamani, Vehicle Dynamics and Control,
Section 2.5). State x = [e_y, e_y_dot, e_psi, e_psi_dot], input u = steering
angle delta [rad]. The road enters through curvature kappa (the desired yaw
rate is psi_des_dot = vx * kappa).

Cf, Cr here are treated as WHOLE-AXLE cornering stiffness (N/rad), so there is
no factor of 2. With the current placeholder params this reproduces the
verified open-loop eigenvalues (~ 0, 0, -35, -34 at vx = 10 m/s).

NOTE [Hedi]: this is the stub version. Drop in your verified error_model.py if
its sign/factor conventions differ -- the interface below must stay the same.

MATLAB note: cont2discrete(...,'zoh') is the same idea as c2d(ss(A,B,C,D),dt,'zoh').
"""
import numpy as np
from scipy.signal import cont2discrete


class LateralErrorPlant:
    def __init__(self, P, vx):
        """
        P  : Params object
        vx : the ACTUAL longitudinal speed used to build the plant [m/s].
             (Q3 deviates this from P.vx_op to create model mismatch.)
        """
        self.P = P
        self.vx = float(vx)

        m, Iz, lf, lr = P.m, P.Iz, P.lf, P.lr
        Cf, Cr = P.Cf, P.Cr
        vx = self.vx

        # --- continuous-time matrices (Rajamani 2.5, axle stiffness) ---
        A = np.array([
            [0.0, 1.0,                       0.0,                 0.0],
            [0.0, -(Cf + Cr) / (m * vx),     (Cf + Cr) / m,       (-Cf * lf + Cr * lr) / (m * vx)],
            [0.0, 0.0,                       0.0,                 1.0],
            [0.0, -(Cf * lf - Cr * lr) / (Iz * vx), (Cf * lf - Cr * lr) / Iz,
                  -(Cf * lf**2 + Cr * lr**2) / (Iz * vx)],
        ])
        # steering input column
        B = np.array([[0.0], [Cf / m], [0.0], [Cf * lf / Iz]])
        # road-curvature disturbance column (multiplies psi_des_dot = vx*kappa)
        E = np.array([
            [0.0],
            [-(Cf * lf - Cr * lr) / (m * vx) - vx],
            [0.0],
            [-(Cf * lf**2 + Cr * lr**2) / (Iz * vx)],
        ])

        # discretize [B | E] together with ZOH; inputs are [delta, psi_des_dot]
        Baug = np.hstack([B, E])
        C = np.eye(4)
        D = np.zeros((4, 2))
        Ad, Bd_aug, *_ = cont2discrete((A, Baug, C, D), P.dt, method="zoh")
        self.Ad = Ad
        self.Bd = Bd_aug[:, 0:1]   # delta column
        self.Ed = Bd_aug[:, 1:2]   # psi_des_dot column

        self.x = np.zeros(4)

    def reset(self, x0=None):
        self.x = np.zeros(4) if x0 is None else np.asarray(x0, dtype=float)
        return self.x

    def step(self, u, kappa):
        """Advance one step. u = delta [rad], kappa = road curvature [1/m]."""
        psi_des_dot = self.vx * kappa
        self.x = (self.Ad @ self.x
                  + self.Bd.flatten() * float(u)
                  + self.Ed.flatten() * psi_des_dot)
        return self.x

    def eig(self):
        """Continuous-time check is in the report; here just the discrete A."""
        return np.linalg.eigvals(self.Ad)
