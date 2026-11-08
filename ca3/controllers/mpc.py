"""
mpc.py  [Hedi]  -- constrained linear MPC (condensed QP, unconstrained v1)
Interface:  u, info = ctrl.solve(x, kappa_ahead)

v1: unconstrained condensed QP with curvature feed-forward.
    U* = -H^{-1} (E x0 + Ed_dist D),  take first step, clip to delta_max.
TODO v2: enforce |delta| <= delta_max inside the QP via OSQP (drop the clip).
"""
import time
import numpy as np
from ca3.vehicle import LateralErrorPlant   # reuse the SAME Ad/Bd/Ed


class MPCController:
    def __init__(self, P, vx, N=20, vx_lin=None):
        self.P = P
        self.vx = vx
        self.N = N
        self.delta_max = P.delta_max

        # --- (1) prediction model: borrow the plant's discrete matrices ---
        # vx_lin = speed the MPC linearises at. Default = real vx (Q1/Q2).
        # Q3: pass vx_lin = P.vx_op to freeze the model while the plant runs faster.
        vx_lin = vx if vx_lin is None else vx_lin
        model = LateralErrorPlant(P, vx_lin)
        self.Ad = model.Ad
        self.Bd = model.Bd
        Ed_vec  = model.Ed          # plant disturbance column (4x1), local only
        self.vx_lin = vx_lin

        # --- (2) weights ---
        # state [e_y, e_y_dot, e_psi, e_psi_dot]: penalise e_y and e_psi
        self.Q = np.diag([1.0, 0.0, 0.5, 0.0])
        self.R = np.array([[0.1]])
        self.F = self.Q

        n = self.Ad.shape[0]        # 4
        N = self.N

        # --- (3) prediction matrices M, C, Cd (built once, offline) ---
        M  = np.zeros((N * n, n))
        C  = np.zeros((N * n, N))
        Cd = np.zeros((N * n, N))
        Apow = np.eye(n)
        for i in range(N):                       # i = 0..N-1  (predict x_1..x_N)
            Apow = self.Ad @ Apow                # Apow = Ad^(i+1)
            M[i*n:(i+1)*n, :] = Apow
            Aj = np.eye(n)
            for j in range(i, -1, -1):           # fill lower triangle: (i,j)=Ad^(i-j)·col
                C[i*n:(i+1)*n,  j:j+1] = Aj @ self.Bd
                Cd[i*n:(i+1)*n, j:j+1] = Aj @ Ed_vec
                Aj = self.Ad @ Aj
        self.M, self.C, self.Cd = M, C, Cd

        # --- (4) stack weights, form QP coefficients ---
        Qbar = np.zeros((N * n, N * n))
        for i in range(N - 1):
            Qbar[i*n:(i+1)*n, i*n:(i+1)*n] = self.Q
        Qbar[(N-1)*n:N*n, (N-1)*n:N*n] = self.F
        Rbar = np.kron(np.eye(N), self.R)

        self.H       = C.T @ Qbar @ C  + Rbar    # Hessian (N x N)
        self.E       = C.T @ Qbar @ M            # links x0   (N x n)
        self.Ed_dist = C.T @ Qbar @ Cd           # links D    (N x N)

    def solve(self, x, kappa_ahead):
        t0 = time.perf_counter()
        x0 = np.asarray(x, dtype=float).reshape(-1)

        # (1) curvature preview -> disturbance sequence D = vx_lin * kappa
        kap = np.asarray(kappa_ahead, dtype=float).reshape(-1)[:self.N]
        if kap.size < self.N:
            kap = np.concatenate([kap, np.full(self.N - kap.size, kap[-1])])
        D = self.vx_lin * kap

        # (2) unconstrained QP:  H U = -(E x0 + Ed_dist D)
        f = self.E @ x0 + self.Ed_dist @ D
        U = np.linalg.solve(self.H, -f)

        # (3) receding horizon: first move, clipped to steering limit
        delta0 = float(np.clip(U[0], -self.delta_max, self.delta_max))
        return delta0, {"solve_time": time.perf_counter() - t0}
