"""
mpc.py  [Hedi]  -- linear MPC (condensed QP), v1: unconstrained + curvature feed-forward
Interface:  u, info = ctrl.solve(x, kappa_ahead)
TODO v2: enforce |delta| <= delta_max inside the QP via OSQP (drop the clip).
"""

import time
import numpy as np

from ca3.vehicle import LateralErrorPlant   # Ad/Bd/Ed

import osqp
from scipy import sparse

class MPCController:
    def __init__(self, P, vx, N=20, vx_lin=None, constrained=True):
        self.P = P
        self.vx = vx
        self.N = N
        self.delta_max = P.delta_max

        # vx_lin = speed the MPC model is linearised at (default = real vx; Q3: vx_op)
        vx_lin = vx if vx_lin is None else vx_lin
        model = LateralErrorPlant(P, vx_lin)
        self.Ad = model.Ad
        self.Bd = model.Bd
        Ed_vec  = model.Ed           
        self.vx_lin = vx_lin

        # weights
        # state matrices [e_y, e_y_dot, e_psi, e_psi_dot]：focus on e_y、e_psi
        self.Q = np.diag([1.0, 0.0, 0.5, 0.0])
        self.R = np.array([[0.1]])      
        self.F = self.Q                 # F=Q

        # prediction matrices M, C, Cd 
        n = self.Ad.shape[0]          # state dimension = 4
        N = self.N
        M  = np.zeros((N * n, n))    
        C  = np.zeros((N * n, N))     # δ
        Cd = np.zeros((N * n, N))     # ψ̇_des

        Apow = np.eye(n)              
        for i in range(N):          
            Apow = self.Ad @ Apow    
            M[i*n:(i+1)*n, :] = Apow
            Aj = np.eye(n)
            for j in range(i, -1, -1):         
                C[i*n:(i+1)*n,  j:j+1] = (Aj @ self.Bd)
                Cd[i*n:(i+1)*n, j:j+1] = (Aj @ Ed_vec)
                Aj = self.Ad @ Aj             
        self.M, self.C, self.Cd = M, C, Cd
        
        # diagonal weight matrices Q̄, R̄ 
        Qbar = np.zeros((N * n, N * n))
        for i in range(N - 1):
            Qbar[i*n:(i+1)*n, i*n:(i+1)*n] = self.Q
        Qbar[(N-1)*n:N*n, (N-1)*n:N*n] = self.F    
        Rbar = np.kron(np.eye(N), self.R)           

        # condensed QP  
        self.H  = C.T @ Qbar @ C + Rbar     # Hessian, (N x N)
        self.E  = C.T @ Qbar @ M            # (N x n)
        self.Ed_dist = C.T @ Qbar @ Cd      # (N x N)
        
        # --- (6) constrained QP solver (OSQP), set up once ---
        # OSQP standard form:  min 1/2 U'PU + q'U   s.t.  l <= A U <= u
        # P = H and A = I never change, so OSQP factorises them once;
        # each step only q (= f) is updated.
        self.constrained = constrained
        if constrained:
            self.qp = osqp.OSQP()
            self.qp.setup(
                P=sparse.triu(sparse.csc_matrix(self.H), format="csc"),
                q=np.zeros(N),
                A=sparse.eye(N, format="csc"),
                l=-self.delta_max * np.ones(N),
                u= self.delta_max * np.ones(N),
                verbose=False, eps_abs=1e-6, eps_rel=1e-6,
            )

    def solve(self, x, kappa_ahead):
        t0 = time.perf_counter()
        x0 = np.asarray(x, dtype=float).reshape(-1)   # current error state (4,)

        # (1) curvature preview -> disturbance sequence D = vx_lin * kappa_ahead
        #     use the MPC's own vx_lin (the speed its prediction model assumes)
        kap = np.asarray(kappa_ahead, dtype=float).reshape(-1)[:self.N]
        if kap.size < self.N:                         # pad with the last value if the preview is too short
            kap = np.concatenate([kap, np.full(self.N - kap.size, kap[-1])])
        D = self.vx_lin * kap                         # (N,)

        # (2) unconstrained QP:  H U = -(E x0 + Ed_dist D)
        f = self.E @ x0 + self.Ed_dist @ D            # linear term (N,)
        U = np.linalg.solve(self.H, -f)               # U* = -H^-1 f, full input sequence (N,)

        if self.constrained:
            # v2: constraint is inside the QP -- the whole planned sequence respects delta_max
            self.qp.update(q=f)
            res = self.qp.solve()
            if res.info.status not in ("solved", "solved inaccurate"):
                # safety fallback: never return garbage to the car
                U = np.clip(np.linalg.solve(self.H, -f), -self.delta_max, self.delta_max)
            else:
                U = res.x
            d0 = float(U[0])
        else:
            # v1: unconstrained solve, then clip only the first move
            U = np.linalg.solve(self.H, -f)
            d0 = float(np.clip(U[0], -self.delta_max, self.delta_max))

        return d0, {"solve_time": time.perf_counter() - t0}

        return delta0, {"solve_time": time.perf_counter() - t0}