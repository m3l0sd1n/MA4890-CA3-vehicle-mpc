"""
metrics.py  
scores = metrics(log)  ->  dict of scalar performance numbers.
log is the dict returned by run_sim (numpy arrays: t, x, u, kappa, solve_time).

A working basic version is provided. TODO: add whatever Q1-Q3 need
(e.g. settling, ay utilisation), and the plotting lives in make_plots.py.
"""

import numpy as np


def metrics(log, delta_max=None):
    x = log["x"]            # shape (T, 4): [e_y, e_y_dot, e_psi, e_psi_dot]
    u = log["u"]            # shape (T,)
    st = log["solve_time"]  # shape (T,)

    e_y = x[:, 0]
    scores = {
        "rms_ey":          float(np.sqrt(np.mean(e_y**2))),
        "max_abs_ey":      float(np.max(np.abs(e_y))),
        "rms_delta":       float(np.sqrt(np.mean(u**2))),
        "mean_solve_time": float(np.mean(st)),
        "max_solve_time":  float(np.max(st)),
    }
    if delta_max is not None:
        scores["steer_violations"] = int(np.sum(np.abs(u) > delta_max + 1e-9))
    return scores
