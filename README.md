# MA4890 CA3 — Vehicle Path-Tracking MPC 

Clone, then from the repo root:

```bash
pip install numpy scipy pyyaml matplotlib osqp

# results for the report (figures go to results/)
python scripts/run_q1.py          # Q1: MPC vs Stanley vs PID, skidpad + DLC
python scripts/run_constraint.py  # MPC v1 (clip) vs v2 (constraint in QP)
python scripts/run_sweep.py       # Q3: speed mismatch on DLC

# single run for debugging
python scripts/run_sim.py --ctrl stanley --track dlc --ey0 0
python scripts/stanley_plots.py   # Stanley gain tuning (k = 1, 3, 10, 13)
python scripts/pid_analysis.py    # PID gain search and Ki sweep
```

A figure is saved to `results/`. If it runs and you get a plot, your setup is good
and you can start on your file.

## Interface contract (frozen — build your part against these, not the internals)

```python
x       = plant.step(u, kappa)        # x = [e_y, e_y_dot, e_psi, e_psi_dot], u = delta [rad]
u, info = ctrl.solve(x, kappa_ahead)  # every controller returns (u, {"solve_time": seconds})
log     = run_sim(plant, ctrl, ref, dt, vx)   # dict of numpy arrays: t, x, u, kappa, solve_time
scores  = metrics(log, delta_max)             # dict of scalars
```

As long as your file matches its signature, the whole loop keeps working. You can
develop against the stubs already in the repo without waiting for anyone else.

## Notes
- `params/vehicle_params.yml` is auto-synced from the FYP repo. **Do not edit it here** —
  changes get overwritten on the next sync. Pull/sync before you start each session.
- `results/` is not committed (`.gitignore`).
