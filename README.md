# MA4890 CA3 — Vehicle Path-Tracking MPC 

Clone, then from the repo root:

```bash
pip install numpy scipy pyyaml matplotlib
python scripts/run_sim.py                 # MPC (stub), skidpad, vx=10, N=20
python scripts/run_sim.py --ctrl stanley  # Stanley baseline
python scripts/run_sim.py --vx 16 --N 40  # Q2/Q3 knobs
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
- The stub MPC is **not** horizon-aware, so `--N` has no effect yet; it starts mattering
  once the real condensed-QP MPC lands. That is expected, not a bug.
