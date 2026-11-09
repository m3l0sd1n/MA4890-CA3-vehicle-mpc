"""
pid_analysis.py  [Yash -- Part B]  -- PID baseline: how it was tuned, and why it lags on curves
Run from the repo root:   python scripts/pid_analysis.py        (~30 s)
                          python scripts/pid_analysis.py --quick (skips the gain search)

Uses only the frozen interfaces (plant.step, ctrl.solve, run_sim); it edits
nothing else in the repo. Prints every number quoted in the PID section of the
report and saves four figures to results/:

  pid_1_structural.png  PD on the skidpad: standing error = delta_ss / Kp (sim vs
                        formula), and the delay margin each Kp leaves
  pid_2_traces.png      skidpad + DLC time histories: PD vs PID vs PID + curvature
                        feed-forward, and the PID steering against what the road needs
  pid_3_tuning_space.png every PID tuning that meets the robustness limits, against
                        the one-line feed-forward fix
  pid_4_speed.png       fixed gains tuned at 10 m/s, run at 6-16 m/s

Robustness numbers come from the loop broken at the steering input of the
exact 50 Hz discrete plant:  L(z) = K(z) (zI - Ad)^-1 Bd,
K(z) = [Kp + Ki*dt*z/(z-1), Kd, 0, 0].  Delay margin = phase margin / crossover
frequency = the extra loop delay the closed loop survives before going unstable.
The plant has no delay or actuator lag, so this margin is what stands in for
them; the script checks it by re-running with an injected input delay.
"""
import os
import sys
import argparse
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ca3.params import Params
from ca3.vehicle import LateralErrorPlant
from ca3.reference import Reference
from ca3.controllers.pid import PIDController
from run_sim import run_sim

P = Params()
VX = P.vx_op
DT = P.dt
OUT = os.path.join(ROOT, "results")

# baseline gains = the defaults in pid.py, so the analysis can never drift from the code
_base = PIDController(P, VX)
KP, KI, KD = _base.Kp, _base.Ki, _base.Kd
DM_REQ = 0.100          # [s] delay-margin requirement used in the tuning
ZETA_REQ = 0.6          # minimum closed-loop damping ratio used in the tuning
J_TOL = 0.01            # selection: best-damped feasible set within 1 % of the minimum cost

# same two runs as scripts/run_q1.py
CASES = {"skidpad": (0.5, 10.0), "dlc": (0.0, 80.0 / VX)}
REFS = {"skidpad": Reference(kind="skidpad"), "dlc": Reference(kind="dlc")}

# colours: PID keeps the aqua it has in the group's Q1/Q3 figures; PD and
# PID+FF take the next validated slots (checked with the dataviz validator)
C_PD, C_PID, C_FF = "#eda100", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


# ----------------------------------------------------------------------------- helpers
class DelayedPlant:
    """Wraps the plant and delays the steering command by d samples (robustness check only)."""

    def __init__(self, plant, d):
        self.plant, self.d = plant, d
        self.buf = [0.0] * d

    @property
    def x(self):
        return self.plant.x

    def reset(self, x0=None):
        self.buf = [0.0] * self.d
        return self.plant.reset(x0)

    def step(self, u, kappa):
        if self.d:
            self.buf.append(u)
            u = self.buf.pop(0)
        return self.plant.step(u, kappa)


def run(track, Kp=KP, Ki=KI, Kd=KD, ff=False, vx=VX, T=None, delay=0, dist=None):
    ey0, T0 = CASES[track]
    if dist is not None:
        T0 = dist / vx
    plant = LateralErrorPlant(P, vx)
    if delay:
        plant = DelayedPlant(plant, delay)
    ctrl = PIDController(P, vx, Kp=Kp, Ki=Ki, Kd=Kd, feedforward=ff)
    return run_sim(plant, ctrl, REFS[track], dt=DT, vx=vx,
                   T=T0 if T is None else T, x0=[ey0, 0.0, 0.0, 0.0])


def summary(log):
    ey, u = log["x"][:, 0], log["u"]
    return {
        "rms": float(np.sqrt(np.mean(ey ** 2))),
        "max": float(np.max(np.abs(ey))),
        "final": float(np.mean(ey[-int(1.0 / DT):])),   # mean over last 1 s, as in run_q1
        "umax": float(np.max(np.abs(u))),
    }


class Loop:
    """Exact discrete loop at the steering input for one speed (frequency grid cached)."""

    def __init__(self, vx):
        pl = LateralErrorPlant(P, vx)
        self.Ad, self.Bd = pl.Ad, pl.Bd
        self.w = np.logspace(-2, np.log10(0.9999 * np.pi / DT), 6000)
        self.z = np.exp(1j * self.w * DT)
        self.X = np.array([np.linalg.solve(zi * np.eye(4) - self.Ad, self.Bd).ravel()
                           for zi in self.z])                    # x/u at each frequency

    def margins(self, Kp, Ki, Kd):
        """Return (crossover [rad/s], phase margin [deg], delay margin [s])."""
        z = self.z
        L = (Kp + Ki * DT * z / (z - 1.0)) * self.X[:, 0] + Kd * self.X[:, 1]
        mag = np.abs(L)
        idx = np.where(np.diff(np.sign(mag - 1.0)))[0]
        if len(idx) == 0:
            return np.nan, np.nan, np.nan
        dms, pms = [], []
        for i in idx:                                    # worst case over all crossovers
            pm = (180.0 + np.degrees(np.angle(L[i]))) % 360.0
            pm = pm - 360.0 if pm > 180.0 else pm
            pms.append(pm)
            dms.append(np.radians(pm) / self.w[i] if pm > 0 else 0.0)
        j = int(np.argmin(dms))
        return float(self.w[idx[j]]), float(pms[j]), float(dms[j])

    def damping(self, Kp, Ki, Kd):
        """Closed-loop eigenvalues (plant + integrator state). Returns (spectral radius, min zeta)."""
        Kx = np.array([[Kp + Ki * DT, Kd, 0.0, 0.0]])
        A = np.block([[self.Ad - self.Bd @ Kx, -self.Bd * Ki],
                      [np.array([[DT, 0.0, 0.0, 0.0]]), np.ones((1, 1))]])
        zs = np.linalg.eigvals(A)
        if Ki == 0:     # integrator state is then decoupled (eigenvalue exactly 1): drop it
            zs = np.delete(zs, np.argmin(np.abs(zs - 1.0)))
        s = np.log(zs.astype(complex)) / DT
        s = s[np.abs(s) > 1e-9]
        return float(np.max(np.abs(zs))), float(np.min(-np.real(s) / np.abs(s)))


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)


def save(fig, name):
    # Hedi's rule: flag placeholder parameters on every plot
    flagged = [k for k in ("Cf", "Cr", "Iz") if P._raw[k]["status"] == "placeholder"]
    if flagged:
        fig.text(0.995, 0.003, f"placeholder params: {', '.join(flagged)}; delta_max assumed "
                 f"{P.delta_max:.2f} rad" if P._raw["delta_max"]["value"] is None else
                 f"placeholder params: {', '.join(flagged)}", ha="right", va="bottom",
                 fontsize=7, color=MUTED)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------------------------------------------------------- 1. facts
def section_facts(loop):
    print("\n[1] Baseline and what the road demands")
    kappa = REFS["skidpad"].curvature(0.0)
    ff = _base.ff_gain
    print(f"  gains Kp={KP} rad/m, Ki={KI} rad/(m s), Kd={KD} rad s/m, vx={VX:g} m/s, "
          f"delta_max={np.degrees(P.delta_max):.1f} deg")
    print(f"  skidpad R={1/kappa:.3f} m: a_y = vx^2*kappa = {VX**2*kappa:.2f} m/s^2, "
          f"steering needed delta_ss=(L+K_v vx^2)kappa = {ff*kappa:.5f} rad "
          f"({np.degrees(ff*kappa):.2f} deg); L*kappa alone = {P.L*kappa:.5f} rad")
    wc, pm, dm = loop.margins(KP, KI, KD)
    rho, zeta = loop.damping(KP, KI, KD)
    print(f"  baseline loop: crossover {wc:.2f} rad/s, phase margin {pm:.1f} deg, "
          f"delay margin {dm*1e3:.1f} ms ({dm/DT:.2f} samples), min damping {zeta:.3f}")

    # check the delay margin by injecting a real delay
    d_ok, d_bad = int(np.floor(dm / DT)) - 1, int(np.floor(dm / DT)) + 1
    for d in (d_ok, d_bad):
        ey = run("skidpad", T=30.0, delay=d)["x"][:, 0]
        tail = np.max(np.abs(ey[-int(5 / DT):]))
        print(f"  check with injected delay {d} samples ({d*DT*1e3:.0f} ms): "
              f"max|e_y| over last 5 s = {tail*1e3:.3g} mm -> "
              f"{'settles' if tail < 0.05 else 'unstable / limit cycle'}")
    return kappa, ff


# ----------------------------------------------------------------------------- 2. structural
def section_structural(loop, kappa, ff):
    print("\n[2] PD on the skidpad: standing error vs Kp (Kd scaled to keep damping ~0.7)")
    g = VX ** 2 / ff                       # low-frequency gain of delta -> e_y'' [1/s^2]
    Kps = np.round(np.arange(0.2, 3.01, 0.1), 2)
    rows = []
    for Kp in Kps:
        Kd = 2 * 0.7 * np.sqrt(Kp / g)     # double-integrator rule: zeta = g*Kd / (2 sqrt(g Kp))
        e_sim = summary(run("skidpad", Kp=Kp, Ki=0.0, Kd=Kd, T=20.0))["final"]
        e_formula = -ff * kappa / Kp
        _, _, dm = loop.margins(Kp, 0.0, Kd)
        rows.append((Kp, Kd, e_sim, e_formula, dm))
    rows = np.array(rows)
    err = np.max(np.abs(rows[:, 2] - rows[:, 3]))
    print(f"  formula e_ss = -delta_ss/Kp vs simulation, worst mismatch over Kp=0.2..3: {err*1e3:.2e} mm")
    print(f"  equivalently e_ss = a_y / wn^2 with wn^2 = g*Kp, g = vx^2/(L+K_v vx^2) = {g:.2f} 1/s^2")
    for Kp in (0.6, 1.0, 2.0, 3.0):
        r = rows[np.argmin(np.abs(rows[:, 0] - Kp))]
        print(f"   Kp={r[0]:.1f}: e_ss={r[2]*1e3:7.1f} mm (formula {r[3]*1e3:7.1f}), delay margin {r[4]*1e3:5.1f} ms")
    feasible = rows[rows[:, 4] >= DM_REQ]
    print(f"  largest Kp meeting the {DM_REQ*1e3:.0f} ms delay margin: {feasible[-1,0]:.1f} rad/m "
          f"-> smallest PD standing error {abs(feasible[-1,2])*1e3:.0f} mm")

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True)
    ax[0].plot(rows[:, 0], np.abs(rows[:, 3]) * 1e3, color=MUTED, lw=2, label=r"formula  $\delta_{ss}/K_p$")
    ax[0].plot(rows[:, 0], np.abs(rows[:, 2]) * 1e3, "o", color=C_PD, ms=8, mec="white", mew=1.5,
               label="simulation (PD)", zorder=3)
    ax[0].set_ylabel("standing error |e_y| on skidpad [mm]", color=INK)
    ax[0].set_title("(a) A P/PD controller must hold an error to hold the steering the curve needs",
                    loc="left", fontsize=10, color=INK)
    ax[0].legend(frameon=False, labelcolor=INK)
    ax[1].plot(rows[:, 0], rows[:, 4] * 1e3, color=C_PD, lw=2)
    ax[1].axhline(DM_REQ * 1e3, color=INK, ls="--", lw=1)
    ax[1].fill_between(rows[:, 0], 0, DM_REQ * 1e3, color=GRID, alpha=0.6, lw=0)
    ax[1].annotate(f"{DM_REQ*1e3:.0f} ms requirement: below this line the gain is not usable on a car with delay",
                   (rows[-1, 0], DM_REQ * 1e3), xytext=(0, 5), textcoords="offset points",
                   ha="right", fontsize=8, color=MUTED)
    ax[1].set_ylabel("delay margin [ms]", color=INK)
    ax[1].set_xlabel(r"proportional gain $K_p$ [rad/m]", color=INK)
    ax[1].set_title("(b) ...and every increase in gain spends robustness", loc="left", fontsize=10, color=INK)
    ax[1].set_ylim(0, None)
    for a in ax:
        a.axvline(KP, color=MUTED, lw=1, ls=":")
        style(a)
    ax[0].annotate(f"chosen Kp = {KP:g}", (KP, ax[0].get_ylim()[1]), xytext=(4, -12),
                   textcoords="offset points", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "pid_1_structural.png")


# ----------------------------------------------------------------------------- 3. tuning
def section_tuning(loop, quick):
    print("\n[3] Gain tuning: min RMS(skidpad)+RMS(DLC) s.t. delay margin >= "
          f"{DM_REQ*1e3:.0f} ms and damping >= {ZETA_REQ}")
    if quick:
        print("  --quick: search skipped")
        return None
    grid = itertools.product(np.round(np.arange(0.5, 1.301, 0.05), 3),
                             np.round(np.arange(0.10, 0.301, 0.01), 3),
                             np.round(np.arange(0.0, 3.001, 0.25), 3))
    rows, n = [], 0
    for Kp, Kd, Ki in grid:
        n += 1
        rho, zeta = loop.damping(Kp, Ki, Kd)
        if rho >= 1.0 or zeta < ZETA_REQ:
            continue
        _, _, dm = loop.margins(Kp, Ki, Kd)
        if not dm >= DM_REQ:
            continue
        s, d = summary(run("skidpad", Kp, Ki, Kd)), summary(run("dlc", Kp, Ki, Kd))
        rows.append((Kp, Ki, Kd, dm, zeta, s["rms"], s["final"], d["rms"], d["max"]))
    R = np.array(rows)
    J = R[:, 5] + R[:, 7]
    print(f"  {n} gain sets checked, {len(R)} meet both robustness limits")
    hdr = "   Kp    Ki    Kd   DM[ms]  zeta | skid rms  final | dlc rms   max  [mm] |  J [mm]"
    print(hdr)

    def line(r, j):
        return (f"  {r[0]:4.2f} {r[1]:5.2f} {r[2]:5.2f}  {r[3]*1e3:6.1f} {r[4]:5.2f} |"
                f" {r[5]*1e3:7.1f} {r[6]*1e3:6.1f} | {r[7]*1e3:6.1f} {r[8]*1e3:6.1f}      | {j*1e3:6.1f}")
    for i in np.argsort(J)[:5]:
        print(line(R[i], J[i]))
    ib = np.argmin(J)
    pd = np.where(R[:, 1] == 0, J, np.inf)
    ipd = np.argmin(pd)
    near = np.where(J <= J[ib] * (1.0 + J_TOL))[0]
    isel = near[np.argmax(R[near, 4])]          # selection rule: most damping within J_TOL of the optimum
    print("  best PD (Ki=0):")
    print(line(R[ipd], J[ipd]))
    print(f"  selected (most damped of the {len(near)} sets within {J_TOL:.0%} of the minimum cost):")
    print(line(R[isel], J[isel]))
    print(f"  selected cost is {100*(J[isel]/J[ib]-1):.1f}% above the optimum; "
          f"damping {R[isel,4]:.2f} vs {R[ib,4]:.2f} at the optimum")
    sel = tuple(np.round(R[isel, :3], 3))
    if np.allclose(sel, (KP, KI, KD)):
        print(f"  pid.py defaults (Kp, Ki, Kd) = {(KP, KI, KD)} match the selection")
    else:
        print(f"  WARNING: pid.py defaults {(KP, KI, KD)} differ from the selection {sel} "
              "(parameters changed?) -- update the defaults in pid.py")
    print(f"  best PID vs best PD: J {J[ib]*1e3:.1f} vs {J[ipd]*1e3:.1f} mm")
    return R


# ----------------------------------------------------------------------------- 4. traces
class TermRecorder:
    """Pass-through around a PIDController that records its integrator state each call."""

    def __init__(self, ctrl):
        self.ctrl, self.integral = ctrl, []

    def solve(self, x, kappa_ahead):
        out = self.ctrl.solve(x, kappa_ahead)
        self.integral.append(self.ctrl._integral)
        return out


def run_terms(track):
    """Baseline PID run that also returns its P, I and D steering contributions [rad]."""
    ey0, T = CASES[track]
    ctrl = PIDController(P, VX)
    rec = TermRecorder(ctrl)
    log = run_sim(LateralErrorPlant(P, VX), rec, REFS[track], dt=DT, vx=VX, T=T,
                  x0=[ey0, 0.0, 0.0, 0.0])
    terms = {"P": -ctrl.Kp * log["x"][:, 0],
             "I": -ctrl.Ki * np.array(rec.integral),
             "D": -ctrl.Kd * log["x"][:, 1]}
    return log, terms


C_P, C_I, C_D = "#e87ba4", "#008300", MUTED     # P magenta, I green, D muted ink (validated pair)


def section_traces(kappa, ff):
    print("\n[4] Time histories, PD vs PID vs PID + curvature feed-forward (Q1 runs)")
    variants = [("PD (Ki = 0)", dict(Ki=0.0), C_PD, "--"),
                ("PID (baseline)", dict(), C_PID, "-"),
                ("PID + feed-forward", dict(ff=True), C_FF, "-.")]
    logs = {}
    print("  variant              | skid rms  max  final | dlc rms   max | max steer [deg]")
    for name, kw, _, _ in variants:
        s_log, d_log = run("skidpad", **kw), run("dlc", **kw)
        logs[name] = (s_log, d_log)
        s, d = summary(s_log), summary(d_log)
        print(f"  {name:20s} | {s['rms']*1e3:7.1f} {s['max']*1e3:5.0f} {s['final']*1e3:6.1f} |"
              f" {d['rms']*1e3:6.1f} {d['max']*1e3:5.1f} | {np.degrees(max(s['umax'], d['umax'])):5.1f}")
    pdff = summary(run("dlc", Ki=0.0, ff=True))
    print(f"  (PD + feed-forward, Ki=0: DLC rms {pdff['rms']*1e3:.1f} mm, max {pdff['max']*1e3:.1f} mm)")

    s_log, s_terms = run_terms("skidpad")
    d_log, d_terms = run_terms("dlc")
    ey = s_log["x"][:, 0]
    k_pk = int(np.argmin(ey))
    out = np.where(np.abs(ey) > 0.01)[0]
    print(f"  PID skidpad: crosses to {ey[k_pk]*1e3:.0f} mm at t={s_log['t'][k_pk]:.2f} s, "
          f"inside 1 cm for good after t={(out[-1]+1)*DT:.2f} s; at the end the I term supplies "
          f"{np.degrees(s_terms['I'][-1]):.2f} deg of the {np.degrees(ff*kappa):.2f} deg needed")
    pd_s = logs["PD (Ki = 0)"][0]
    print(f"  PD skidpad: steady steering {np.degrees(pd_s['u'][-1]):.3f} deg vs road demand "
          f"{np.degrees(ff*kappa):.3f} deg (all of it from the P term)")
    i = int(np.argmax(np.abs(d_log["x"][:, 0])))
    kpk = REFS["dlc"].kappa_peak
    print(f"  DLC peak error {d_log['x'][i,0]*1e3:.0f} mm at t={d_log['t'][i]:.2f} s; there the steering is "
          f"P {np.degrees(d_terms['P'][i]):.2f} + I {np.degrees(d_terms['I'][i]):.2f} + "
          f"D {np.degrees(d_terms['D'][i]):.2f} deg vs demand {np.degrees(ff*d_log['kappa'][i]):.2f} deg")
    print(f"  rule of thumb: peak error ~ (L kappa_peak)/Kp = {ff*kpk/KP*1e3:.0f} mm "
          f"(sim: PD {summary(logs['PD (Ki = 0)'][1])['max']*1e3:.0f}, PID {summary(d_log)['max']*1e3:.0f} mm)")

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex="col")
    for col, track, title in ((0, "skidpad", "Skidpad (R = 9.125 m), start 0.5 m off"),
                              (1, "dlc", "Double lane change, start on path")):
        a = ax[0, col]
        for name, _, c, ls in variants:
            lg = logs[name][col]
            a.plot(lg["t"], lg["x"][:, 0] * 1e3, color=c, lw=2, ls=ls, label=name)
        a.axhline(0, color=MUTED, lw=0.8)
        a.set_title(f"({'ab'[col]}) {title}: cross-track error", loc="left", fontsize=10, color=INK)
        a.set_ylabel("e_y [mm]", color=INK)
        a.legend(frameon=False, fontsize=8, labelcolor=INK,
                 loc="upper right" if col == 0 else "lower left")

        lg, terms = (s_log, s_terms) if col == 0 else (d_log, d_terms)
        b = ax[1, col]
        b.plot(lg["t"], np.degrees(ff * lg["kappa"]), color=INK, lw=3, alpha=0.25,
               label="steering the road needs")
        b.plot(lg["t"], np.degrees(terms["P"]), color=C_P, lw=2, label=r"P term  $-K_p e_y$")
        b.plot(lg["t"], np.degrees(terms["I"]), color=C_I, lw=2, ls="--", label=r"I term  $-K_i\int e_y$")
        b.plot(lg["t"], np.degrees(terms["D"]), color=C_D, lw=1.2, ls=":", label=r"D term  $-K_d \dot e_y$")
        b.axhline(0, color=MUTED, lw=0.8)
        b.set_ylabel("steering contribution [deg]", color=INK)
        b.set_xlabel("time [s]", color=INK)
        if col == 0:
            b.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
        else:
            b.set_ylim(-7.5, 11.5)       # headroom so the legend sits above the data
            b.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", ncol=2)
    ax[1, 0].set_title("(c) PID: the I term takes over from P within ~1.3 s, so the error can go",
                       loc="left", fontsize=10, color=INK)
    ax[1, 1].set_title("(d) PID: the I term arrives late and fights the next curve; P (= error) steers",
                       loc="left", fontsize=10, color=INK)
    ax[1, 0].set_ylim(-31, 31)
    n_sat = int(np.sum(np.abs(s_log["u"]) >= P.delta_max - 1e-12))
    ax[1, 0].annotate(f"output clipped at δmax = {np.degrees(P.delta_max):.1f}°\nfor {n_sat} step(s) at the start",
                      (0.35, -27), xytext=(8, 0), textcoords="offset points", fontsize=7, color=MUTED,
                      va="center")
    ax[0, 0].set_xlim(0, 5)
    for a in ax.ravel():
        style(a)
    fig.tight_layout()
    save(fig, "pid_2_traces.png")
    return logs


# ----------------------------------------------------------------------------- 5. tuning space
def section_space(R):
    if R is None:
        return
    print("\n[5] Every robust tuning vs the feed-forward fix")
    ff_s, ff_d = summary(run("skidpad", ff=True)), summary(run("dlc", ff=True))
    b_s, b_d = summary(run("skidpad")), summary(run("dlc"))
    pd_mask = R[:, 1] == 0
    print(f"  best DLC max over all {len(R)} robust tunings: {R[:,8].min()*1e3:.1f} mm; "
          f"baseline gains + feed-forward: {ff_d['max']*1e3:.1f} mm")
    zero = np.abs(R[:, 6]) < 1e-3
    print(f"  tunings with skidpad standing error < 1 mm: {zero.sum()} (all have Ki > 0: {np.all(R[zero,1] > 0)}), "
          f"their best DLC max {R[zero, 8].min()*1e3:.1f} mm")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(R[pd_mask, 8] * 1e3, np.abs(R[pd_mask, 6]) * 1e3, s=22, color=C_PD,
               edgecolor="white", lw=0.6, label=f"PD tunings (Ki = 0), n = {pd_mask.sum()}")
    ax.scatter(R[~pd_mask, 8] * 1e3, np.abs(R[~pd_mask, 6]) * 1e3, s=22, color=C_PID,
               edgecolor="white", lw=0.6, label=f"PID tunings (Ki > 0), n = {(~pd_mask).sum()}")
    ax.scatter([b_d["max"] * 1e3], [abs(b_s["final"]) * 1e3], s=90, marker="D", color=C_PID,
               edgecolor=INK, lw=1.2, zorder=4)
    ax.annotate("chosen baseline", (b_d["max"] * 1e3, abs(b_s["final"]) * 1e3),
                xytext=(-20, 55), textcoords="offset points", fontsize=8, color=INK, ha="right",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    best = R[:, 8].min() * 1e3
    ax.axvline(best, color=MUTED, lw=1, ls=":")
    ax.annotate(f"best any robust\ntuning reaches: {best:.0f} mm", (best, 130), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    ax.scatter([ff_d["max"] * 1e3], [abs(ff_s["final"]) * 1e3], s=140, marker="*", color=C_FF,
               edgecolor=INK, lw=0.8, zorder=4, label="baseline gains + feed-forward")
    ax.annotate("same gains + one line of\ncurvature feed-forward", (ff_d["max"] * 1e3, abs(ff_s["final"]) * 1e3),
                xytext=(10, 22), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(0, None)
    ax.set_ylim(-8, None)
    ax.set_xlabel("double lane change: max |e_y| [mm]", color=INK)
    ax.set_ylabel("skidpad: standing error |e_y| [mm]", color=INK)
    ax.set_title(f"All {len(R)} gain sets with delay margin ≥ {DM_REQ*1e3:.0f} ms and damping ≥ {ZETA_REQ}:\n"
                 "re-tuning moves you around the cloud; adding the road curvature moves you out of it",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    style(ax)
    fig.tight_layout()
    save(fig, "pid_3_tuning_space.png")


# ----------------------------------------------------------------------------- 6. speed
def section_speed():
    print("\n[6] Fixed gains (tuned at 10 m/s) across speed; DLC over the same 80 m as run_sweep")
    speeds = np.arange(6.0, 16.01, 1.0)
    rows = []
    for vx in speeds:
        loop = Loop(vx)
        wc, _, dm = loop.margins(KP, KI, KD)
        _, zeta = loop.damping(KP, KI, KD)
        d = summary(run("dlc", vx=vx, dist=80.0))
        dff = summary(run("dlc", vx=vx, dist=80.0, ff=True))
        rows.append((vx, wc, dm, zeta, d["max"], dff["max"]))
    rows = np.array(rows)
    print("  vx[m/s]  crossover[rad/s]  delay margin[ms]  damping | DLC max PID  PID+FF [mm]")
    for r in rows:
        print(f"  {r[0]:6.0f}  {r[1]:14.1f}  {r[2]*1e3:15.1f}  {r[3]:8.2f} | {r[4]*1e3:10.1f} {r[5]*1e3:7.1f}")
    dm_ok, z_ok = rows[rows[:, 2] >= DM_REQ, 0], rows[rows[:, 3] >= ZETA_REQ, 0]
    print(f"  delay margin >= {DM_REQ*1e3:.0f} ms for vx <= {dm_ok.max():.0f} m/s; damping >= {ZETA_REQ} "
          f"for vx = {z_ok.min():.0f}-{z_ok.max():.0f} m/s (run_sweep uses 8-14 m/s)")

    fig, ax = plt.subplots(3, 1, figsize=(7.5, 7.6), sharex=True)
    mk = dict(lw=2, marker="o", ms=7, mec="white", mew=1.5)
    ax[0].plot(rows[:, 0], rows[:, 2] * 1e3, color=C_PID, **mk)
    ax[0].axhline(DM_REQ * 1e3, color=INK, ls="--", lw=1)
    ax[0].fill_between(rows[:, 0], 0, DM_REQ * 1e3, color=GRID, alpha=0.6, lw=0)
    ax[0].set_ylim(0, None)
    ax[0].set_ylabel("delay margin [ms]", color=INK)
    ax[0].set_title(r"(a) Faster: plant gain grows with $v_x^2$, so the same gains leave less delay margin",
                    loc="left", fontsize=10, color=INK)
    ax[1].plot(rows[:, 0], rows[:, 3], color=C_PID, **mk)
    ax[1].axhline(ZETA_REQ, color=INK, ls="--", lw=1)
    ax[1].fill_between(rows[:, 0], 0, ZETA_REQ, color=GRID, alpha=0.6, lw=0)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("min damping ratio", color=INK)
    ax[1].set_title("(b) ...and damping is only good near the tuning speed (slower is oscillatory too)",
                    loc="left", fontsize=10, color=INK)
    ax[2].plot(rows[:, 0], rows[:, 4] * 1e3, color=C_PID, label="PID (baseline)", **mk)
    ax[2].plot(rows[:, 0], rows[:, 5] * 1e3, color=C_FF, ls="-.", label="PID + curvature feed-forward",
               **{**mk, "marker": "s"})
    ax[2].axhline(_base.ff_gain * REFS["dlc"].kappa_peak / KP * 1e3, color=MUTED, lw=1, ls=":")
    ax[2].annotate(r"$L\kappa_{peak}/K_p$", (rows[0, 0], _base.ff_gain * REFS["dlc"].kappa_peak / KP * 1e3),
                   xytext=(2, 4), textcoords="offset points", fontsize=8, color=MUTED)
    ax[2].set_ylim(0, None)
    ax[2].set_ylabel("DLC max |e_y| [mm]", color=INK)
    ax[2].set_xlabel(r"vehicle speed $v_x$ [m/s]  (gains tuned at 10 m/s)", color=INK)
    ax[2].set_title("(c) Error stays near Lκ/Kp at every speed: the steering demand does not depend on speed",
                    loc="left", fontsize=10, color=INK)
    ax[2].legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right")
    for a in ax:
        a.axvline(VX, color=MUTED, lw=1, ls=":")
        style(a)
    fig.tight_layout()
    save(fig, "pid_4_speed.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the gain search (and figure 3)")
    args = ap.parse_args()
    loop = Loop(VX)
    kappa, ff = section_facts(loop)
    section_structural(loop, kappa, ff)
    R = section_tuning(loop, args.quick)
    section_traces(kappa, ff)
    section_space(R)
    section_speed()


if __name__ == "__main__":
    main()
