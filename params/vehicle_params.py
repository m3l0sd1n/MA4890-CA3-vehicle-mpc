"""
vehicle_params.py — NTU Formula Racing vehicle parameters (single source of truth)
FYP A014 · MPC for a Formula Student vehicle

Usage:
    from vehicle_params import VehicleParams
    p = VehicleParams()          # default parameter set
    p.m, p.lf, p.Cf              # attribute access with IDE autocomplete
    p.report()                   # print placeholders and missing values

Conventions:
    - All values in SI units
    - Cf / Cr are PER-AXLE cornering stiffnesses (= 2 x single tyre)
    - Comment tags:  [measured]    locked value from team data
                     [derived]     computed from measured values
                     [placeholder] estimate, MUST be replaced before final results
                     [todo]        not yet known, value is None
    - Sources: team vehicle parameter sheet + FYP_HuangXingjian suspension report
      NOTE: confirm whether these values are NVF-2 or NVF-3
"""

from dataclasses import dataclass, fields

# Parameters that are estimates only — flagged by report()
PLACEHOLDERS = ("Iz", "Cf", "Cr")


@dataclass
class VehicleParams:
    # ---------------- Mass & geometry ----------------
    m: float = 320.0                 # [kg]      [measured] total vehicle mass
    wheelbase: float = 1.60          # [m]       [measured]
    weight_front: float = 0.45       # [-]       [measured] front axle load fraction (45:55)
    track_front: float = None        # [m]       [todo]
    track_rear: float = None         # [m]       [todo]
    h_cg: float = None               # [m]       [todo] CG height (needed for L2 load transfer)

    # ---------------- Inertia ----------------
    Iz: float = 203.0                # [kg m^2]  [placeholder] yaw inertia, estimated as m*lf*lr
    Ix: float = None                 # [kg m^2]  [todo] roll inertia (L2 only)

    # ---------------- Tyres ----------------
    Cf: float = 50000.0              # [N/rad]   [placeholder] front axle cornering stiffness
    Cr: float = 61000.0              # [N/rad]   [placeholder] rear axle cornering stiffness
    wheel_radius: float = None       # [m]       [todo]
    mu_peak: float = None            # [-]       [todo] peak friction coefficient
    pacejka_B: float = None          # [-]       [todo] Magic Formula, report Appendix B
    pacejka_C: float = None          # [-]       [todo]
    pacejka_D: float = None          # [-]       [todo]
    pacejka_E: float = None          # [-]       [todo]

    # ---------------- Suspension (L2 double-track only) ----------------
    roll_stiffness_front: float = None   # [N m/rad] [todo]
    roll_stiffness_rear: float = None    # [N m/rad] [todo]
    roll_centre_front: float = None      # [m]       [todo] roll centre height
    roll_centre_rear: float = None       # [m]       [todo]

    # ---------------- Steering & powertrain (= MPC constraints) ----------------
    delta_max: float = None          # [rad]     [todo] front wheel steering angle limit
    delta_rate_max: float = None     # [rad/s]   [todo] steering rate limit
    gear_ratio: float = None         # [-]       [todo] single rear motor: EMRAX 228 + CM200
    T_motor_max: float = None        # [N m]     [todo] see EMRAX 228 datasheet
    a_max: float = None              # [m/s^2]   [todo] acceleration limit
    a_min: float = None              # [m/s^2]   [todo] braking limit (negative)

    # ---------------- Simulation settings (not vehicle physics) ----------------
    vx_op: float = 10.0              # [m/s]     linearisation operating point
    dt: float = 0.02                 # [s]       control period (50 Hz)

    # ---------------- Derived quantities ----------------
    # Accessed like attributes (p.lf) but computed on the fly,
    # so they update automatically when wheelbase or weight_front change.
    @property
    def lf(self):                    # [m] [derived] CG to front axle
        return self.wheelbase * (1 - self.weight_front)

    @property
    def lr(self):                    # [m] [derived] CG to rear axle
        return self.wheelbase * self.weight_front

    @property
    def K_us(self):                  # [rad/(m/s^2)] understeer gradient, > 0 = understeer
        return self.m / self.wheelbase * (self.lr / self.Cf - self.lf / self.Cr)

    # ---------------- Checks ----------------
    def missing(self):
        """Return the names of all parameters that are still None."""
        return [f.name for f in fields(self) if getattr(self, f.name) is None]

    def require(self, *names):
        """Call before a model uses these parameters; raises if any is None,
        so missing values never leak silently into the computation.
        Example: p.require("h_cg", "track_front")"""
        gaps = [n for n in names if getattr(self, n) is None]
        if gaps:
            raise ValueError(f"Missing parameters, fill in vehicle_params.py: {gaps}")

    def report(self):
        print(f"lf = {self.lf:.3f} m, lr = {self.lr:.3f} m, K_us = {self.K_us:.2e}")
        print(f"[placeholder] {list(PLACEHOLDERS)}")
        print(f"[todo] {len(self.missing())}: {self.missing()}")


if __name__ == "__main__":
    VehicleParams().report()
