"""
params.py  
Load vehicle_params.yml and expose a plain object with the numbers the rest
of the code needs. lf / lr are computed from wheelbase + weight distribution
(they are NOT stored in the yml).

"""
import os
import yaml

# path to params/vehicle_params.yml, relative to the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_YML = os.path.join(_HERE, "..", "params", "vehicle_params.yml")


class Params:
    """Flat container for vehicle parameters. Access as P.m, P.Cf, P.lf ..."""

    def __init__(self, yml_path=_YML):
        with open(yml_path, "r") as f:
            raw = yaml.safe_load(f)

        # --- read raw values (each entry is {value, unit, status, source}) ---
        self.m         = raw["m"]["value"]
        self.L         = raw["wheelbase"]["value"]
        self.wf        = raw["weight_front"]["value"]   # front axle load fraction
        self.Iz        = raw["Iz"]["value"]
        self.Cf        = raw["Cf"]["value"]
        self.Cr        = raw["Cr"]["value"]
        self.vx_op     = raw["vx_op"]["value"]
        self.dt        = raw["dt"]["value"]
        self.delta_max = raw["delta_max"]["value"]

        # --- derived geometry ---
        # front axle load fraction wf = lr / L  ->  lr = wf * L
        self.lr = self.wf * self.L
        self.lf = (1.0 - self.wf) * self.L

        # delta_max is still null in the yml. Until it is set in the FYP source
        # yml (then auto-synced), fall back to an assumed value so the pipeline
        # runs. Set the real value in the FYP repo and remove this fallback.
        if self.delta_max is None:
            self.delta_max = 0.50   # rad, ~28.6 deg, ASSUMED -- confirm in FYP yml
            print("[params] WARNING: delta_max is null in yml; using assumed "
                  "0.50 rad. Set the real value in the FYP source yml.")

        self._raw = raw  # keep the full dict around if anyone needs unit/status

    def understeer_gradient(self):
        """K_us = Wf/Cf - Wr/Cr  (Rajamani). ~0 => near-neutral steer."""
        g = 9.81
        Wf = self.m * g * self.lr / self.L
        Wr = self.m * g * self.lf / self.L
        return Wf / self.Cf - Wr / self.Cr


if __name__ == "__main__":
    P = Params()
    print(f"m={P.m} kg, L={P.L} m, lf={P.lf:.3f}, lr={P.lr:.3f}")
    print(f"Iz={P.Iz}, Cf={P.Cf}, Cr={P.Cr}, vx_op={P.vx_op}, dt={P.dt}")
    print(f"delta_max={P.delta_max} rad, K_us={P.understeer_gradient():.2e}")
