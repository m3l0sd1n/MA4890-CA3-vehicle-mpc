"""
The track, expressed as curvature kappa(s) along arc length s [m].

STUB: a constant-curvature skidpad is provided so the pipeline runs today.
TODO:
  - make skidpad radius match the real Formula Student skidpad
  - add the double lane change profile (kappa(s) that goes +, 0, -, 0)
Keep the method names below unchanged -- run_sim.py and the controllers rely on them.
"""
import numpy as np


class Reference:
    def __init__(self, kind="skidpad", R=9.125, length=200.0):
        self.kind = kind
        self.R = R
        self.length = length  # total track length [m]

    def curvature(self, s):
        """Curvature at arc length s [1/m]."""
        if self.kind == "skidpad":
            return 1.0 / self.R
        # TODO [Member 2]: double lane change, etc.
        return 0.0

    def curvature_preview(self, s, N, ds):
        """Curvature at the next N points ahead, spacing ds [m]. Shape (N,)."""
        return np.array([self.curvature(s + i * ds) for i in range(N)])
