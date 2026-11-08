"""Reference-track curvature as a function of arc length.

The public interface is shared by the simulator and all controllers::

    ref = Reference(kind="skidpad")
    kappa = ref.curvature(s)
    ahead = ref.curvature_preview(s, N, ds)

Distances are in metres and signed curvature is in inverse metres (1/m).
"""

import numpy as np


class Reference:
    """Provide the skidpad and double-lane-change reference scenarios.

    ``skidpad`` preserves the original constant-curvature reference. The
    ``double_lane_change`` profile is a smooth, illustrative simulation
    scenario rather than a prescribed Formula Student event layout.
    """

    _VALID_KINDS = {"skidpad", "double_lane_change", "dlc"}

    def __init__(
        self,
        kind="skidpad",
        R=9.125,
        length=200.0,
        start=15.0,
        turn_length=10.0,
        hold_length=10.0,
        kappa_peak=0.07,
    ):
        self.kind = kind
        self.R = float(R)
        self.length = float(length)
        self.start = float(start)
        self.turn_length = float(turn_length)
        self.hold_length = float(hold_length)
        self.kappa_peak = float(kappa_peak)

        if self.kind not in self._VALID_KINDS:
            choices = ", ".join(sorted(self._VALID_KINDS))
            raise ValueError(f"Unknown track kind {kind!r}; choose from: {choices}")
        if self.R <= 0.0 or self.length <= 0.0:
            raise ValueError("R and length must be positive")
        if self.start < 0.0 or self.turn_length <= 0.0 or self.hold_length < 0.0:
            raise ValueError("Invalid double-lane-change segment lengths")
        if self.kappa_peak < 0.0:
            raise ValueError("kappa_peak must be nonnegative")

    @staticmethod
    def _smooth_bump(distance, segment_length):
        """Return a sine-squared bump with zero value and slope at each end."""
        return float(np.sin(np.pi * distance / segment_length) ** 2)

    def curvature(self, s):
        """Return signed reference curvature at arc length ``s`` [1/m]."""
        s = float(s)

        if self.kind == "skidpad":
            return 1.0 / self.R

        # Four smooth lobes produce: change lane, hold, then return.
        z = s - self.start
        L = self.turn_length
        h = self.hold_length
        A = self.kappa_peak

        if 0.0 <= z < L:
            return A * self._smooth_bump(z, L)
        if L <= z < 2.0 * L:
            return -A * self._smooth_bump(z - L, L)
        if 2.0 * L + h <= z < 3.0 * L + h:
            return -A * self._smooth_bump(z - (2.0 * L + h), L)
        if 3.0 * L + h <= z < 4.0 * L + h:
            return A * self._smooth_bump(z - (3.0 * L + h), L)
        return 0.0

    def curvature_preview(self, s, N, ds):
        """Return ``N`` curvature samples spaced ``ds`` metres, shape ``(N,)``."""
        return np.array(
            [self.curvature(s + i * ds) for i in range(N)],
            dtype=float,
        )
