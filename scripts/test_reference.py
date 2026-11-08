"""Validate and visualise the reference-track geometry.

Run from the repository root with::

    python scripts/test_reference.py
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ca3.reference import Reference  # noqa: E402


def integrate_centreline(ref, ds=0.01):
    """Integrate heading and position from a curvature profile."""
    s = np.arange(0.0, ref.length + ds * 0.5, ds)
    kappa = np.array([ref.curvature(si) for si in s])

    heading = np.zeros_like(s)
    heading[1:] = np.cumsum((kappa[:-1] + kappa[1:]) * ds / 2.0)

    x = np.zeros_like(s)
    y = np.zeros_like(s)
    midpoint_heading = (heading[1:] + heading[:-1]) / 2.0
    x[1:] = np.cumsum(np.cos(midpoint_heading) * ds)
    y[1:] = np.cumsum(np.sin(midpoint_heading) * ds)
    return s, kappa, heading, x, y


def main():
    try:
        Reference(kind="typo")
    except ValueError:
        pass
    else:
        raise AssertionError("An unknown reference kind must raise ValueError")

    skidpad = Reference(kind="skidpad")
    expected_curvature = 1.0 / 9.125
    assert np.isclose(skidpad.curvature(0.0), expected_curvature)
    assert np.allclose(
        skidpad.curvature_preview(0.0, 5, 2.0),
        expected_curvature,
    )

    dlc = Reference(kind="double_lane_change", length=80.0)
    alias = Reference(kind="dlc", length=80.0)
    s, kappa, heading, x, y = integrate_centreline(dlc)
    assert np.allclose(
        dlc.curvature_preview(0.0, 401, 0.2),
        alias.curvature_preview(0.0, 401, 0.2),
    )

    at = lambda at_s: np.searchsorted(s, at_s)
    first_change_end = dlc.start + 2.0 * dlc.turn_length
    manoeuvre_end = first_change_end + dlc.hold_length + 2.0 * dlc.turn_length

    print(f"Skidpad radius: {skidpad.R:.3f} m")
    print(f"Skidpad curvature: {skidpad.curvature(0.0):.5f} 1/m")
    print(f"Lateral shift after first lane change: {y[at(first_change_end)]:.3f} m")
    print(f"Lateral shift after returning: {y[at(manoeuvre_end)]:.6f} m")
    print(f"Heading after returning: {heading[at(manoeuvre_end)]:.6f} rad")

    assert np.isclose(np.max(kappa), dlc.kappa_peak, atol=1e-6)
    assert np.isclose(np.min(kappa), -dlc.kappa_peak, atol=1e-6)
    assert abs(y[at(first_change_end)] - 3.5) < 0.2
    assert abs(y[at(manoeuvre_end)]) < 0.01
    assert abs(heading[at(manoeuvre_end)]) < 0.001

    for boundary in (15.0, 25.0, 35.0, 45.0, 55.0, 65.0):
        assert np.isclose(dlc.curvature(boundary), 0.0, atol=1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].plot(s, kappa)
    axes[0].axhline(0.0, linewidth=0.6)
    axes[0].set(
        title="Double-lane-change curvature profile",
        xlabel="Arc length s (m)",
        ylabel="Curvature (1/m)",
    )
    axes[0].grid(True)

    axes[1].plot(x, y)
    axes[1].set(
        title="Corresponding reference centreline",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True)

    fig.tight_layout()
    out = ROOT / "results" / "reference_validation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"Saved plot: {out}")
    print("All reference tests passed.")


if __name__ == "__main__":
    main()
