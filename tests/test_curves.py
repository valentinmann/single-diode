"""Curve shape, and finding the maximum power point without sampling for it.

The maximum power point is the root of ``dP/dV``, not the largest value on a
grid. The difference is testable: a root does not move when the sampling
changes, and a grid maximum does.
"""

from __future__ import annotations

import numpy as np
import pytest

from single_diode.curves import characteristic_points, di_dv, iv_curve
from single_diode.model import DiodeParameters, current_from_voltage

# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------


def test_curve_spans_short_to_open_circuit(reference_params):
    curve = iv_curve(reference_params, num_points=100)
    point = characteristic_points(reference_params)
    assert curve.voltage[0] == pytest.approx(0.0)
    assert curve.voltage[-1] == pytest.approx(point.v_oc, rel=1e-12)
    assert curve.current[0] == pytest.approx(point.i_sc, rel=1e-9)
    assert curve.current[-1] == pytest.approx(0.0, abs=1e-9)


def test_current_is_strictly_decreasing(reference_params):
    curve = iv_curve(reference_params, num_points=300)
    assert np.all(np.diff(curve.current) < 0.0)


def test_power_is_unimodal(reference_params):
    """P(V) rises then falls, with exactly one sign change in its difference."""
    curve = iv_curve(reference_params, num_points=500)
    sign_changes = np.sum(np.diff(np.sign(np.diff(curve.power))) != 0)
    assert sign_changes == 1


def test_slope_is_always_negative(reference_params):
    point = characteristic_points(reference_params)
    for v in np.linspace(0.0, point.v_oc, 25):
        assert di_dv(float(v), reference_params) < 0.0


def test_num_points_is_validated(reference_params):
    with pytest.raises(ValueError, match="num_points"):
        iv_curve(reference_params, num_points=1)


# --------------------------------------------------------------------------
# the maximum power point
# --------------------------------------------------------------------------


def test_mpp_is_the_root_of_dp_dv(reference_params):
    """At the maximum, dI/dV = -Imp/Vmp exactly."""
    point = characteristic_points(reference_params)
    slope = di_dv(point.v_mp, reference_params)
    assert slope == pytest.approx(-point.i_mp / point.v_mp, rel=1e-8)


def test_mpp_beats_every_sampled_point(reference_params):
    """A stronger statement than 'close to the grid maximum'."""
    point = characteristic_points(reference_params)
    curve = iv_curve(reference_params, num_points=2000)
    assert point.p_mp >= curve.power.max() - 1e-9


def test_mpp_does_not_move_with_sampling(reference_params):
    """The reason for solving rather than sampling, asserted."""
    reference = characteristic_points(reference_params)
    for num_points in (10, 100, 5000):
        grid = iv_curve(reference_params, num_points=num_points)
        grid_v = grid.voltage[int(np.argmax(grid.power))]
        solved = characteristic_points(reference_params).v_mp
        assert solved == pytest.approx(reference.v_mp, rel=1e-12)
        # The grid answer, by contrast, is only as good as the spacing.
        if num_points == 10:
            assert abs(grid_v - reference.v_mp) > 1e-3


def test_mpp_lies_strictly_inside_the_curve(reference_params):
    point = characteristic_points(reference_params)
    assert 0.0 < point.v_mp < point.v_oc
    assert 0.0 < point.i_mp < point.i_sc


def test_fill_factor_is_physical(synthetic_cases):
    for name, _, params, _ in synthetic_cases:
        point = characteristic_points(params)
        assert 0.0 < point.fill_factor < 1.0, name
        # Crystalline silicon sits in this band; outside it something is wrong.
        assert 0.5 < point.fill_factor < 0.9, name


def test_characteristic_points_match_the_curve(reference_params):
    point = characteristic_points(reference_params)
    assert float(current_from_voltage(point.v_mp, reference_params)) == pytest.approx(
        point.i_mp, rel=1e-10
    )
    assert point.p_mp == pytest.approx(point.i_mp * point.v_mp, rel=1e-12)


def test_unilluminated_cell_has_no_maximum_power_point():
    """With no photocurrent there is no power to maximise.

    The bracket for dP/dV collapses, and the right answer is to say so rather
    than to return a maximum at the origin.
    """
    dark = DiodeParameters(
        photocurrent=0.0,
        saturation_current=1e-10,
        resistance_series=0.3,
        resistance_shunt=400.0,
        thermal_voltage=1.9,
    )
    with pytest.raises(ValueError, match="does not change sign"):
        characteristic_points(dark)


@pytest.mark.parametrize("method", ["lambertw", "brentq", "newton"])
def test_curve_is_the_same_whichever_solver_built_it(reference_params, method):
    reference = iv_curve(reference_params, num_points=40, method="lambertw")
    other = iv_curve(reference_params, num_points=40, method=method)
    assert np.allclose(reference.current, other.current, rtol=1e-8, atol=1e-10)
