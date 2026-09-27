"""Maximum power point trackers, including the failure a static test hides.

Under constant sun both algorithms find the maximum and sit on it. That is the
easy test and it proves little. The test worth having is the ramp: perturb and
observe attributes every change in power to its own last voltage step, so while
the irradiance is moving it can walk steadily the wrong way.
"""

from __future__ import annotations

import numpy as np
import pytest

from single_diode.curves import characteristic_points
from single_diode.mppt import incremental_conductance, perturb_and_observe
from single_diode.translate import translate

TRACKERS = (perturb_and_observe, incremental_conductance)


# --------------------------------------------------------------------------
# steady conditions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tracker", TRACKERS)
def test_finds_the_maximum_under_constant_sun(reference_params, tracker):
    result = tracker(reference_params, steps=300, step_voltage=0.2)
    assert result.final_voltage_error < 0.5


@pytest.mark.parametrize("tracker", TRACKERS)
def test_settles_within_one_perturbation_of_the_maximum(reference_params, tracker):
    """Steady-state oscillation of one step is the algorithm working."""
    step = 0.3
    result = tracker(reference_params, steps=400, step_voltage=step)
    settled = result.voltage[-50:]
    true_v_mp = characteristic_points(reference_params).v_mp
    assert np.max(np.abs(settled - true_v_mp)) < 2.5 * step


@pytest.mark.parametrize("tracker", TRACKERS)
def test_tracking_efficiency_is_high_under_constant_sun(reference_params, tracker):
    result = tracker(reference_params, steps=300, step_voltage=0.2)
    assert result.tracking_efficiency > 0.97


@pytest.mark.parametrize("tracker", TRACKERS)
def test_converges_from_either_side(reference_params, tracker):
    true_v_mp = characteristic_points(reference_params).v_mp
    for start in (0.3 * true_v_mp, 1.6 * true_v_mp):
        result = tracker(
            reference_params, steps=400, step_voltage=0.2, initial_voltage=start
        )
        assert result.final_voltage_error < 0.6, f"failed from {start:.1f} V"


# --------------------------------------------------------------------------
# a moving condition
# --------------------------------------------------------------------------


def _ramp(params, datasheet, steps, start=1000.0, end=200.0):
    """A fast irradiance ramp, one condition per control step."""
    return [
        translate(params, irradiance=g, temperature_c=25.0, alpha_sc=datasheet.alpha_sc)
        for g in np.linspace(start, end, steps)
    ]


@pytest.mark.parametrize("tracker", TRACKERS)
def test_a_ramp_degrades_both_trackers(fitted, datasheet, tracker):
    """The classic failure, provoked and measured rather than described.

    With the sun falling fast, the power drop caused by the weather is larger
    than the power change caused by the controller's own step, so the sign
    test that drives the algorithm reads the wrong signal.
    """
    steps = 150
    conditions = _ramp(fitted, datasheet, steps)

    ramped = tracker(conditions, steps=steps, step_voltage=0.4)
    steady = tracker(
        conditions[-1],
        steps=steps,
        step_voltage=0.4,
        initial_voltage=float(ramped.voltage[0]),
    )

    assert ramped.final_voltage_error > steady.final_voltage_error


def test_incremental_conductance_is_not_immune_to_a_ramp(fitted, datasheet):
    """Measured, and it corrected what this module first claimed.

    Incremental conductance is usually described as robust to changing
    irradiance because its test is a property of the curve rather than a
    comparison of two powers in time. That holds for the continuous
    algorithm. This implementation, like every discrete one, estimates
    ``dI/dV`` from consecutive samples, and during a ramp those samples come
    from different curves - so the same contamination enters by the same
    door. On this ramp the two trackers follow identical trajectories.

    If a future implementation separates the two effects, this test should
    fail and be replaced by one asserting the advantage.
    """
    steps = 150
    conditions = _ramp(fitted, datasheet, steps)

    po = perturb_and_observe(conditions, steps=steps, step_voltage=0.4)
    inc = incremental_conductance(conditions, steps=steps, step_voltage=0.4)

    assert inc.final_voltage_error == pytest.approx(po.final_voltage_error, rel=1e-9)


@pytest.mark.parametrize("tracker", TRACKERS)
def test_recovers_after_the_ramp_stops(fitted, datasheet, tracker):
    """Whatever happens during the transient, steady sun must be recovered."""
    steps = 400
    conditions = _ramp(fitted, datasheet, steps // 2)
    conditions += [conditions[-1]] * (steps - len(conditions))

    result = tracker(conditions, steps=steps, step_voltage=0.3)
    assert result.final_voltage_error < 1.0


# --------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tracker", TRACKERS)
def test_condition_sequence_length_is_checked(reference_params, tracker):
    with pytest.raises(ValueError, match="conditions for"):
        tracker([reference_params] * 3, steps=10)


@pytest.mark.parametrize("tracker", TRACKERS)
def test_rejects_a_nonpositive_start(reference_params, tracker):
    with pytest.raises(ValueError, match="initial_voltage"):
        tracker(reference_params, steps=10, initial_voltage=0.0)


@pytest.mark.parametrize("tracker", TRACKERS)
def test_result_arrays_line_up(reference_params, tracker):
    steps = 25
    result = tracker(reference_params, steps=steps, step_voltage=0.4)
    assert result.voltage.shape == (steps,)
    assert result.power.shape == (steps,)
    assert result.true_v_mp.shape == (steps,)
    assert result.true_p_mp.shape == (steps,)
    assert np.all(result.voltage > 0.0)
    assert np.all(np.isfinite(result.power))


@pytest.mark.parametrize("tracker", TRACKERS)
def test_measured_power_never_exceeds_the_true_maximum(reference_params, tracker):
    """A tracker cannot extract more than the curve offers."""
    result = tracker(reference_params, steps=120, step_voltage=0.4)
    assert np.all(result.power <= result.true_p_mp + 1e-9)
