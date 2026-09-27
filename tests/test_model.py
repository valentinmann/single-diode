"""The single diode equation and the three solvers.

The acceptance criterion throughout is the residual: a solver that returns a
point where the defining equation is not satisfied is wrong, whatever else it
reports. Agreement between three independent methods is the second line of
defence, and it is the one that catches a formulation error the residual
cannot, because a wrong closed form can still be self-consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from single_diode.model import (
    DiodeParameters,
    current_from_voltage,
    residual,
    thermal_voltage,
    voltage_from_current,
)

METHODS = ("lambertw", "brentq", "newton")


# --------------------------------------------------------------------------
# parameter validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"saturation_current": 0.0}, "saturation_current"),
        ({"saturation_current": -1e-10}, "saturation_current"),
        ({"resistance_shunt": 0.0}, "resistance_shunt"),
        ({"resistance_shunt": -5.0}, "resistance_shunt"),
        ({"resistance_series": -0.1}, "resistance_series"),
        ({"thermal_voltage": 0.0}, "thermal_voltage"),
    ],
)
def test_rejects_unphysical_parameters(kwargs, message):
    base = {
        "photocurrent": 9.4,
        "saturation_current": 5e-10,
        "resistance_series": 0.35,
        "resistance_shunt": 420.0,
        "thermal_voltage": 1.9,
    }
    with pytest.raises(ValueError, match=message):
        DiodeParameters(**{**base, **kwargs})


def test_infinite_shunt_is_refused_explicitly():
    """``inf`` would propagate silently; a large finite value is the way."""
    with pytest.raises(ValueError, match="not inf"):
        DiodeParameters(9.4, 5e-10, 0.35, -np.inf, 1.9)


def test_thermal_voltage_needs_kelvin():
    with pytest.raises(ValueError, match="kelvin"):
        thermal_voltage(1.2, 60, -10.0)


def test_thermal_voltage_scales_with_cells_and_temperature():
    a60 = thermal_voltage(1.0, 60, 298.15)
    assert thermal_voltage(1.0, 120, 298.15) == pytest.approx(2 * a60)
    assert thermal_voltage(2.0, 60, 298.15) == pytest.approx(2 * a60)
    assert thermal_voltage(1.0, 60, 596.30) == pytest.approx(2 * a60)


# --------------------------------------------------------------------------
# the equation itself
# --------------------------------------------------------------------------


def test_residual_is_zero_on_the_solved_curve(reference_params):
    v = np.linspace(0.0, float(voltage_from_current(0.0, reference_params)), 40)
    i = current_from_voltage(v, reference_params)
    assert np.max(np.abs(residual(i, v, reference_params))) < 1e-10


def test_short_circuit_and_open_circuit_are_consistent(reference_params):
    """Isc and Voc found from opposite directions must agree with the equation."""
    i_sc = float(current_from_voltage(0.0, reference_params))
    v_oc = float(voltage_from_current(0.0, reference_params))
    assert float(residual(i_sc, 0.0, reference_params)) == pytest.approx(0.0, abs=1e-10)
    assert float(residual(0.0, v_oc, reference_params)) == pytest.approx(0.0, abs=1e-10)
    assert 0.0 < i_sc <= reference_params.photocurrent
    assert v_oc > 0.0


def test_current_and_voltage_are_mutual_inverses(reference_params):
    v_oc = float(voltage_from_current(0.0, reference_params))
    v = np.linspace(0.1, 0.98 * v_oc, 25)
    i = current_from_voltage(v, reference_params)
    back = voltage_from_current(i, reference_params)
    assert np.allclose(back, v, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# agreement between methods
# --------------------------------------------------------------------------


def test_all_methods_agree_on_current(reference_params):
    v_oc = float(voltage_from_current(0.0, reference_params))
    v = np.linspace(0.0, v_oc, 30)
    results = {m: current_from_voltage(v, reference_params, method=m) for m in METHODS}
    for method in ("brentq", "newton"):
        assert np.allclose(results["lambertw"], results[method], rtol=1e-8, atol=1e-10), (
            f"lambertw and {method} disagree"
        )


def test_all_methods_agree_on_voltage(reference_params):
    i_sc = float(current_from_voltage(0.0, reference_params))
    i = np.linspace(0.0, 0.99 * i_sc, 20)
    results = {m: voltage_from_current(i, reference_params, method=m) for m in METHODS}
    for method in ("brentq", "newton"):
        assert np.allclose(results["lambertw"], results[method], rtol=1e-8, atol=1e-8)


def test_methods_agree_across_a_parameter_sweep(synthetic_cases):
    """The broad check: every synthetic module, every method, one criterion."""
    for _, _, params, _ in synthetic_cases:
        v_oc = float(voltage_from_current(0.0, params))
        v = np.linspace(0.0, v_oc, 12)
        reference = current_from_voltage(v, params, method="lambertw")
        for method in ("brentq", "newton"):
            other = current_from_voltage(v, params, method=method)
            assert np.allclose(reference, other, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_unknown_method_is_rejected(reference_params, method):
    with pytest.raises(ValueError, match="unknown method"):
        current_from_voltage(1.0, reference_params, method=method + "_typo")
    with pytest.raises(ValueError, match="unknown method"):
        voltage_from_current(1.0, reference_params, method=method + "_typo")


# --------------------------------------------------------------------------
# shape and edge cases
# --------------------------------------------------------------------------


def test_zero_series_resistance_uses_the_explicit_branch(reference_params):
    """Rs = 0 divides by zero in the Lambert W form; it has its own path."""
    ideal = DiodeParameters(
        reference_params.photocurrent,
        reference_params.saturation_current,
        0.0,
        reference_params.resistance_shunt,
        reference_params.thermal_voltage,
    )
    v = np.linspace(0.0, 30.0, 20)
    i = current_from_voltage(v, ideal)
    assert np.all(np.isfinite(i))
    assert np.max(np.abs(residual(i, v, ideal))) < 1e-12


def test_scalar_in_scalar_out(reference_params):
    assert np.ndim(current_from_voltage(1.0, reference_params)) == 0
    assert np.ndim(current_from_voltage([1.0], reference_params)) == 1
    assert np.ndim(voltage_from_current(1.0, reference_params)) == 0


def test_current_decreases_with_voltage(reference_params):
    v_oc = float(voltage_from_current(0.0, reference_params))
    v = np.linspace(0.0, v_oc, 200)
    i = current_from_voltage(v, reference_params)
    assert np.all(np.diff(i) < 0.0), "I(V) must be strictly decreasing"


def test_beyond_open_circuit_the_current_is_negative(reference_params):
    """Past Voc the device consumes current; the model must not clamp at zero."""
    v_oc = float(voltage_from_current(0.0, reference_params))
    assert float(current_from_voltage(v_oc * 1.05, reference_params)) < 0.0


def test_no_module_in_the_corpus_overflows(synthetic_cases):
    """Every module must solve, whatever its Lambert W exponent."""
    for name, _, params, _ in synthetic_cases:
        v_oc = float(voltage_from_current(0.0, params))
        assert np.isfinite(v_oc), f"{name} produced a non-finite Voc"
        assert v_oc > 0.0


def _log_theta(params: DiodeParameters) -> float:
    """The exponent the closed form needs at open circuit."""
    return float(
        np.log(params.resistance_shunt * params.saturation_current / params.thermal_voltage)
        + params.resistance_shunt
        * (params.photocurrent + params.saturation_current)
        / params.thermal_voltage
    )


def test_the_corpus_straddles_the_overflow_ceiling(synthetic_cases):
    """The measurement the figure and the docstrings quote.

    Some modules need an exponent far above the float64 ceiling and some sit
    below it, so this is a failure you cannot rule out by inspecting a
    datasheet. If the corpus ever stops straddling the line, the claim in
    ``plotting.py`` needs rewriting.
    """
    ceiling = float(np.log(np.finfo(np.float64).max))
    exponents = {name: _log_theta(params) for name, _, params, _ in synthetic_cases}

    above = [n for n, v in exponents.items() if v > ceiling]
    below = [n for n, v in exponents.items() if v <= ceiling]
    assert above, f"expected some modules above {ceiling:.0f}, got {exponents}"
    assert below, f"expected some modules below {ceiling:.0f}, got {exponents}"


def test_a_module_past_the_ceiling_still_solves(synthetic_cases):
    """Where the naive composition is inf, this must still return a number."""
    ceiling = float(np.log(np.finfo(np.float64).max))
    overflowing = [
        (name, params)
        for name, _, params, _ in synthetic_cases
        if _log_theta(params) > ceiling
    ]
    assert overflowing, "the corpus should contain at least one such module"

    for name, params in overflowing:
        with np.errstate(over="ignore"):
            assert np.isinf(np.exp(_log_theta(params))), name
        v_oc = float(voltage_from_current(0.0, params))
        assert np.isfinite(v_oc), name
        assert v_oc > 0.0, name
        assert abs(float(residual(0.0, v_oc, params))) < 1e-9, name
