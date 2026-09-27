"""The robust Lambert W, including proof that it is needed.

Three groups: agreement with scipy everywhere scipy can answer, correctness
where scipy cannot, and the defining equation held to machine precision across
a range no float can represent.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import lambertw

from single_diode.lambertw import OMEGA, lambertw_exp

# log(float64 max). Above this, exp(x) is inf and the naive route is dead.
OVERFLOW_LIMIT = float(np.log(np.finfo(np.float64).max))


# --------------------------------------------------------------------------
# known values
# --------------------------------------------------------------------------


def test_omega_constant():
    """W(1) is the omega constant, the root of w + ln(w) = 0."""
    assert lambertw_exp(0.0) == pytest.approx(OMEGA, rel=1e-15)


def test_w_of_e_is_one():
    """W(e) = 1 exactly, since 1 * e^1 = e."""
    assert lambertw_exp(1.0) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(
    "x", [-700.0, -100.0, -30.0, -5.0, -1.0, 0.0, 1.0, 5.0, 50.0, 300.0, 700.0]
)
def test_agrees_with_scipy_where_scipy_can_answer(x):
    """Below the overflow ceiling, this must match the reference implementation."""
    expected = float(lambertw(np.exp(x)).real)
    assert lambertw_exp(x) == pytest.approx(expected, rel=1e-13)


# --------------------------------------------------------------------------
# the regime the module exists for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("x", [750.0, 1000.0, 2083.0, 1e4, 1e6])
def test_naive_route_overflows_where_this_does_not(x):
    """The motivating failure, asserted rather than asserted about.

    If this ever stops failing, numpy has changed its float semantics and the
    justification for this whole module needs rereading.
    """
    assert x > OVERFLOW_LIMIT
    with np.errstate(over="ignore"):
        # The overflow is the point of the test, not an accident.
        assert np.isinf(np.exp(x)), "exp(x) should overflow in this regime"
        assert np.isinf(lambertw(np.exp(x)).real), "the naive composition should be inf"

    value = lambertw_exp(x)
    assert np.isfinite(value)
    assert value > 0.0


@pytest.mark.parametrize(
    "x", [-700.0, -300.0, -37.0, -1.0, 0.0, 1.0, 10.0, 709.0, 750.0, 2083.0, 1e4, 1e5, 1e6]
)
def test_defining_equation_holds(x):
    """W + ln(w) = x to machine precision, which is the whole specification."""
    w = float(lambertw_exp(x))
    assert w > 0.0
    assert w + np.log(w) == pytest.approx(x, rel=1e-13, abs=1e-12)


def test_large_argument_asymptotic():
    """For large x, W(e^x) ~ x - ln(x); check the leading behaviour."""
    x = 5000.0
    w = float(lambertw_exp(x))
    assert w == pytest.approx(x - np.log(x), rel=1e-3)
    assert w < x  # W(e^x) is always below x for x > 1


def test_small_argument_asymptotic():
    """For very negative x, W(e^x) ~ e^x."""
    x = -200.0
    assert float(lambertw_exp(x)) == pytest.approx(np.exp(x), rel=1e-12)


# --------------------------------------------------------------------------
# shape, monotonicity, and refusing to guess
# --------------------------------------------------------------------------


def test_vectorised_matches_scalar():
    xs = [-10.0, 0.0, 3.0, 900.0]
    vector = lambertw_exp(xs)
    assert vector.shape == (4,)
    for x, got in zip(xs, vector, strict=True):
        assert got == pytest.approx(float(lambertw_exp(x)), rel=1e-15)


def test_scalar_input_returns_scalar():
    assert np.ndim(lambertw_exp(1.0)) == 0
    assert np.ndim(lambertw_exp([1.0])) == 1


def test_strictly_increasing():
    """W is monotone; a non-monotone result would mean a branch was crossed."""
    xs = np.linspace(-50.0, 5000.0, 400)
    w = lambertw_exp(xs)
    assert np.all(np.diff(w) > 0.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_already_overflowed_input(bad):
    """An inf argument means something upstream overflowed; do not paper over it."""
    with pytest.raises(ValueError, match="nan or \\+inf"):
        lambertw_exp(bad)


def test_negative_infinity_is_allowed():
    """-inf is a legitimate limit: W(e^-inf) = W(0) = 0."""
    assert float(lambertw_exp(-np.inf)) == pytest.approx(0.0, abs=1e-300)
