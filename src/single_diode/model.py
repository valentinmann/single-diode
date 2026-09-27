"""The single diode equation, and three ways of solving it.

A solar cell under illumination is modelled as a current source in parallel
with a diode and a shunt resistance, in series with a series resistance:

    I = IL - I0 * (exp((V + I*Rs) / a) - 1) - (V + I*Rs) / Rsh

The equation is implicit in ``I``: the current appears inside the exponential.
It has no elementary solution, and the three routes here are the ones used in
practice:

``lambertw``
    A closed form via the Lambert W function, evaluated through
    :func:`~single_diode.lambertw.lambertw_exp` so that the huge intermediate
    argument is never formed. Exact, vectorised and the default.
``brentq``
    Bracketed bisection on the residual. Slow, derivative-free, and it cannot
    miss a root inside its bracket, which makes it the honest referee when the
    other two disagree.
``newton``
    Newton iteration on the residual. Fast, and it will happily converge to
    nonsense from a poor start, so it is here to be compared against.

They are kept separate rather than collapsed into one "best" method because
agreement between them is the strongest correctness check this package has,
and `tests/test_model.py` asserts it across a wide parameter sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, newton

from .lambertw import lambertw_exp

__all__ = [
    "BOLTZMANN",
    "ELEMENTARY_CHARGE",
    "DiodeParameters",
    "current_from_voltage",
    "residual",
    "thermal_voltage",
    "voltage_from_current",
]

# CODATA 2018, exact by the 2019 SI redefinition.
BOLTZMANN = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C

_METHODS = ("lambertw", "brentq", "newton")


@dataclass(frozen=True)
class DiodeParameters:
    """The five parameters, at one irradiance and temperature.

    These are the operating-point parameters, not the reference ones. Getting
    from a datasheet to the reference set is :mod:`single_diode.extract`;
    getting from the reference set to another condition is
    :mod:`single_diode.translate`.

    Attributes:
        photocurrent: ``IL``, light-generated current [A].
        saturation_current: ``I0``, diode reverse saturation current [A].
            Spans roughly 1e-12 to 1e-8 for silicon, which is why everything
            that fits it works in ``log(I0)``.
        resistance_series: ``Rs`` [ohm]. Zero is allowed and handled exactly.
        resistance_shunt: ``Rsh`` [ohm]. Must be strictly positive; infinite
            shunt is expressed as a very large value, not as ``inf``.
        thermal_voltage: ``a = n * Ns * k * T / q`` [V], the modified ideality
            factor. This is the whole-module quantity, not the per-cell one.
    """

    photocurrent: float
    saturation_current: float
    resistance_series: float
    resistance_shunt: float
    thermal_voltage: float

    def __post_init__(self) -> None:
        """Reject parameter sets that do not describe a physical device."""
        if self.saturation_current <= 0.0:
            raise ValueError("saturation_current must be > 0")
        if self.resistance_shunt <= 0.0:
            raise ValueError("resistance_shunt must be > 0 (use a large value, not inf)")
        if self.resistance_series < 0.0:
            raise ValueError("resistance_series must be >= 0")
        if self.thermal_voltage <= 0.0:
            raise ValueError("thermal_voltage must be > 0")


def thermal_voltage(
    ideality_factor: float, cells_in_series: int, temperature_k: float
) -> float:
    """``a = n * Ns * k * T / q`` in volts.

    Args:
        ideality_factor: ``n``, dimensionless, typically 1.0 to 1.5 for silicon.
        cells_in_series: ``Ns``, the number of cells in series in the module.
        temperature_k: cell temperature in kelvin, not celsius.
    """
    if temperature_k <= 0.0:
        raise ValueError("temperature must be in kelvin and > 0")
    return ideality_factor * cells_in_series * BOLTZMANN * temperature_k / ELEMENTARY_CHARGE


def residual(
    current: ArrayLike, voltage: ArrayLike, params: DiodeParameters
) -> NDArray[np.float64]:
    """The single diode equation written as ``f(I, V) = 0``.

    This is the definition everything else is checked against. A solver that
    returns a point where this is not zero is wrong, whatever else it claims,
    and the tests use exactly that as the acceptance criterion.
    """
    i = np.asarray(current, dtype=np.float64)
    v = np.asarray(voltage, dtype=np.float64)
    vd = v + i * params.resistance_series
    return (
        params.photocurrent
        - params.saturation_current * np.expm1(vd / params.thermal_voltage)
        - vd / params.resistance_shunt
        - i
    )


def _i_from_v_lambertw(v: NDArray[np.float64], p: DiodeParameters) -> NDArray[np.float64]:
    """Closed form for ``I(V)``.

    With ``Rs > 0``:

        I = (Rsh*(IL + I0) - V) / (Rs + Rsh) - (a/Rs) * W(theta)

    where ``log(theta)`` is formed directly so ``theta`` itself never exists.
    """
    rs, rsh, a = p.resistance_series, p.resistance_shunt, p.thermal_voltage
    il, i0 = p.photocurrent, p.saturation_current

    if rs == 0.0:
        # Degenerate but common (ideal series resistance): the equation is
        # already explicit, and the Lambert W form divides by Rs.
        return il - i0 * np.expm1(v / a) - v / rsh

    total = rs + rsh
    log_theta = np.log(rs * i0 * rsh / (a * total)) + (
        rsh * (rs * (il + i0) + v) / (a * total)
    )
    return (rsh * (il + i0) - v) / total - (a / rs) * lambertw_exp(log_theta)


def _v_from_i_lambertw(i: NDArray[np.float64], p: DiodeParameters) -> NDArray[np.float64]:
    """Closed form for ``V(I)``.

        V = (IL + I0 - I)*Rsh - I*Rs - a * W(theta)

    Again via ``log(theta)``. For a 96-cell module at open circuit the
    exponent here is above 2000, so this is the call that makes the naive
    formulation return ``nan``.
    """
    rs, rsh, a = p.resistance_series, p.resistance_shunt, p.thermal_voltage
    il, i0 = p.photocurrent, p.saturation_current

    log_theta = np.log(rsh * i0 / a) + rsh * (il + i0 - i) / a
    return (il + i0 - i) * rsh - i * rs - a * lambertw_exp(log_theta)


def _bracket_current(v: float, p: DiodeParameters) -> tuple[float, float]:
    """Bracket ``I(V)`` so the root is guaranteed to lie inside.

    The residual is strictly decreasing in ``I``, so any pair straddling zero
    works. ``I`` cannot exceed ``IL + I0`` (all the light current plus the
    diode's own leakage) and the lower end only needs to be negative enough,
    which matters past open circuit where the current goes negative.
    """
    hi = p.photocurrent + p.saturation_current
    lo = -hi - abs(v) / p.resistance_shunt - 1.0
    return lo, hi


def current_from_voltage(
    voltage: ArrayLike, params: DiodeParameters, method: str = "lambertw"
) -> float | NDArray[np.float64]:
    """Current at a given voltage.

    Args:
        voltage: terminal voltage [V], scalar or array.
        params: the five parameters at this operating condition.
        method: one of ``"lambertw"``, ``"brentq"``, ``"newton"``.

    Returns:
        Current [A]. A float for scalar input, an array otherwise, matching
        the shape of ``voltage``.

    Raises:
        ValueError: on an unknown method.
    """
    v = np.atleast_1d(np.asarray(voltage, dtype=np.float64))
    scalar = np.ndim(voltage) == 0

    if method == "lambertw":
        out = _i_from_v_lambertw(v, params)
    elif method == "brentq":
        out = np.array(
            [
                brentq(
                    lambda i, vv=vv: float(residual(i, vv, params)),
                    *_bracket_current(float(vv), params),
                    xtol=1e-14,
                    rtol=1e-15,
                )
                for vv in v
            ]
        )
    elif method == "newton":
        guess = _i_from_v_lambertw(v, params)
        out = np.array(
            [
                newton(
                    lambda i, vv=vv: float(residual(i, vv, params)),
                    float(g),
                    tol=1e-14,
                    maxiter=100,
                )
                for vv, g in zip(v, np.atleast_1d(guess), strict=True)
            ]
        )
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {_METHODS}")

    return float(out[0]) if scalar else out


def voltage_from_current(
    current: ArrayLike, params: DiodeParameters, method: str = "lambertw"
) -> float | NDArray[np.float64]:
    """Voltage at a given current.

    Args:
        current: terminal current [A], scalar or array.
        params: the five parameters at this operating condition.
        method: one of ``"lambertw"``, ``"brentq"``, ``"newton"``.

    Returns:
        Voltage [V]. A float for scalar input, an array otherwise, matching
        the shape of ``current``.

    Raises:
        ValueError: on an unknown method.
    """
    i = np.atleast_1d(np.asarray(current, dtype=np.float64))
    scalar = np.ndim(current) == 0

    if method == "lambertw":
        out = _v_from_i_lambertw(i, params)
    elif method in ("brentq", "newton"):
        guess = _v_from_i_lambertw(i, params)
        out = np.empty_like(i)
        for k, (ii, g) in enumerate(zip(i, np.atleast_1d(guess), strict=True)):
            if method == "brentq":
                # V is decreasing in I; bracket generously around the closed form.
                span = max(1.0, abs(float(g)))
                out[k] = brentq(
                    lambda vv, ii=ii: float(residual(ii, vv, params)),
                    float(g) - span,
                    float(g) + span,
                    xtol=1e-14,
                    rtol=1e-15,
                )
            else:
                out[k] = newton(
                    lambda vv, ii=ii: float(residual(ii, vv, params)),
                    float(g),
                    tol=1e-14,
                    maxiter=100,
                )
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {_METHODS}")

    return float(out[0]) if scalar else out
