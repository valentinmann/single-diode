"""I-V and P-V curves, and the four points that characterise them.

The maximum power point is found by solving ``dP/dV = 0`` directly rather than
by taking the largest value on a sampled curve. Sampling gives an answer whose
error is set by the sample spacing and which moves when the spacing changes;
the root of the derivative does not. The derivative is analytic:

    dI/dV = -g / (1 + Rs*g),    g = (I0/a)*exp((V + I*Rs)/a) + 1/Rsh

so ``dP/dV = I + V*dI/dV`` costs one curve evaluation and no differencing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .model import DiodeParameters, current_from_voltage, voltage_from_current

__all__ = ["CharacteristicPoints", "IVCurve", "characteristic_points", "di_dv", "iv_curve"]


@dataclass(frozen=True)
class IVCurve:
    """A sampled I-V curve, from short circuit to open circuit."""

    voltage: NDArray[np.float64]
    current: NDArray[np.float64]

    @property
    def power(self) -> NDArray[np.float64]:
        """Power at each sampled point [W]."""
        return self.voltage * self.current


@dataclass(frozen=True)
class CharacteristicPoints:
    """The four numbers a datasheet would quote, computed from the model.

    Attributes:
        i_sc: current at zero volts [A].
        v_oc: voltage at zero current [V].
        i_mp, v_mp: the maximum power point [A], [V].
        p_mp: ``i_mp * v_mp`` [W].
        fill_factor: ``p_mp / (i_sc * v_oc)``, strictly below 1.
    """

    i_sc: float
    v_oc: float
    i_mp: float
    v_mp: float

    @property
    def p_mp(self) -> float:
        """Maximum power [W]."""
        return self.i_mp * self.v_mp

    @property
    def fill_factor(self) -> float:
        """``p_mp / (i_sc * v_oc)``, strictly below 1 for a real device."""
        return self.p_mp / (self.i_sc * self.v_oc)


def di_dv(voltage: float, params: DiodeParameters) -> float:
    """Slope of the I-V curve at a voltage, analytically.

    Always negative: more voltage, less current. A positive value from this
    function means the parameters are unphysical.
    """
    i = float(current_from_voltage(voltage, params))
    vd = voltage + i * params.resistance_series
    g = (
        params.saturation_current
        / params.thermal_voltage
        * np.exp(vd / params.thermal_voltage)
        + 1.0 / params.resistance_shunt
    )
    return float(-g / (1.0 + params.resistance_series * g))


def _dp_dv(voltage: float, params: DiodeParameters) -> float:
    """``dP/dV = I + V * dI/dV``. Positive left of the knee, negative right."""
    i = float(current_from_voltage(voltage, params))
    return i + voltage * di_dv(voltage, params)


def characteristic_points(params: DiodeParameters) -> CharacteristicPoints:
    """Short circuit, open circuit and the maximum power point.

    The maximum power point is bracketed between zero volts, where ``dP/dV``
    equals the short-circuit current and is therefore positive, and open
    circuit, where the current is zero and the slope negative. A continuous
    function with opposite signs at the ends has a root between them, so the
    bracket cannot fail for physical parameters.
    """
    i_sc = float(current_from_voltage(0.0, params))
    v_oc = float(voltage_from_current(0.0, params))

    lo, hi = 0.0, v_oc
    f_lo, f_hi = _dp_dv(lo, params), _dp_dv(hi, params)
    if f_lo <= 0.0 or f_hi >= 0.0:
        raise ValueError(
            "dP/dV does not change sign between short and open circuit "
            f"(dP/dV(0)={f_lo:.3g}, dP/dV(Voc)={f_hi:.3g}); the parameters do "
            "not describe a photovoltaic device"
        )

    v_mp = brentq(_dp_dv, lo, hi, args=(params,), xtol=1e-12, rtol=1e-15)
    i_mp = float(current_from_voltage(v_mp, params))
    return CharacteristicPoints(i_sc=i_sc, v_oc=v_oc, i_mp=i_mp, v_mp=v_mp)


def iv_curve(
    params: DiodeParameters, num_points: int = 200, method: str = "lambertw"
) -> IVCurve:
    """Sample the I-V curve from short circuit to open circuit.

    Args:
        params: the five parameters at the operating condition.
        num_points: number of samples, at least 2.
        method: solver passed through to
            :func:`~single_diode.model.current_from_voltage`.

    Returns:
        The sampled curve. The last point sits exactly at open circuit, where
        the current is zero to solver tolerance.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    v_oc = float(voltage_from_current(0.0, params))
    voltage = np.linspace(0.0, v_oc, num_points)
    current = np.atleast_1d(current_from_voltage(voltage, params, method=method))
    return IVCurve(voltage=voltage, current=current)
