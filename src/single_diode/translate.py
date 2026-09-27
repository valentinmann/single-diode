"""Moving the five parameters from reference conditions to any other.

A datasheet describes a module at Standard Test Conditions: 1000 W/m^2 and a
cell temperature of 25 C. A module on a roof is almost never there. The De
Soto formulation states each parameter as a function of absorbed irradiance
``G`` and cell temperature ``T``:

    a(T)     = a_ref * T / T_ref
    IL(G,T)  = (G / G_ref) * (IL_ref + alpha_sc * (T - T_ref))
    I0(T)    = I0_ref * (T / T_ref)^3 * exp( (Eg_ref/T_ref - Eg(T)/T) / k )
    Rsh(G)   = Rsh_ref * G_ref / G
    Rs       = Rs_ref                       (constant)

with the band gap itself weakly temperature dependent,

    Eg(T) = Eg_ref * (1 - 0.0002677 * (T - T_ref))

for silicon. Reference: W. De Soto, S.A. Klein, W.A. Beckman, "Improvement and
validation of a model for photovoltaic array performance", Solar Energy 80
(2006) 78-88, as summarised by the Sandia PV Performance Modeling
Collaborative.

Two things here are easy to get wrong and are tested rather than assumed: the
band gap term is a ratio of two ``Eg/T`` values and not a difference of
reciprocals, and ``Rsh`` rises as irradiance falls, which is what makes low
light the numerically awkward regime everywhere else in this package.
"""

from __future__ import annotations

import math

from .model import DiodeParameters

__all__ = [
    "BANDGAP_SILICON_EV",
    "BANDGAP_TEMPERATURE_COEFFICIENT",
    "BOLTZMANN_EV",
    "REFERENCE_IRRADIANCE",
    "REFERENCE_TEMPERATURE_C",
    "bandgap",
    "celsius_to_kelvin",
    "translate",
]

BOLTZMANN_EV = 8.617333262e-5  # eV/K
BANDGAP_SILICON_EV = 1.121  # Eg at 25 C
BANDGAP_TEMPERATURE_COEFFICIENT = 0.0002677  # 1/K, De Soto's dEg/dT / Eg
REFERENCE_IRRADIANCE = 1000.0  # W/m^2
REFERENCE_TEMPERATURE_C = 25.0
_ABSOLUTE_ZERO_C = -273.15


def celsius_to_kelvin(temperature_c: float) -> float:
    """Convert to kelvin, refusing anything below absolute zero."""
    if temperature_c <= _ABSOLUTE_ZERO_C:
        raise ValueError(f"temperature {temperature_c} C is at or below absolute zero")
    return temperature_c - _ABSOLUTE_ZERO_C


def bandgap(
    temperature_c: float,
    bandgap_ref_ev: float = BANDGAP_SILICON_EV,
    reference_temperature_c: float = REFERENCE_TEMPERATURE_C,
) -> float:
    """Band gap [eV] at a cell temperature, linearised about the reference."""
    return bandgap_ref_ev * (
        1.0 - BANDGAP_TEMPERATURE_COEFFICIENT * (temperature_c - reference_temperature_c)
    )


def translate(
    reference: DiodeParameters,
    *,
    irradiance: float,
    temperature_c: float,
    alpha_sc: float,
    reference_irradiance: float = REFERENCE_IRRADIANCE,
    reference_temperature_c: float = REFERENCE_TEMPERATURE_C,
    bandgap_ref_ev: float = BANDGAP_SILICON_EV,
) -> DiodeParameters:
    """Translate reference parameters to another irradiance and temperature.

    Args:
        reference: the five parameters at reference conditions.
        irradiance: absorbed irradiance [W/m^2]. Must be > 0; a module in the
            dark is not described by this model, and dividing by zero
            irradiance is how ``Rsh`` becomes ``inf``.
        temperature_c: cell temperature [C], not ambient.
        alpha_sc: temperature coefficient of short-circuit current [A/K], the
            absolute one from the datasheet rather than the %/K form.
        reference_irradiance: irradiance the reference set describes [W/m^2].
        reference_temperature_c: temperature the reference set describes [C].
        bandgap_ref_ev: band gap at the reference temperature [eV]. The
            default is silicon; thin film technologies differ.

    Returns:
        The five parameters at the requested condition.

    Raises:
        ValueError: if irradiance is not strictly positive.
    """
    if irradiance <= 0.0:
        raise ValueError("irradiance must be > 0 W/m^2")

    t_k = celsius_to_kelvin(temperature_c)
    t_ref_k = celsius_to_kelvin(reference_temperature_c)
    ratio = t_k / t_ref_k

    eg_ref = bandgap_ref_ev
    eg = bandgap(temperature_c, bandgap_ref_ev, reference_temperature_c)

    photocurrent = (irradiance / reference_irradiance) * (
        reference.photocurrent + alpha_sc * (temperature_c - reference_temperature_c)
    )
    saturation_current = (
        reference.saturation_current
        * ratio**3
        * _exp((eg_ref / t_ref_k - eg / t_k) / BOLTZMANN_EV)
    )
    return DiodeParameters(
        photocurrent=photocurrent,
        saturation_current=saturation_current,
        resistance_series=reference.resistance_series,
        resistance_shunt=reference.resistance_shunt * reference_irradiance / irradiance,
        thermal_voltage=reference.thermal_voltage * ratio,
    )


def _exp(x: float) -> float:
    """``exp`` that says which term overflowed instead of returning ``inf``.

    The band gap exponent is a difference of two large numbers divided by a
    small one, so a unit slip - eV against J, celsius against kelvin - lands
    here rather than producing a merely wrong answer.
    """
    if x > 700.0:
        raise OverflowError(
            f"band gap exponent {x:.3g} overflows; check that the band gap is "
            "in eV and the temperatures in kelvin"
        )
    return math.exp(x)
