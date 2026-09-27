"""Moving parameters to other conditions, and the datasheet check that validates it.

The strongest test in this file is the one that closes the loop: fit a module
from its datasheet, translate the fit to a warmer cell, and check that the
open-circuit voltage falls at exactly the rate the datasheet publishes. Nothing
in the translation was told that number directly, so agreement is evidence
about the whole chain rather than about one equation.
"""

from __future__ import annotations

import pytest

from single_diode.curves import characteristic_points
from single_diode.datasheet import Datasheet
from single_diode.translate import (
    BANDGAP_SILICON_EV,
    REFERENCE_IRRADIANCE,
    REFERENCE_TEMPERATURE_C,
    bandgap,
    celsius_to_kelvin,
    translate,
)

# --------------------------------------------------------------------------
# units and guards
# --------------------------------------------------------------------------


def test_celsius_to_kelvin():
    assert celsius_to_kelvin(0.0) == pytest.approx(273.15)
    assert celsius_to_kelvin(25.0) == pytest.approx(298.15)


def test_below_absolute_zero_is_refused():
    with pytest.raises(ValueError, match="absolute zero"):
        celsius_to_kelvin(-300.0)


def test_zero_irradiance_is_refused(fitted, datasheet):
    """Rsh goes as 1/G, so zero irradiance is an infinity, not a dark module."""
    with pytest.raises(ValueError, match="irradiance"):
        translate(fitted, irradiance=0.0, temperature_c=25.0, alpha_sc=datasheet.alpha_sc)


def test_bandgap_falls_with_temperature():
    assert bandgap(25.0) == pytest.approx(BANDGAP_SILICON_EV)
    assert bandgap(75.0) < bandgap(25.0) < bandgap(-25.0)


# --------------------------------------------------------------------------
# the identity translation
# --------------------------------------------------------------------------


def test_translating_to_reference_conditions_changes_nothing(fitted, datasheet):
    same = translate(
        fitted,
        irradiance=REFERENCE_IRRADIANCE,
        temperature_c=REFERENCE_TEMPERATURE_C,
        alpha_sc=datasheet.alpha_sc,
    )
    assert same.photocurrent == pytest.approx(fitted.photocurrent, rel=1e-12)
    assert same.saturation_current == pytest.approx(fitted.saturation_current, rel=1e-12)
    assert same.resistance_series == pytest.approx(fitted.resistance_series, rel=1e-12)
    assert same.resistance_shunt == pytest.approx(fitted.resistance_shunt, rel=1e-12)
    assert same.thermal_voltage == pytest.approx(fitted.thermal_voltage, rel=1e-12)


# --------------------------------------------------------------------------
# closing the loop on the published coefficients
# --------------------------------------------------------------------------


def test_open_circuit_voltage_falls_at_the_published_rate(fitted, datasheet):
    """dVoc/dT must reproduce beta_oc from the datasheet.

    A sign error, a kelvin/celsius slip or a wrong band gap term all land here
    and nowhere else, because this is the only quantity the fit was given that
    is not one of the four operating points.
    """
    warm = translate(
        fitted,
        irradiance=REFERENCE_IRRADIANCE,
        temperature_c=REFERENCE_TEMPERATURE_C + 20.0,
        alpha_sc=datasheet.alpha_sc,
    )
    measured = (characteristic_points(warm).v_oc - datasheet.v_oc) / 20.0
    assert measured == pytest.approx(datasheet.beta_oc, rel=0.02)


def test_short_circuit_current_rises_at_the_published_rate(fitted, datasheet):
    warm = translate(
        fitted,
        irradiance=REFERENCE_IRRADIANCE,
        temperature_c=REFERENCE_TEMPERATURE_C + 25.0,
        alpha_sc=datasheet.alpha_sc,
    )
    measured = (characteristic_points(warm).i_sc - datasheet.i_sc) / 25.0
    assert measured == pytest.approx(datasheet.alpha_sc, rel=0.02)


# --------------------------------------------------------------------------
# signs and scaling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [0.0, 25.0, 45.0, 65.0])
def test_warmer_cells_lose_voltage_and_power(fitted, datasheet, temperature):
    hot = translate(
        fitted,
        irradiance=REFERENCE_IRRADIANCE,
        temperature_c=temperature,
        alpha_sc=datasheet.alpha_sc,
    )
    cold = translate(
        fitted,
        irradiance=REFERENCE_IRRADIANCE,
        temperature_c=temperature - 10.0,
        alpha_sc=datasheet.alpha_sc,
    )
    assert characteristic_points(hot).v_oc < characteristic_points(cold).v_oc
    assert characteristic_points(hot).p_mp < characteristic_points(cold).p_mp
    assert characteristic_points(hot).i_sc > characteristic_points(cold).i_sc


def test_photocurrent_is_proportional_to_irradiance(fitted, datasheet):
    full = translate(
        fitted, irradiance=1000.0, temperature_c=25.0, alpha_sc=datasheet.alpha_sc
    )
    half = translate(
        fitted, irradiance=500.0, temperature_c=25.0, alpha_sc=datasheet.alpha_sc
    )
    assert half.photocurrent == pytest.approx(0.5 * full.photocurrent, rel=1e-12)


def test_shunt_resistance_rises_as_irradiance_falls(fitted, datasheet):
    """Rsh ~ 1/G. Getting this backwards makes low light look too good."""
    dim = translate(
        fitted, irradiance=200.0, temperature_c=25.0, alpha_sc=datasheet.alpha_sc
    )
    assert dim.resistance_shunt == pytest.approx(5.0 * fitted.resistance_shunt, rel=1e-12)


def test_series_resistance_is_unchanged(fitted, datasheet):
    moved = translate(
        fitted, irradiance=350.0, temperature_c=41.0, alpha_sc=datasheet.alpha_sc
    )
    assert moved.resistance_series == pytest.approx(fitted.resistance_series, rel=1e-12)


def test_thermal_voltage_is_proportional_to_absolute_temperature(fitted, datasheet):
    moved = translate(
        fitted, irradiance=1000.0, temperature_c=100.0, alpha_sc=datasheet.alpha_sc
    )
    expected = fitted.thermal_voltage * celsius_to_kelvin(100.0) / celsius_to_kelvin(25.0)
    assert moved.thermal_voltage == pytest.approx(expected, rel=1e-12)


def test_saturation_current_rises_steeply_with_temperature(fitted, datasheet):
    """I0 roughly doubles every 10 K; an order of magnitude over 40 K."""
    warm = translate(
        fitted, irradiance=1000.0, temperature_c=65.0, alpha_sc=datasheet.alpha_sc
    )
    ratio = warm.saturation_current / fitted.saturation_current
    assert 10.0 < ratio < 1000.0


def test_low_irradiance_improves_the_fill_factor(fitted, datasheet):
    """Less current means smaller series losses and a squarer curve."""
    bright = characteristic_points(
        translate(
            fitted, irradiance=1000.0, temperature_c=25.0, alpha_sc=datasheet.alpha_sc
        )
    )
    dim = characteristic_points(
        translate(fitted, irradiance=200.0, temperature_c=25.0, alpha_sc=datasheet.alpha_sc)
    )
    assert dim.fill_factor > bright.fill_factor


def test_unit_slip_in_the_bandgap_is_caught(fitted, datasheet):
    """Passing the band gap in joules instead of eV overflows, loudly.

    1.121 eV is 1.8e-19 J. Expressed in the units the exponent expects, that
    is off by nineteen orders of magnitude, and a silent inf would poison
    every downstream number.
    """
    with pytest.raises(OverflowError, match="band gap"):
        translate(
            fitted,
            irradiance=1000.0,
            temperature_c=60.0,
            alpha_sc=datasheet.alpha_sc,
            bandgap_ref_ev=1.0e6,
        )


# --------------------------------------------------------------------------
# datasheet validation
# --------------------------------------------------------------------------


def _sheet(**overrides):
    base = {
        "name": "t",
        "i_sc": 9.0,
        "v_oc": 37.0,
        "i_mp": 8.5,
        "v_mp": 30.0,
        "alpha_sc": 0.004,
        "beta_oc": -0.12,
        "cells_in_series": 60,
        "source": "synthetic",
    }
    return Datasheet(**{**base, **overrides})


def test_datasheet_rejects_positive_beta_oc():
    with pytest.raises(ValueError, match="beta_oc"):
        _sheet(beta_oc=0.12)


def test_datasheet_rejects_negative_alpha_sc():
    with pytest.raises(ValueError, match="alpha_sc"):
        _sheet(alpha_sc=-0.004)


def test_datasheet_rejects_mpp_outside_the_curve():
    with pytest.raises(ValueError, match="i_mp"):
        _sheet(i_mp=9.5)
    with pytest.raises(ValueError, match="v_mp"):
        _sheet(v_mp=40.0)


def test_datasheet_rejects_an_impossible_fill_factor():
    with pytest.raises(ValueError, match="fill factor"):
        _sheet(i_mp=1.0, v_mp=2.0)


def test_datasheet_fill_factor_and_power(datasheet):
    assert datasheet.p_mp == pytest.approx(datasheet.i_mp * datasheet.v_mp)
    assert 0.0 < datasheet.fill_factor < 1.0
