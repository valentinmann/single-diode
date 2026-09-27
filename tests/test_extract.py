"""Fitting five parameters, and measuring what makes the fit converge.

The round trips are the strongest tests here: a synthetic module has a known
answer, so recovery can be asserted against the parameters themselves rather
than against the four points the fit was handed. A fit that reproduces its own
inputs proves very little; one that recovers parameters it never saw proves
the system of equations is right.

The convergence-map tests are the ones that decided the design. They are
written so that they would fail if the claim in ``extract.py`` - that the
starting ideality factor is what matters and the parameterisation is not -
stopped being true.
"""

from __future__ import annotations

import numpy as np
import pytest

from single_diode.curves import characteristic_points
from single_diode.datasheet import Datasheet
from single_diode.extract import (
    IDEALITY_GRID,
    ExtractionError,
    extract,
    initial_guess,
    naive_ideality_estimate,
)
from single_diode.model import thermal_voltage
from single_diode.translate import celsius_to_kelvin
from tests.conftest import SYNTHETIC_SPECS

# --------------------------------------------------------------------------
# round trips against known answers
# --------------------------------------------------------------------------


def test_recovers_known_parameters(synthetic_cases):
    """Build a datasheet from known parameters, fit it, get them back."""
    for name, sheet, truth, _ in synthetic_cases:
        got = extract(sheet).parameters
        assert got.photocurrent == pytest.approx(truth.photocurrent, rel=1e-6), name
        assert got.saturation_current == pytest.approx(
            truth.saturation_current, rel=1e-4
        ), name
        assert got.resistance_series == pytest.approx(truth.resistance_series, rel=1e-4), (
            name
        )
        assert got.resistance_shunt == pytest.approx(truth.resistance_shunt, rel=1e-4), name
        assert got.thermal_voltage == pytest.approx(truth.thermal_voltage, rel=1e-6), name


def test_recovers_the_ideality_factor(synthetic_cases):
    """``n`` is the interpretable parameter; a slip in Ns or T shows up here."""
    by_name = {spec[0]: spec[2] for spec in SYNTHETIC_SPECS}
    for name, sheet, _, _ in synthetic_cases:
        result = extract(sheet)
        assert result.ideality_factor == pytest.approx(by_name[name], rel=1e-5), name


def test_fit_reproduces_the_datasheet_points(datasheet):
    """The real module: no ground truth, so check the four published points."""
    point = characteristic_points(extract(datasheet).parameters)
    assert point.i_sc == pytest.approx(datasheet.i_sc, rel=1e-6)
    assert point.v_oc == pytest.approx(datasheet.v_oc, rel=1e-6)
    assert point.i_mp == pytest.approx(datasheet.i_mp, rel=1e-5)
    assert point.v_mp == pytest.approx(datasheet.v_mp, rel=1e-5)
    assert point.p_mp == pytest.approx(datasheet.p_mp, rel=1e-5)


def test_fitted_parameters_are_physically_sensible(datasheet):
    result = extract(datasheet)
    params = result.parameters
    assert 0.5 <= result.ideality_factor <= 2.5
    assert params.saturation_current < 1e-6
    assert 0.0 < params.resistance_series < datasheet.v_oc / datasheet.i_sc
    assert params.resistance_shunt > datasheet.v_oc / datasheet.i_sc
    assert params.photocurrent == pytest.approx(datasheet.i_sc, rel=0.05)


# --------------------------------------------------------------------------
# why the multi-start exists
# --------------------------------------------------------------------------


def test_the_textbook_ideality_estimate_is_badly_wrong(synthetic_cases):
    """The closed form ignoring both resistances overestimates ``a``.

    This is the measurement behind the design: the naive estimate is far
    enough out that the ``I0`` it implies is orders of magnitude wrong.
    """
    overestimates = 0
    for _name, sheet, truth, _ in synthetic_cases:
        naive_a = naive_ideality_estimate(sheet)
        if naive_a > truth.thermal_voltage * 1.2:
            overestimates += 1
    assert overestimates >= len(synthetic_cases) - 1, (
        "the naive estimate is expected to overestimate on nearly every module"
    )


def test_a_wrong_ideality_start_carries_an_enormous_saturation_current_error(datasheet):
    """A 50% error in ``a`` is four orders of magnitude in ``I0``."""
    good = initial_guess(datasheet, 1.0)
    bad = initial_guess(datasheet, 1.5)
    ratio = bad[1] / good[1]
    assert ratio > 1e3, f"expected a large I0 blow-up, got {ratio:.3g}"


def test_single_start_from_the_naive_estimate_fails_where_multi_start_works(
    synthetic_cases, datasheet
):
    """The design decision, measured.

    Restricting the grid to the ideality factor the textbook formula implies
    reproduces a single-start fit. On this corpus it fails on two modules of
    seven - including the one real module, which is the case that matters -
    while the full grid solves all seven. Two in seven is not a dramatic
    failure rate, and the number is written down here rather than rounded up
    into a better story.
    """
    sheets = [sheet for _, sheet, _, _ in synthetic_cases] + [datasheet]
    naive_failures = 0
    for sheet in sheets:
        unit = thermal_voltage(
            1.0, sheet.cells_in_series, celsius_to_kelvin(sheet.reference_temperature_c)
        )
        naive_n = naive_ideality_estimate(sheet) / unit
        try:
            extract(sheet, ideality_grid=(naive_n,))
        except ExtractionError:
            naive_failures += 1

        extract(sheet)  # the full grid must always work

    assert naive_failures >= 1, (
        "the naive single start is expected to fail on part of the corpus"
    )


def test_the_real_module_is_one_the_naive_start_cannot_fit(datasheet):
    """Named explicitly, because it is the case the design was chosen for."""
    unit = thermal_voltage(
        1.0,
        datasheet.cells_in_series,
        celsius_to_kelvin(datasheet.reference_temperature_c),
    )
    naive_n = naive_ideality_estimate(datasheet) / unit
    with pytest.raises(ExtractionError):
        extract(datasheet, ideality_grid=(naive_n,))
    assert extract(datasheet).max_abs_residual < 1e-8


def test_every_module_is_solved_by_the_default_grid(synthetic_cases, datasheet):
    sheets = [sheet for _, sheet, _, _ in synthetic_cases] + [datasheet]
    for sheet in sheets:
        result = extract(sheet)
        assert result.max_abs_residual < 1e-8, sheet.name
        assert result.ideality_start in IDEALITY_GRID


def test_attempts_counts_the_starting_points_used(datasheet):
    result = extract(datasheet)
    assert result.attempts >= 1
    assert result.attempts <= len(IDEALITY_GRID)
    assert IDEALITY_GRID[result.attempts - 1] == result.ideality_start


def test_empty_grid_is_rejected(datasheet):
    with pytest.raises(ValueError, match="at least one"):
        extract(datasheet, ideality_grid=())


def test_a_grid_of_hopeless_starts_raises(datasheet):
    with pytest.raises(ExtractionError, match="could not fit"):
        extract(datasheet, ideality_grid=(0.01, 60.0))


# --------------------------------------------------------------------------
# the parameterisation, kept honest
# --------------------------------------------------------------------------


@pytest.mark.parametrize("parameterisation", ["linear", "log"])
def test_both_parameterisations_solve_the_corpus(
    synthetic_cases, datasheet, parameterisation
):
    """Measured, not assumed.

    The module docstring claims neither parameterisation is the deciding
    factor once the multi-start is in place. This is that claim.
    """
    sheets = [sheet for _, sheet, _, _ in synthetic_cases] + [datasheet]
    for sheet in sheets:
        result = extract(sheet, parameterisation=parameterisation)
        assert result.max_abs_residual < 1e-8, f"{sheet.name} / {parameterisation}"


def test_parameterisations_agree_on_the_answer(synthetic_cases):
    """Two routes to the same root must land on the same parameters."""
    for name, sheet, _, _ in synthetic_cases:
        linear = extract(sheet, parameterisation="linear").parameters
        logged = extract(sheet, parameterisation="log").parameters
        assert logged.photocurrent == pytest.approx(linear.photocurrent, rel=1e-6), name
        assert logged.saturation_current == pytest.approx(
            linear.saturation_current, rel=1e-3
        ), name
        assert logged.thermal_voltage == pytest.approx(linear.thermal_voltage, rel=1e-5), (
            name
        )


def test_unknown_parameterisation_is_rejected(datasheet):
    with pytest.raises(ValueError, match="unknown parameterisation"):
        extract(datasheet, parameterisation="quadratic")


# --------------------------------------------------------------------------
# the initial guess
# --------------------------------------------------------------------------


def test_initial_guess_is_positive_and_ordered(datasheet):
    il, i0, rs, rsh, a = initial_guess(datasheet, 1.2)
    assert il > 0
    assert i0 > 0
    assert rs >= 0
    assert rsh > 0
    assert a > 0
    assert rsh > rs, "shunt resistance must start above series resistance"
    assert i0 < il, "saturation current is many orders below the light current"


def test_initial_guess_scales_with_the_ideality_factor(datasheet):
    a_low = initial_guess(datasheet, 1.0)[4]
    a_high = initial_guess(datasheet, 2.0)[4]
    assert a_high == pytest.approx(2.0 * a_low, rel=1e-12)


def test_initial_guess_rejects_a_nonpositive_ideality(datasheet):
    with pytest.raises(ValueError, match="ideality_factor"):
        initial_guess(datasheet, 0.0)


# --------------------------------------------------------------------------
# refusing bad answers
# --------------------------------------------------------------------------


def _mislabelled() -> Datasheet:
    """The real module's numbers with the cell count badly wrong.

    ``n`` has to absorb the discrepancy, so the residuals stay small while the
    physics does not.
    """
    return Datasheet(
        name="mislabelled",
        i_sc=5.10,
        v_oc=59.40,
        i_mp=4.69,
        v_mp=46.90,
        alpha_sc=0.004539,
        beta_oc=-0.222156,
        cells_in_series=6,
        source="synthetic, deliberately mislabelled",
    )


# A 96-cell module declared as 6 cells needs n ~ 16 to absorb the discrepancy,
# which is far outside the default grid, so the start has to be given.
_MISLABELLED_GRID = (16.0,)


def test_physical_check_rejects_an_absurd_ideality_factor():
    with pytest.raises(ExtractionError, match="ideality factor"):
        extract(_mislabelled(), ideality_grid=_MISLABELLED_GRID, check_physical=True)


def test_physical_check_can_be_disabled():
    result = extract(_mislabelled(), ideality_grid=_MISLABELLED_GRID, check_physical=False)
    assert result.ideality_factor > 2.5
    assert result.max_abs_residual < 1e-8, (
        "the fit matches the datasheet perfectly; only the physics is wrong"
    )


def test_the_default_grid_alone_rejects_a_mislabelled_module():
    """The bounded grid is itself a physical constraint, not only a heuristic."""
    with pytest.raises(ExtractionError):
        extract(_mislabelled(), check_physical=False)


def test_tolerance_is_enforced(datasheet):
    with pytest.raises(ExtractionError):
        extract(datasheet, tolerance=1e-30)


def test_result_reports_residuals(datasheet):
    result = extract(datasheet)
    assert result.residuals.shape == (5,)
    assert np.all(np.isfinite(result.residuals))
