"""Five parameters from four datasheet numbers and a temperature coefficient.

A datasheet gives short-circuit current, open-circuit voltage, the maximum
power point, and how open-circuit voltage moves with temperature. The model
needs ``IL``, ``I0``, ``Rs``, ``Rsh`` and ``a``. Five unknowns, so five
equations:

1. the I-V curve passes through ``(0, Isc)``;
2. it passes through ``(Voc, 0)``;
3. it passes through ``(Vmp, Imp)``;
4. ``dP/dV = 0`` at the maximum power point, i.e. ``dI/dV = -Imp/Vmp``;
5. ``dVoc/dT`` matches the datasheet's ``beta_oc``, evaluated by translating
   the parameters to a slightly warmer cell and re-solving for ``Voc``.

Why this is not simply "call a root finder"
-------------------------------------------

Handing those five equations to a solver from the textbook starting estimate
fails on part of any real corpus: two modules in seven here, including the one
whose numbers come from a manufacturer rather than from this package's own
tests. This is a known difficulty rather than a peculiarity of this
implementation - pvlib's own ``fit_desoto`` carried a convergence failure of
exactly this kind (pvlib-python issue #1014), converging for a saturation
current of 1e-10 A and failing for 8e-10 A, with the solver reporting that
"the iteration is not making good progress".

The cause is that ``I0`` enters through ``exp(V/a)``, so the two are coupled
exponentially. Estimating ``a`` from the datasheet by the usual closed form,
which ignores both resistances, is typically 50 to 70 percent high; through
the exponential that becomes an initial ``I0`` wrong by **four orders of
magnitude**, far outside the basin the solver can recover from.

What actually fixes it
----------------------

A multi-start over the ideality factor. ``a = n * Ns * k * T / q`` with ``n``
between roughly 0.8 and 2 for any silicon module, so the one badly known
quantity lives on a short, physically bounded interval. Starting from each
``n`` in turn and keeping the first solve that converges to a physical answer
turns a fragile fit into a reliable one.

A note on what did *not* fix it: this module was first written in the belief
that the problem was scaling, and that solving for ``log(I0)`` and
``log(Rsh)`` would condition the Jacobian and settle it. Measured across the
test corpus, that belief is wrong. The logarithmic parameterisation is not
better than the linear one - on some modules it is worse, because an
unbounded ``log(Rsh)`` lets the solver escape to ``Rsh = 1e8`` for the price
of a small step and settle in a spurious minimum there. Both
parameterisations succeed from a good start and both fail from a bad one, so
the choice is kept as an option and the starting point is what is defended.
``tests/test_extract.py`` measures the convergence map that this paragraph
summarises; if it ever stops matching, this paragraph is what is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from .datasheet import Datasheet
from .model import DiodeParameters, residual, thermal_voltage, voltage_from_current
from .translate import REFERENCE_IRRADIANCE, celsius_to_kelvin, translate

__all__ = [
    "IDEALITY_GRID",
    "ExtractionError",
    "ExtractionResult",
    "extract",
    "initial_guess",
    "naive_ideality_estimate",
]

# Temperature step for the finite-difference dVoc/dT in equation 5. Small
# enough that the derivative is local, large enough that the difference of two
# solved open-circuit voltages is not dominated by solver tolerance.
_DELTA_T = 1.0

# Starting ideality factors, most likely first. Crystalline silicon sits near
# 1.0 to 1.3; the wider values are there for the modules that do not.
IDEALITY_GRID: tuple[float, ...] = (1.0, 1.2, 1.1, 1.4, 0.9, 1.6, 0.8, 1.8, 2.0)

_PARAMETERISATIONS = ("linear", "log")


class ExtractionError(RuntimeError):
    """No starting point produced a converged, physical fit."""


@dataclass(frozen=True)
class ExtractionResult:
    """What the solve produced, and enough detail to judge whether to trust it.

    Attributes:
        parameters: the five parameters at reference conditions.
        ideality_factor: ``n``, recovered from ``a``. The interpretable one:
            values outside roughly 0.5 to 2.5 mean the fit is physically
            doubtful even if it reproduces the datasheet exactly.
        residuals: the five equation residuals at the solution. Units are
            mixed - amperes, siemens, volts per kelvin - so judge them
            individually rather than by their norm.
        ideality_start: which entry of :data:`IDEALITY_GRID` succeeded.
        attempts: how many starting points were tried. Greater than one means
            the first guesses failed, which is information about the module.
        parameterisation: which variable change was used.
    """

    parameters: DiodeParameters
    ideality_factor: float
    residuals: np.ndarray
    ideality_start: float
    attempts: int
    parameterisation: str

    @property
    def max_abs_residual(self) -> float:
        """Largest equation residual, the single number to judge the fit by."""
        return float(np.max(np.abs(self.residuals)))


def naive_ideality_estimate(sheet: Datasheet) -> float:
    """The textbook closed form for ``a``, ignoring both resistances.

    From the two points ``(Voc, 0)`` and ``(Vmp, Imp)`` with ``IL ~ Isc``:

        a = (Vmp - Voc) / ln(1 - Imp/Isc)

    Exact in the resistance-free limit, and the reason a single-start fit
    fails: for a real module it overestimates ``a`` by half, and ``I0`` is
    exponential in ``a``. Kept because `tests/test_extract.py` uses it to
    demonstrate the failure this module is built to avoid.
    """
    return (sheet.v_mp - sheet.v_oc) / float(np.log1p(-sheet.i_mp / sheet.i_sc))


def initial_guess(
    sheet: Datasheet, ideality_factor: float
) -> tuple[float, float, float, float, float]:
    """A starting point for one assumed ideality factor.

    Everything else follows from it:

    * ``a`` from ``n``, the number of cells and the reference temperature;
    * ``I0`` from open circuit, where ``Isc ~ I0 * exp(Voc/a)``;
    * both resistances from ``Voc/Isc``, the module's characteristic
      impedance - a series resistance is a small fraction of it and a shunt a
      large multiple.

    Args:
        sheet: the module's published values.
        ideality_factor: assumed ``n``, the only quantity being guessed.

    Returns:
        ``(IL, I0, Rs, Rsh, a)``.
    """
    if ideality_factor <= 0.0:
        raise ValueError("ideality_factor must be > 0")

    a = thermal_voltage(
        ideality_factor,
        sheet.cells_in_series,
        celsius_to_kelvin(sheet.reference_temperature_c),
    )
    z = sheet.v_oc / sheet.i_sc
    return sheet.i_sc, sheet.i_sc * float(np.exp(-sheet.v_oc / a)), 0.1 * z, 100.0 * z, a


def _pack(
    values: tuple[float, float, float, float, float], parameterisation: str
) -> np.ndarray:
    il, i0, rs, rsh, a = values
    if parameterisation == "log":
        return np.array([il, np.log(i0), rs, np.log(rsh), np.log(a)])
    return np.array([il, i0, rs, rsh, a])


def _unpack(x: np.ndarray, parameterisation: str) -> tuple[float, ...]:
    if parameterisation == "log":
        return (
            float(x[0]),
            float(np.exp(x[1])),
            float(x[2]),
            float(np.exp(x[3])),
            float(np.exp(x[4])),
        )
    return tuple(float(v) for v in x)


def _slope_term(v: float, i: float, p: DiodeParameters) -> float:
    """``g = I0/a * exp((V + I*Rs)/a) + 1/Rsh``, the diode plus shunt conductance.

    The curve slope is ``dI/dV = -g / (1 + Rs*g)``, from implicit
    differentiation of the single diode equation.
    """
    vd = v + i * p.resistance_series
    return float(
        p.saturation_current / p.thermal_voltage * np.exp(vd / p.thermal_voltage)
        + 1.0 / p.resistance_shunt
    )


def _equations(x: np.ndarray, sheet: Datasheet, parameterisation: str) -> np.ndarray:
    """The five residuals. Zero at the answer."""
    il, i0, rs, rsh, a = _unpack(x, parameterisation)

    # The linear parameterisation can propose these; returning a large finite
    # residual steers the solver back rather than raising, which would abort a
    # search that might still recover.
    if i0 <= 0.0 or rsh <= 0.0 or a <= 0.0 or rs < 0.0:
        return np.full(5, 1e6)

    p = DiodeParameters(il, i0, rs, rsh, a)

    # A far-fetched trial point overflows the diode exponential. That is the
    # solver exploring, not a defect, so the arithmetic is allowed to run hot
    # and the non-finite result is converted below into a large residual that
    # steers it back.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r_sc = float(residual(sheet.i_sc, 0.0, p))
        r_oc = float(residual(0.0, sheet.v_oc, p))
        r_mp = float(residual(sheet.i_mp, sheet.v_mp, p))

        # dP/dV = 0 at the maximum power point  <=>  dI/dV = -Imp/Vmp.
        g = _slope_term(sheet.v_mp, sheet.i_mp, p)
        r_slope = g / (1.0 + rs * g) - sheet.i_mp / sheet.v_mp

    if not all(np.isfinite([r_sc, r_oc, r_mp, r_slope])):
        return np.full(5, 1e6)

    # dVoc/dT against the datasheet coefficient, by translating and re-solving.
    try:
        warm = translate(
            p,
            irradiance=REFERENCE_IRRADIANCE,
            temperature_c=sheet.reference_temperature_c + _DELTA_T,
            alpha_sc=sheet.alpha_sc,
            reference_temperature_c=sheet.reference_temperature_c,
            bandgap_ref_ev=sheet.bandgap_ev,
        )
        voc_warm = float(voltage_from_current(0.0, warm))
    except (OverflowError, ValueError, RuntimeError):
        return np.full(5, 1e6)

    r_beta = (voc_warm - sheet.v_oc) / _DELTA_T - sheet.beta_oc
    if not np.isfinite(r_beta):
        return np.full(5, 1e6)

    return np.array([r_sc, r_oc, r_mp, r_slope, r_beta])


def _solve_once(
    sheet: Datasheet, ideality_start: float, parameterisation: str, tolerance: float
) -> tuple[DiodeParameters, np.ndarray] | None:
    """One solve from one starting ideality factor. ``None`` if it failed."""
    x0 = _pack(initial_guess(sheet, ideality_start), parameterisation)

    # Levenberg-Marquardt rather than the default hybrid Powell method: a
    # damped least-squares step is the right tool for a system this badly
    # scaled, and the pvlib thread on the same fit reports 'hybr' stalling
    # where 'lm' converges.
    try:
        solution = root(
            _equations,
            x0,
            args=(sheet, parameterisation),
            method="lm",
            options={"maxiter": 3000, "xtol": 1e-14, "ftol": 1e-14},
        )
    except (OverflowError, ValueError, RuntimeError):
        return None

    if not solution.success:
        return None

    il, i0, rs, rsh, a = _unpack(solution.x, parameterisation)
    if i0 <= 0.0 or rsh <= 0.0 or a <= 0.0 or rs < 0.0:
        return None

    residuals = _equations(solution.x, sheet, parameterisation)
    if np.max(np.abs(residuals)) > tolerance:
        return None

    return DiodeParameters(il, i0, rs, rsh, a), residuals


def extract(
    sheet: Datasheet,
    *,
    parameterisation: str = "linear",
    tolerance: float = 1e-8,
    check_physical: bool = True,
    ideality_grid: tuple[float, ...] = IDEALITY_GRID,
) -> ExtractionResult:
    """Fit the five De Soto parameters to a datasheet.

    Tries each starting ideality factor in turn and returns the first solve
    that converges to a physical answer. On a well-behaved module the first
    start succeeds; a module needing several is telling you something.

    Args:
        sheet: the module's published values at reference conditions.
        parameterisation: ``"linear"`` (default) or ``"log"``. Measured across
            the test corpus these perform comparably; see the module
            docstring. The option exists so the claim stays falsifiable.
        tolerance: largest acceptable absolute residual.
        check_physical: reject a converged solution whose ideality factor or
            resistances are outside physically sensible bounds. A fit can
            reproduce all four datasheet points and still be nonsense.
        ideality_grid: starting ideality factors, tried in order.

    Returns:
        The parameters and the diagnostics needed to judge them.

    Raises:
        ValueError: on an unknown parameterisation or an empty grid.
        ExtractionError: if no starting point yields an acceptable fit.
    """
    if parameterisation not in _PARAMETERISATIONS:
        raise ValueError(
            f"unknown parameterisation {parameterisation!r}; "
            f"expected one of {_PARAMETERISATIONS}"
        )
    if not ideality_grid:
        raise ValueError("ideality_grid must contain at least one starting value")

    rejected: list[str] = []

    for attempt, n_start in enumerate(ideality_grid, start=1):
        outcome = _solve_once(sheet, n_start, parameterisation, tolerance)
        if outcome is None:
            continue

        params, residuals = outcome
        n = params.thermal_voltage * _ideality_scale(sheet)

        if check_physical:
            problem = _physical_complaint(n, params, sheet)
            if problem is not None:
                rejected.append(f"n0={n_start}: {problem}")
                continue

        return ExtractionResult(
            parameters=params,
            ideality_factor=n,
            residuals=residuals,
            ideality_start=n_start,
            attempts=attempt,
            parameterisation=parameterisation,
        )

    detail = ("; ".join(rejected)) if rejected else "no start converged"
    raise ExtractionError(
        f"could not fit {sheet.name!r} from any of {len(ideality_grid)} starting "
        f"ideality factors using the {parameterisation} parameterisation ({detail})"
    )


def _ideality_scale(sheet: Datasheet) -> float:
    """Factor turning ``a`` into the dimensionless ideality factor ``n``."""
    unit = thermal_voltage(
        1.0, sheet.cells_in_series, celsius_to_kelvin(sheet.reference_temperature_c)
    )
    return 1.0 / unit


def _physical_complaint(n: float, params: DiodeParameters, sheet: Datasheet) -> str | None:
    """Why this fit should be rejected, or ``None`` if it is acceptable."""
    if not 0.5 <= n <= 2.5:
        return (
            f"ideality factor {n:.3f} is outside the physical range 0.5 to 2.5; "
            "the fit reproduces the datasheet but the device it describes does not exist"
        )
    z = sheet.v_oc / sheet.i_sc
    if params.resistance_series > z:
        return (
            f"series resistance {params.resistance_series:.3g} ohm exceeds the "
            f"characteristic impedance {z:.3g} ohm"
        )
    if params.resistance_shunt < z:
        return (
            f"shunt resistance {params.resistance_shunt:.3g} ohm is below the "
            f"characteristic impedance {z:.3g} ohm"
        )
    return None
