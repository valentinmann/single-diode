"""What a manufacturer publishes, as a validated object.

A module datasheet gives four operating points and two temperature
coefficients. That is the whole input to this package, and most of the ways it
goes wrong are unit slips rather than physics: a temperature coefficient in
%/K where the model wants A/K, a positive ``beta_oc`` where open-circuit
voltage in fact falls with temperature.

:class:`Datasheet` rejects those at construction, because a sign error in
``beta_oc`` produces a fit that converges, reproduces three of the four
datasheet points, and is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from .translate import BANDGAP_SILICON_EV, REFERENCE_TEMPERATURE_C

__all__ = ["CS5P_220M", "REFERENCE_MODULES", "Datasheet"]


@dataclass(frozen=True)
class Datasheet:
    """Published module values at reference conditions.

    Attributes:
        name: the module designation, used in errors and plot labels.
        i_sc: short-circuit current [A].
        v_oc: open-circuit voltage [V].
        i_mp: current at the maximum power point [A].
        v_mp: voltage at the maximum power point [V].
        alpha_sc: temperature coefficient of ``i_sc`` [A/K], absolute, not %/K.
            Positive: a warmer cell passes slightly more current.
        beta_oc: temperature coefficient of ``v_oc`` [V/K], absolute.
            Negative: a warmer cell loses voltage, and this dominates the
            temperature behaviour of the module.
        cells_in_series: ``Ns``.
        reference_temperature_c: the temperature the values describe [C].
        bandgap_ev: band gap at the reference temperature [eV]; the default is
            crystalline silicon.
        source: where the numbers came from. Required, so that a fixture can
            never quietly become an invention.
    """

    name: str
    i_sc: float
    v_oc: float
    i_mp: float
    v_mp: float
    alpha_sc: float
    beta_oc: float
    cells_in_series: int
    reference_temperature_c: float = REFERENCE_TEMPERATURE_C
    bandgap_ev: float = BANDGAP_SILICON_EV
    source: str = ""

    def __post_init__(self) -> None:
        """Reject published values that cannot belong to one real module."""
        for field, value in (
            ("i_sc", self.i_sc),
            ("v_oc", self.v_oc),
            ("i_mp", self.i_mp),
            ("v_mp", self.v_mp),
        ):
            if value <= 0.0:
                raise ValueError(f"{field} must be > 0, got {value}")

        if self.i_mp >= self.i_sc:
            raise ValueError(
                f"i_mp ({self.i_mp}) must be below i_sc ({self.i_sc}); "
                "the maximum power point lies inside the curve"
            )
        if self.v_mp >= self.v_oc:
            raise ValueError(
                f"v_mp ({self.v_mp}) must be below v_oc ({self.v_oc}); "
                "the maximum power point lies inside the curve"
            )
        if self.beta_oc >= 0.0:
            raise ValueError(
                f"beta_oc ({self.beta_oc}) must be negative: open-circuit voltage "
                "falls as a cell warms. A positive value usually means a %/K "
                "figure was passed where V/K was expected."
            )
        if self.alpha_sc <= 0.0:
            raise ValueError(
                f"alpha_sc ({self.alpha_sc}) must be positive: short-circuit "
                "current rises slightly as a cell warms."
            )
        if self.cells_in_series <= 0:
            raise ValueError("cells_in_series must be a positive integer")
        if not 0.4 <= self.fill_factor <= 0.9:
            raise ValueError(
                f"fill factor {self.fill_factor:.3f} is outside 0.4 to 0.9; "
                "check that the four operating points belong to the same module"
            )

    @property
    def p_mp(self) -> float:
        """Nameplate power [W]."""
        return self.i_mp * self.v_mp

    @property
    def fill_factor(self) -> float:
        """``Pmp / (Isc * Voc)``. Always below 1 for a real device."""
        return self.p_mp / (self.i_sc * self.v_oc)


# The canonical worked example in the PV modelling literature and in pvlib's
# own documentation. Values are the California Energy Commission module
# database entry, which is public. Internally consistent: 4.69 A * 46.9 V =
# 219.96 W, matching the published nameplate of 219.961 W.
CS5P_220M = Datasheet(
    name="Canadian Solar CS5P-220M",
    i_sc=5.10,
    v_oc=59.40,
    i_mp=4.69,
    v_mp=46.90,
    alpha_sc=0.004539,
    beta_oc=-0.222156,
    cells_in_series=96,
    source="California Energy Commission module database (mono-c-Si, 96 cells)",
)

REFERENCE_MODULES: tuple[Datasheet, ...] = (CS5P_220M,)
