"""The figure the example writes.

Two panels, because the package makes two separate claims and they need
different evidence.

Left: the fitted model reproduces the datasheet it was given, and behaves
correctly away from it. The four published points sit on the curve; the other
curves show what happens when the cell warms or the sun goes in.

Right: why any of the numerical care was necessary. The closed form needs
``W(theta)`` with ``log(theta) ~ Rsh * IL / a``, and double precision stops at
``e^709``. Whether that overflows depends on the module: across the test
corpus the exponent runs from 286 to 2757, so some modules are safe and some
are not. The panel sweeps shunt resistance, the parameter that drives it, and
marks where the fitted module falls - which for this one is within a percent
of the ceiling. A failure mode you cannot predict from a datasheet, and which
returns ``inf`` rather than an error, is worth removing rather than risking.

Deterministic: fixed colours, fixed size, PNG metadata pinned so no version
string or timestamp is written. Regenerating on one machine reproduces the
file byte for byte.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .curves import characteristic_points, iv_curve
from .datasheet import Datasheet
from .model import DiodeParameters
from .translate import REFERENCE_IRRADIANCE, translate

__all__ = ["OVERFLOW_LIMIT", "plot_overview"]

# log(largest finite float64) ~ 709.78: above this, exp() is inf.
OVERFLOW_LIMIT = float(np.log(np.finfo(np.float64).max))

COLOR_STC = "#0B6E4F"
COLOR_HOT = "#C1462F"
COLOR_DIM = "#3B6EA5"
COLOR_DATASHEET = "#1A1A1A"
COLOR_BAD = "#C1462F"
COLOR_GOOD = "#0B6E4F"

PNG_METADATA = {"Software": "matplotlib"}


def _log_theta(params: DiodeParameters) -> float:
    """``log`` of the Lambert W argument at open circuit, the worst case."""
    return float(
        np.log(params.resistance_shunt * params.saturation_current / params.thermal_voltage)
        + params.resistance_shunt
        * (params.photocurrent + params.saturation_current)
        / params.thermal_voltage
    )


def plot_overview(
    sheet: Datasheet,
    reference: DiodeParameters,
    *,
    path: Path,
) -> Path:
    """Write the two-panel overview figure. Returns the path written."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9))

    # ---------------- left: curves against the datasheet ----------------
    conditions = [
        (REFERENCE_IRRADIANCE, 25.0, COLOR_STC, "1000 W/m$^2$, 25 $\\degree$C (STC)"),
        (REFERENCE_IRRADIANCE, 60.0, COLOR_HOT, "1000 W/m$^2$, 60 $\\degree$C"),
        (400.0, 25.0, COLOR_DIM, "400 W/m$^2$, 25 $\\degree$C"),
    ]

    axp = ax.twinx()
    for irradiance, temperature, color, label in conditions:
        params = (
            reference
            if (irradiance, temperature) == (REFERENCE_IRRADIANCE, 25.0)
            else translate(
                reference,
                irradiance=irradiance,
                temperature_c=temperature,
                alpha_sc=sheet.alpha_sc,
                bandgap_ref_ev=sheet.bandgap_ev,
            )
        )
        curve = iv_curve(params, num_points=400)
        ax.plot(curve.voltage, curve.current, color=color, linewidth=2.0, label=label)
        axp.plot(
            curve.voltage,
            curve.power,
            color=color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.85,
        )
        point = characteristic_points(params)
        ax.plot(
            point.v_mp,
            point.i_mp,
            "o",
            color=color,
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
        )

    # The published values, which the fit was given and must reproduce.
    ax.plot(
        [0.0, sheet.v_mp, sheet.v_oc],
        [sheet.i_sc, sheet.i_mp, 0.0],
        "P",
        color=COLOR_DATASHEET,
        markersize=8,
        linestyle="none",
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=6,
        label="published datasheet points",
    )

    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current [A]")
    axp.set_ylabel("Power [W]   (dotted)", color="#555555")
    axp.tick_params(axis="y", colors="#555555")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    axp.set_ylim(bottom=0)
    ax.grid(color="#EDEDED", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    ax.set_title(f"{sheet.name}: fitted model vs datasheet", fontsize=11, loc="left")

    # ---------------- right: the overflow ceiling ----------------
    # Shunt resistance is what drives the exponent (log(theta) ~ Rsh*IL/a) and
    # is also what varies most between modules, so sweeping it shows where a
    # given device falls relative to the ceiling.
    shunts = np.linspace(40.0, 1200.0, 300)
    exponents = np.array(
        [
            _log_theta(
                DiodeParameters(
                    photocurrent=reference.photocurrent,
                    saturation_current=reference.saturation_current,
                    resistance_series=reference.resistance_series,
                    resistance_shunt=float(rsh),
                    thermal_voltage=reference.thermal_voltage,
                )
            )
            for rsh in shunts
        ]
    )

    top = max(exponents.max(), OVERFLOW_LIMIT) * 1.08
    ax2.axhspan(OVERFLOW_LIMIT, top, color="#FBEAE6", zorder=0)
    ax2.plot(
        shunts,
        exponents,
        color=COLOR_BAD,
        linewidth=2.2,
        zorder=3,
        label="exponent needed by the closed form",
    )
    ax2.axhline(
        OVERFLOW_LIMIT, color=COLOR_DATASHEET, linewidth=1.2, linestyle="--", zorder=4
    )
    ax2.text(
        shunts[-1],
        OVERFLOW_LIMIT,
        "  float64 ceiling, $e^{709}$  ",
        va="bottom",
        ha="right",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

    here = _log_theta(reference)
    ax2.plot(
        [reference.resistance_shunt],
        [here],
        "o",
        color=COLOR_DATASHEET,
        markersize=7,
        zorder=5,
        markeredgecolor="white",
        markeredgewidth=0.9,
        label=f"this module ($R_{{sh}}$ = {reference.resistance_shunt:.0f} $\\Omega$)",
    )
    ax2.annotate(
        f"{here:.0f}, or {100 * (1 - here / OVERFLOW_LIMIT):.0f}% below the ceiling",
        xy=(reference.resistance_shunt, here),
        xytext=(0.30, 0.32),
        textcoords="axes fraction",
        fontsize=8.5,
        color="#333333",
        arrowprops={"arrowstyle": "-", "color": "#AAAAAA", "linewidth": 0.9},
    )

    ax2.set_xlabel("Shunt resistance [$\\Omega$]")
    ax2.set_ylabel("$\\log(\\theta)$ at open circuit")
    ax2.set_ylim(0, top)
    ax2.set_xlim(shunts[0], shunts[-1])
    ax2.grid(color="#EDEDED", linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax2.set_title(
        "Whether the naive form overflows depends on the module", fontsize=11, loc="left"
    )
    ax2.text(
        0.97,
        0.06,
        "shaded: $\\exp(\\theta)$ is $\\it{inf}$\n"
        "solved here via $w + \\ln w = \\log\\theta$",
        transform=ax2.transAxes,
        fontsize=8.5,
        color="#7A3B2E",
        ha="right",
        va="bottom",
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata=PNG_METADATA)
    plt.close(fig)
    return path
