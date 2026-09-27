"""From a published datasheet to a working module model, end to end.

    python examples/datasheet_to_iv.py

Fits the five single diode parameters to a real module's published values,
checks the fit against those values, shows what the module does when it is hot
or shaded, runs both maximum power point trackers, and writes the figure used
in the README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from single_diode import (
    CS5P_220M,
    characteristic_points,
    extract,
    incremental_conductance,
    perturb_and_observe,
    translate,
)
from single_diode.extract import naive_ideality_estimate
from single_diode.model import thermal_voltage
from single_diode.plotting import plot_overview
from single_diode.translate import celsius_to_kelvin

DEFAULT_FIGURE = Path(__file__).resolve().parents[1] / "docs" / "overview.png"

CONDITIONS = [
    (1000.0, 25.0, "standard test conditions"),
    (1000.0, 60.0, "hot roof"),
    (800.0, 45.0, "typical summer afternoon"),
    (400.0, 25.0, "overcast"),
    (100.0, 15.0, "deep shade"),
]


def main() -> int:
    """Run the worked example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", type=Path, default=DEFAULT_FIGURE, help="where to write the figure"
    )
    parser.add_argument("--no-figure", action="store_true", help="skip the figure")
    args = parser.parse_args()

    sheet = CS5P_220M
    print(f"Module: {sheet.name}")
    print(f"  source: {sheet.source}")
    print(
        f"  published: Isc {sheet.i_sc} A, Voc {sheet.v_oc} V, "
        f"Imp {sheet.i_mp} A, Vmp {sheet.v_mp} V  ->  {sheet.p_mp:.1f} W, "
        f"fill factor {sheet.fill_factor:.3f}"
    )

    # ---- 1. fit -----------------------------------------------------------
    print("\n[1/4] Fitting five parameters to four published points")
    unit = thermal_voltage(
        1.0, sheet.cells_in_series, celsius_to_kelvin(sheet.reference_temperature_c)
    )
    naive_n = naive_ideality_estimate(sheet) / unit
    print(f"  textbook ideality estimate: n = {naive_n:.3f}  (ignores both resistances)")

    result = extract(sheet)
    params = result.parameters
    print(
        f"  converged from n = {result.ideality_start} "
        f"on attempt {result.attempts} of the grid"
    )
    print(f"  IL  = {params.photocurrent:.4f} A")
    print(f"  I0  = {params.saturation_current:.4g} A")
    print(f"  Rs  = {params.resistance_series:.4f} ohm")
    print(f"  Rsh = {params.resistance_shunt:.4g} ohm")
    print(f"  a   = {params.thermal_voltage:.4f} V   (n = {result.ideality_factor:.4f})")
    print(f"  largest equation residual: {result.max_abs_residual:.2e}")

    # ---- 2. check against the datasheet -----------------------------------
    print("\n[2/4] Reproducing the published values")
    point = characteristic_points(params)
    for label, got, published in (
        ("Isc [A]", point.i_sc, sheet.i_sc),
        ("Voc [V]", point.v_oc, sheet.v_oc),
        ("Imp [A]", point.i_mp, sheet.i_mp),
        ("Vmp [V]", point.v_mp, sheet.v_mp),
        ("Pmp [W]", point.p_mp, sheet.p_mp),
    ):
        error = abs(got - published) / published
        print(
            f"  {label:<9} model {got:9.4f}   published {published:9.4f}   "
            f"error {error * 100:7.4f} %"
        )

    # ---- 3. away from the datasheet ---------------------------------------
    print("\n[3/4] Behaviour at other conditions")
    print(
        f"  {'irradiance':>10} {'cell T':>7} {'Isc [A]':>9} {'Voc [V]':>9} "
        f"{'Pmp [W]':>9} {'FF':>7}   condition"
    )
    for irradiance, temperature, label in CONDITIONS:
        moved = translate(
            params,
            irradiance=irradiance,
            temperature_c=temperature,
            alpha_sc=sheet.alpha_sc,
            bandgap_ref_ev=sheet.bandgap_ev,
        )
        p = characteristic_points(moved)
        print(
            f"  {irradiance:>7.0f} W/m2 {temperature:>5.0f} C {p.i_sc:>9.4f} "
            f"{p.v_oc:>9.3f} {p.p_mp:>9.3f} {p.fill_factor:>7.4f}   {label}"
        )

    # ---- 4. tracking ------------------------------------------------------
    print("\n[4/4] Maximum power point tracking, 300 steps of 0.2 V")
    for name, tracker in (
        ("perturb and observe", perturb_and_observe),
        ("incremental conductance", incremental_conductance),
    ):
        run = tracker(params, steps=300, step_voltage=0.2)
        print(
            f"  {name:<24} final error {run.final_voltage_error:6.3f} V   "
            f"tracking efficiency {run.tracking_efficiency * 100:.3f} %"
        )

    if not args.no_figure:
        written = plot_overview(sheet, params, path=args.figure)
        print(f"\nwrote {written.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
