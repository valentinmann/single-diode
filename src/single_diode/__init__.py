"""Single diode photovoltaic modelling, from a datasheet to an I-V curve.

Four published numbers and two temperature coefficients go in; the five
parameters of the single diode equivalent circuit come out, and with them the
module's behaviour at any irradiance and cell temperature.

    >>> from single_diode import CS5P_220M, extract, characteristic_points
    >>> result = extract(CS5P_220M)
    >>> point = characteristic_points(result.parameters)
    >>> round(point.p_mp, 2)
    219.96

The two places where care was needed, and where the tests are pointed:

* :mod:`single_diode.lambertw` - the closed-form solution needs the Lambert W
  of ``e`` to a power in the thousands, which no float can hold.
* :mod:`single_diode.extract` - the five unknowns span twelve orders of
  magnitude, so the fit is solved in a logarithmic parameterisation.
"""

from .curves import (
    CharacteristicPoints,
    IVCurve,
    characteristic_points,
    di_dv,
    iv_curve,
)
from .datasheet import CS5P_220M, REFERENCE_MODULES, Datasheet
from .extract import ExtractionError, ExtractionResult, extract, initial_guess
from .lambertw import lambertw_exp
from .model import (
    DiodeParameters,
    current_from_voltage,
    residual,
    thermal_voltage,
    voltage_from_current,
)
from .mppt import TrackerResult, incremental_conductance, perturb_and_observe
from .translate import bandgap, celsius_to_kelvin, translate

__version__ = "0.1.0"

__all__ = [
    "CS5P_220M",
    "REFERENCE_MODULES",
    "CharacteristicPoints",
    "Datasheet",
    "DiodeParameters",
    "ExtractionError",
    "ExtractionResult",
    "IVCurve",
    "TrackerResult",
    "__version__",
    "bandgap",
    "celsius_to_kelvin",
    "characteristic_points",
    "current_from_voltage",
    "di_dv",
    "extract",
    "incremental_conductance",
    "initial_guess",
    "iv_curve",
    "lambertw_exp",
    "perturb_and_observe",
    "residual",
    "thermal_voltage",
    "translate",
    "voltage_from_current",
]
