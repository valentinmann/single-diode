"""Two maximum power point trackers, run against the model.

A tracker never sees the curve. It sets an operating voltage, measures the
current there, and decides where to move next. Both algorithms below are the
textbook ones, and both are here mainly because they have a well-known failure
that a static test will not show.

Perturb and observe compares the power it just measured with the power it
measured before. If the irradiance changed in between, the difference it
attributes to its own voltage step actually came from the weather, and the
tracker can walk steadily the wrong way for as long as the ramp lasts. This is
the classic confusion failure, and `tests/test_mppt.py` provokes it
deliberately.

Incremental conductance is usually presented as the answer to it, because it
tests a property of the present curve, ``dI/dV = -I/V``, rather than comparing
two powers separated in time. That argument holds for the continuous
algorithm. It does not survive discretisation: any real controller estimates
``dI/dV`` from consecutive samples, and during a ramp those samples come from
different curves, so the same contamination enters by the same door. Measured
on the ramp in the tests, the two trackers follow identical trajectories.

Neither algorithm is improved here to hide that, and the textbook claim is not
repeated where the measurement contradicts it. The point of the package is
that the behaviour is measured rather than asserted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .curves import characteristic_points
from .model import DiodeParameters, current_from_voltage

__all__ = ["TrackerResult", "incremental_conductance", "perturb_and_observe"]


@dataclass
class TrackerResult:
    """The trajectory a tracker followed.

    Attributes:
        voltage: operating voltage at each step [V].
        power: measured power at each step [W].
        true_v_mp: the actual maximum power point voltage at each step [V],
            which the tracker does not know.
        true_p_mp: the actual maximum power at each step [W].
    """

    voltage: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    power: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    true_v_mp: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    true_p_mp: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))

    @property
    def final_voltage_error(self) -> float:
        """Distance from the true maximum at the last step [V]."""
        return float(abs(self.voltage[-1] - self.true_v_mp[-1]))

    @property
    def tracking_efficiency(self) -> float:
        """Energy captured over energy available, over the whole run."""
        return float(np.sum(self.power) / np.sum(self.true_p_mp))


def _measure(voltage: float, params: DiodeParameters) -> tuple[float, float]:
    """What a tracker can actually observe: the current and power it draws."""
    current = float(current_from_voltage(voltage, params))
    return current, voltage * current


def _prepare(
    conditions: DiodeParameters | Sequence[DiodeParameters], steps: int
) -> list[DiodeParameters]:
    """Accept either one fixed condition or one per step."""
    if isinstance(conditions, DiodeParameters):
        return [conditions] * steps
    sequence = list(conditions)
    if len(sequence) != steps:
        raise ValueError(
            f"got {len(sequence)} conditions for {steps} steps; pass one "
            "DiodeParameters for a constant condition or exactly one per step"
        )
    return sequence


def _truth(
    sequence: list[DiodeParameters],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The answer the tracker is being judged against, computed exactly."""
    points = [characteristic_points(p) for p in sequence]
    return (
        np.array([p.v_mp for p in points]),
        np.array([p.p_mp for p in points]),
    )


def perturb_and_observe(
    conditions: DiodeParameters | Sequence[DiodeParameters],
    *,
    steps: int = 200,
    step_voltage: float = 0.5,
    initial_voltage: float | None = None,
) -> TrackerResult:
    """Perturb and observe.

    Step the voltage; if the power rose, step again the same way; if it fell,
    reverse. In steady state the operating point oscillates around the maximum
    with an amplitude of one ``step_voltage``, which is the algorithm working
    correctly, not a defect.

    Args:
        conditions: one parameter set, or one per step for a changing condition.
        steps: number of control steps.
        step_voltage: perturbation size [V].
        initial_voltage: starting voltage [V]. Defaults to 80% of the first
            open-circuit voltage, the usual practical starting heuristic.

    Returns:
        The trajectory, alongside the true maximum at each step.
    """
    sequence = _prepare(conditions, steps)
    v = _default_start(sequence[0], initial_voltage)

    direction = 1.0
    previous_power = -np.inf
    voltages, powers = np.empty(steps), np.empty(steps)

    for k, params in enumerate(sequence):
        _, power = _measure(v, params)
        voltages[k], powers[k] = v, power

        # The whole algorithm, and the whole flaw: this comparison attributes
        # every change in power to the previous voltage step.
        if power < previous_power:
            direction = -direction
        previous_power = power
        v = float(np.clip(v + direction * step_voltage, 1e-6, None))

    true_v, true_p = _truth(sequence)
    return TrackerResult(voltages, powers, true_v, true_p)


def incremental_conductance(
    conditions: DiodeParameters | Sequence[DiodeParameters],
    *,
    steps: int = 200,
    step_voltage: float = 0.5,
    initial_voltage: float | None = None,
    tolerance: float = 1e-3,
) -> TrackerResult:
    """Incremental conductance.

    At the maximum power point ``dI/dV = -I/V``. The controller compares the
    incremental conductance it measures against the instantaneous one and
    moves towards equality. Because the test is a property of the present
    curve rather than a comparison of two powers separated in time, a change
    in irradiance does not send it the wrong way.

    Args:
        conditions: one parameter set, or one per step.
        steps: number of control steps.
        step_voltage: perturbation size [V].
        initial_voltage: starting voltage [V].
        tolerance: conductance mismatch treated as "at the maximum" [S].

    Returns:
        The trajectory, alongside the true maximum at each step.
    """
    sequence = _prepare(conditions, steps)
    v = _default_start(sequence[0], initial_voltage)

    previous_v, previous_i = v, None
    voltages, powers = np.empty(steps), np.empty(steps)

    for k, params in enumerate(sequence):
        current, power = _measure(v, params)
        voltages[k], powers[k] = v, power

        if previous_i is None:
            move = 1.0
        else:
            dv = v - previous_v
            di = current - previous_i
            if abs(dv) < 1e-12:
                # No voltage change: fall back on the instantaneous test.
                move = 0.0 if abs(di) < tolerance else np.sign(di)
            else:
                mismatch = di / dv + current / v
                move = 0.0 if abs(mismatch) < tolerance else np.sign(mismatch)

        previous_v, previous_i = v, current
        v = float(np.clip(v + move * step_voltage, 1e-6, None))

    true_v, true_p = _truth(sequence)
    return TrackerResult(voltages, powers, true_v, true_p)


def _default_start(params: DiodeParameters, initial_voltage: float | None) -> float:
    if initial_voltage is not None:
        if initial_voltage <= 0.0:
            raise ValueError("initial_voltage must be > 0")
        return initial_voltage
    return 0.8 * characteristic_points(params).v_oc
