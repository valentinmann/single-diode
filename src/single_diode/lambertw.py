"""Lambert W of a huge argument, without ever forming the argument.

The single diode equation has a closed-form solution in terms of the Lambert W
function, and that closed form is unusable as written. Solving for voltage
requires

    W(theta),   theta = (Rsh * I0 / a) * exp(Rsh * (IL + I0 - I) / a)

and for a real silicon module the exponent is enormous. A 96-cell module with
Rsh = 1000 ohm, IL = 5 A and a = 2.4 V puts it at Rsh * IL / a = 2083, so
theta = e^2083. That is not a large float, it is `inf`: the double precision
ceiling is about e^709. Every downstream value becomes `nan`, and the failure
is silent unless something checks for it.

The fix is to never compute theta. Writing w = W(e^x) and using
w * e^w = e^x, take logs:

    w + ln(w) = x

which is a well conditioned scalar equation in w for every finite x, however
large. `lambertw_exp(x)` solves it. Nothing here overflows for x up to the
largest finite double, and the relative accuracy is at the level of the
floating point representation of w itself.

`tests/test_lambertw.py` pins this down, including a test that the naive
`scipy.special.lambertw(exp(x))` route really does return `inf` in the regime
this module is built for. The naive route is not wrong mathematically; it is
wrong in floating point, which is the only kind of wrong that matters here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["OMEGA", "lambertw_exp"]

# The omega constant: the solution of w + ln(w) = 0, i.e. W(1).
OMEGA = 0.5671432904097838

# exp() of anything below this underflows to exactly zero.
_LOG_TINY = -745.0

_MAX_ITER = 60


def _initial_guess(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """A starting point good enough that Halley converges in a few steps.

    Two regimes, split at ``x = 1`` where the large-``x`` asymptotic becomes
    exact (``W(e) = 1``):

    * ``x <= 1`` - ``w ~ e^x``. Exact as ``x`` falls, and never worse than a
      factor of 1.8 out at ``x = 1``, which Halley absorbs in one step.
    * ``x > 1``  - ``w ~ x - ln(x) + ln(x)/x``, the standard asymptotic.

    An earlier version started the middle range from the omega constant. That
    is a fine guess near ``x = 0`` and a disastrous one by ``x = -25``, where
    the true root is ``1e-11``: the damped iteration ran out of steps while
    still halving towards it and returned a silently wrong value. The
    monotonicity test in ``tests/test_lambertw.py`` is what caught it.
    """
    w = np.empty_like(x)

    low = x <= 1.0
    # exp() underflows to zero below _LOG_TINY; clip so the guess stays
    # strictly positive and ln(w) remains finite.
    w[low] = np.exp(np.maximum(x[low], _LOG_TINY))

    high = ~low
    if high.any():
        xh = x[high]
        lx = np.log(xh)
        w[high] = xh - lx + lx / xh

    return w


def lambertw_exp(x: ArrayLike) -> NDArray[np.float64]:
    """Return ``W(exp(x))``, the principal branch, without computing ``exp(x)``.

    Solves ``w + ln(w) = x`` by Halley iteration.

    Args:
        x: The logarithm of the Lambert W argument. Any finite float, including
            values far above ``log(np.finfo(float).max) ~ 709`` where the
            argument itself is not representable.

    Returns:
        ``w > 0`` with ``w * exp(w) == exp(x)`` to floating point accuracy.

    Raises:
        ValueError: if any input is nan or +inf. Both indicate that something
            upstream has already overflowed, and returning a plausible number
            for them would hide exactly the bug this module exists to prevent.

    Examples:
        >>> float(lambertw_exp(0.0))    # the omega constant
        0.5671432904097838
        >>> float(lambertw_exp(1.0))    # W(e) == 1
        1.0
        >>> round(float(lambertw_exp(2083.0)), 6)  # past the overflow ceiling
        2075.362109
    """
    xa = np.asarray(x, dtype=np.float64)
    scalar = xa.ndim == 0
    xa = np.atleast_1d(xa)

    if np.isnan(xa).any() or np.isposinf(xa).any():
        raise ValueError(
            "lambertw_exp received nan or +inf; the argument was probably "
            "computed as exp() of something that overflowed. Pass the "
            "logarithm instead - that is the point of this function."
        )

    w = _initial_guess(xa)

    # Below the underflow floor, w = exp(x) to full precision - there is
    # nothing left for the iteration to improve, and ln(w) of a zero or
    # denormal w would only manufacture infinities. Hold those fixed.
    frozen = xa <= _LOG_TINY
    active = ~frozen
    if not active.any():
        return w[0] if scalar else w

    # Halley's method on f(w) = w + ln(w) - x, which has
    #   f'  = 1 + 1/w
    #   f'' = -1/w^2
    # and is monotone increasing in w > 0, so the iteration cannot jump
    # branches. Cubic convergence puts every realistic input inside ~4 steps.
    #
    # Convergence is judged on the residual rather than on the step size: the
    # specification is that w + ln(w) equals x, and near the root consecutive
    # steps oscillate at the last bit without ever shrinking, so a step-based
    # test can spin until it runs out of iterations on an answer that is
    # already exact.
    tolerance = 4.0 * np.finfo(np.float64).eps
    wa = w[active]
    xact = xa[active]
    scale = np.maximum(1.0, np.abs(xact))

    for _ in range(_MAX_ITER):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            f = wa + np.log(wa) - xact
            if np.all(np.abs(f) <= tolerance * scale):
                break
            fp = 1.0 + 1.0 / wa
            fpp = -1.0 / (wa * wa)
            step = 2.0 * f * fp / (2.0 * fp * fp - f * fpp)

        step = np.where(np.isfinite(step), step, 0.0)
        w_next = wa - step
        # The root is strictly positive. A step that overshoots into w <= 0 is
        # a sign of a bad guess, not of a root there, so damp instead.
        wa = np.where(w_next > 0.0, w_next, wa / 2.0)
    else:
        # Returning an unconverged value is exactly the silent failure this
        # module exists to prevent, so say so instead.
        residuals = np.abs(wa + np.log(wa) - xact)
        worst = int(np.argmax(residuals))
        raise RuntimeError(
            f"lambertw_exp did not converge in {_MAX_ITER} iterations; worst "
            f"input x={xact[worst]:.6g} left a residual of {residuals[worst]:.3g}"
        )

    w[active] = wa
    return w[0] if scalar else w
