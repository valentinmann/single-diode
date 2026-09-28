# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The overview figure drew the dotted power curves across the left panel's
  legend. The legend was attached to the primary axes and the curves to its
  twin, and a twinned axes is drawn on top of the one it was made from
  whatever the zorder or the frame opacity, so the labels were crossed by
  lines. It was only visible at the width the figure is actually rendered at.
  The legend now sits on the twin.

### Changed

- The README quickstart is a complete sequence, from `git clone` to the first
  figure, verified end to end in a fresh clone and a fresh virtual environment.
- The badges are absolute URLs, so they still resolve where the README is read
  outside github.com, and the CI badge reports `main` explicitly. The two
  static badges that only restated what the Development section says are gone.

## [0.1.0] - 2026-09-24

First release.

### Added

- `lambertw_exp`: `W(exp(x))` solved as `w + ln(w) = x`, so the Lambert W
  argument of the single diode closed form is never formed. Correct for
  exponents far past the `e^709` double precision ceiling.
- `model`: the single diode equation with three independent solvers
  (Lambert W, Brent, Newton), cross-checked against each other.
- `extract`: five De Soto parameters from four datasheet points and the
  open-circuit temperature coefficient, by multi-start over the ideality
  factor.
- `translate`: De Soto translation to other irradiance and cell temperature.
- `curves`: I-V and P-V curves, and the maximum power point as the root of
  `dP/dV` rather than the largest value on a grid.
- `mppt`: perturb and observe, and incremental conductance, run against the
  model.
- `datasheet`: published values validated at construction, with the Canadian
  Solar CS5P-220M as the worked example.
- 154 tests, including the docstring examples, at 97% line and branch
  coverage.
- CI across Python 3.10 to 3.13 on Linux, macOS and Windows, plus jobs that
  lint, type-check in strict mode, build and validate the distributions, and
  run the worked example end to end.
- Tooling to the Scientific Python development guidelines: `ruff` with an
  extended rule set, `mypy --strict`, `nox` as the task runner, `pre-commit`
  with `typos` and `zizmor`, and Dependabot for the CI actions.
