# Contributing

This is a small research package maintained by one person. Issues and pull
requests are welcome; please open an issue before a large change so we can
agree on the shape of it first.

## Getting set up

```bash
git clone https://github.com/valentinmann/single-diode.git
cd single-diode
pip install -e ".[dev]"
pre-commit install
```

## Running the checks

Everything CI runs is available locally through [nox](https://nox.thea.codes):

```bash
nox              # lint, types and tests
nox -s tests     # tests on every installed interpreter
nox -s lint      # ruff check and ruff format --check
nox -s types     # mypy, strict mode
nox -s coverage  # tests with a coverage report
nox -s build     # build the distributions and validate the metadata
```

Or directly, if you prefer:

```bash
pytest
ruff check . && ruff format --check .
mypy
```

## What a change needs

- **A test that fails before it and passes after.** This package exists
  because its tests caught four separate defects, every one of which produced
  plausible output; that is the standard to hold.
- **Physics justified in the docstring, not just in the diff.** If a constant
  or a formulation comes from a paper, name the paper.
- **Claims that stay falsifiable.** Several docstrings state a measured
  result, such as which starting points make the parameter fit converge. If
  you change the behaviour, update the measurement and the test that pins it
  rather than the prose alone.
- `ruff`, `ruff format` and `mypy --strict` clean. `pre-commit` enforces all
  three before the commit lands.

## Reporting a problem

For a numerical result that looks wrong, the most useful report is the five
parameters, the operating condition, and what you expected instead. A failing
test case is better still.
