"""The figure is code too, so it is tested like code.

Not the appearance - that is what looking at it is for - but the properties a
committed artefact has to have: it is written, it is not empty, it carries no
build-machine metadata, and regenerating it produces the same bytes so the
repository does not churn on every run.
"""

from __future__ import annotations

import hashlib
import struct
from itertools import pairwise

import numpy as np
import pytest

from single_diode.model import DiodeParameters
from single_diode.plotting import OVERFLOW_LIMIT, _log_theta, plot_overview


def test_overflow_limit_is_the_float64_ceiling():
    assert pytest.approx(float(np.log(np.finfo(np.float64).max))) == OVERFLOW_LIMIT
    with np.errstate(over="ignore"):
        assert np.isinf(np.exp(OVERFLOW_LIMIT * 1.001))
    assert np.isfinite(np.exp(OVERFLOW_LIMIT * 0.999))


def test_log_theta_rises_with_shunt_resistance(reference_params):
    """The panel's premise: Rsh is what drives the exponent."""

    def with_shunt(rsh: float) -> float:
        return _log_theta(
            DiodeParameters(
                reference_params.photocurrent,
                reference_params.saturation_current,
                reference_params.resistance_series,
                rsh,
                reference_params.thermal_voltage,
            )
        )

    values = [with_shunt(r) for r in (100.0, 300.0, 600.0, 1200.0)]
    assert all(b > a for a, b in pairwise(values))


def test_writes_a_figure(tmp_path, datasheet, fitted):
    path = plot_overview(datasheet, fitted, path=tmp_path / "overview.png")
    assert path.exists()
    # A blank canvas is a few kB; a real two-panel figure is far larger.
    assert path.stat().st_size > 20_000


def test_creates_missing_directories(tmp_path, datasheet, fitted):
    path = plot_overview(datasheet, fitted, path=tmp_path / "deep" / "er" / "fig.png")
    assert path.exists()


def test_is_byte_for_byte_reproducible(tmp_path, datasheet, fitted):
    """Otherwise the committed figure changes on every run and pollutes diffs."""
    first = plot_overview(datasheet, fitted, path=tmp_path / "a.png")
    second = plot_overview(datasheet, fitted, path=tmp_path / "b.png")
    digest = (
        hashlib.sha256(first.read_bytes()).hexdigest(),
        hashlib.sha256(second.read_bytes()).hexdigest(),
    )
    assert digest[0] == digest[1]


def _png_text_chunks(blob: bytes) -> dict[str, str]:
    r"""Parse a PNG and return its textual metadata.

    Searching the whole file for marker strings, which is what this test used
    to do, is wrong: the compressed pixel data is effectively random bytes, so
    a short marker turns up in it by chance. ``C:\\`` is three bytes, which in
    170 kB of image data has about a one percent chance of appearing on any
    given run - and since the rendering differs slightly between platforms,
    every platform rolls its own die. Metadata lives in the ``tEXt``, ``iTXt``
    and ``zTXt`` chunks, so that is what gets inspected.
    """
    chunks: dict[str, str] = {}
    offset = 8  # skip the PNG signature
    while offset < len(blob):
        (length,) = struct.unpack(">I", blob[offset : offset + 4])
        kind = blob[offset + 4 : offset + 8]
        data = blob[offset + 8 : offset + 8 + length]
        if kind in (b"tEXt", b"zTXt"):
            key, _, value = data.partition(b"\x00")
            chunks[key.decode("latin-1")] = value.decode("latin-1", "replace")
        elif kind == b"iTXt":
            key, _, rest = data.partition(b"\x00")
            chunks[key.decode("latin-1")] = rest.decode("utf-8", "replace")
        offset += 12 + length  # length + type + data + CRC
    return chunks


def test_carries_no_build_metadata(tmp_path, datasheet, fitted):
    """No version string, no timestamp, no path from the machine that ran it."""
    path = plot_overview(datasheet, fitted, path=tmp_path / "overview.png")
    text = _png_text_chunks(path.read_bytes())

    assert text == {"Software": "matplotlib"}, f"unexpected metadata in the PNG: {text}"
    assert "Creation Time" not in text
    joined = " ".join(text.values())
    for marker in ("/Users/", "/home/", "C:\\", "version"):
        assert marker not in joined, f"{marker!r} leaked into the PNG metadata"
