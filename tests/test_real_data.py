"""Checks against the real files rather than against synthetic data.

Marked `needs_data`, so the suite still runs on a clean clone without the download:

    pytest -m "not needs_data"

These are the ones that catch a reader which works fine on made up data and then says
something absurd about an actual grid recording.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from enf import matcher as mt
from enf.io import pgf

pytestmark = pytest.mark.needs_data

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "manifest" / "grids.yaml"


@pytest.fixture(scope="module")
def registry() -> dict:
    if not REGISTRY.exists():
        pytest.skip("grid registry missing")
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["sites"]


def longest_clean(code: str, meta: dict) -> np.ndarray:
    try:
        df = pgf.load_site(code, nominal_hz=meta.get("nominal_hz"))
    except (FileNotFoundError, pgf.DataIntegrityError) as exc:
        pytest.skip(f"{code}: {exc}")
    t = pgf.epoch_seconds(df.index)
    qi = df["qi"].to_numpy()
    f = df["f_mhz"].to_numpy()
    segs = pgf.clean_segments(t, qi, f=f, max_qi=0, max_bad_run_s=pgf.MAX_BAD_RUN_S)
    if not segs:
        pytest.skip(f"{code}: no clean segment")
    a, b = max(segs, key=lambda s: s[1] - s[0])
    return f[a:b].astype(np.float64)


# Published per site standard deviations in mHz, from the database statistics. Getting
# these back independently is what shows the reader has the units, the separator, the
# column choice and the value scale all right at once.
PUBLISHED_SIGMA_MHZ = {
    "FO01": 152.4,
    "ZA02": 118.2,
    "ZA01": 94.8,
    "GB01": 67.8,
    "GB02": 64.8,
    "ES_GC01": 62.1,
    "ES_PM01": 56.1,
    "IS01": 53.3,
    "SE01": 47.9,
    "MY01": 31.9,
    "EE01": 13.0,
}


@pytest.mark.parametrize("code,expected", sorted(PUBLISHED_SIGMA_MHZ.items()))
def test_site_sigma_matches_published_value(code, expected, registry):
    """Whole record standard deviation over good samples, against the published table."""
    try:
        df = pgf.load_site(code, nominal_hz=registry[code].get("nominal_hz"))
    except (FileNotFoundError, pgf.DataIntegrityError) as exc:
        pytest.skip(f"{code}: {exc}")
    f = df["f_mhz"].to_numpy()
    qi = df["qi"].to_numpy()
    got = float(np.nanstd(f[qi == 0]))
    assert got == pytest.approx(expected, rel=0.03), f"{code}: {got:.1f} vs published {expected}"


def test_deviations_are_physically_plausible_everywhere(registry):
    """No grid is ever more than a few Hz off nominal. Catch all for a misread value
    scale, which is the one mistake that gives you confident nonsense instead of a crash.
    """
    checked = 0
    for code, meta in sorted(registry.items()):
        if meta.get("exclude_from_primary"):
            continue
        try:
            df = pgf.load_site(code, nominal_hz=meta.get("nominal_hz"))
        except (FileNotFoundError, pgf.DataIntegrityError):
            continue
        f = df["f_mhz"].to_numpy()
        qi = df["qi"].to_numpy()
        good = f[qi == 0]
        if good.size == 0:
            continue
        assert np.nanpercentile(np.abs(good), 99.99) < 5000.0, code
        checked += 1
    assert checked >= 20, f"only {checked} sites loaded"


@pytest.mark.parametrize("code", ["EE01", "PT01", "KR01", "ZA02", "MY01"])
def test_noiseless_match_is_exact_on_real_data(code, registry):
    """The zero noise guarantee on actual measurements, not on simulated series.

    A real trace has held stretches, quantisation and heavy tails that a simulation does
    not, so this is a different test from the synthetic one even though it looks the same.
    """
    x = longest_clean(code, registry[code])
    if x.size < 20_000:
        pytest.skip(f"{code}: only {x.size} clean samples")
    ref = x[: min(x.size, 86_400)]
    cache = mt.build_ref_cache(ref)
    k0 = ref.size // 3
    res = mt.match(ref[k0 : k0 + 600].copy(), cache, metric="pearson", k_true=k0)
    assert res.k_hat == k0
    assert res.score == pytest.approx(1.0, abs=1e-9)


def test_no_window_contains_a_nan(registry):
    """The quality flag does not mean what you would assume.

    One site carries 2.57 million non finite samples, all of them sitting at a non zero
    quality flag, so tolerating a short flagged run there drops a NaN straight into the
    data. Finiteness is a hard requirement that the run length tolerance cannot override.
    """
    checked = 0
    for code, meta in sorted(registry.items()):
        if meta.get("exclude_from_primary"):
            continue
        try:
            _t, f, _qi = pgf.load_series(code)
        except (FileNotFoundError, pgf.DataIntegrityError):
            continue
        for w in pgf.select_reference_windows(code, length_s=86_400, k=4):
            seg = f[w.i0 : w.i1]
            n_bad = int(np.count_nonzero(~np.isfinite(seg)))
            assert n_bad == 0, f"{code} window {w.window_id}: {n_bad} non finite"
            mt.build_ref_cache(seg.astype(np.float64))
            checked += 1
    assert checked >= 20, f"only {checked} windows checked"
