"""Correctness of the matcher.

A fast matcher that has never been checked against a slow one is a fast matcher nobody has
checked. The single most important test here is `test_zero_noise_is_exact`: if you plant a
query at a known offset with no noise added, the matcher must return that exact offset with
a correlation of 1. Any bug large enough to matter breaks it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.stats import pearsonr

from enf import matcher as mt


def ar1(n: int, a: float = 0.99, sigma: float = 1.0, *, seed: int = 0) -> np.ndarray:
    """A grid-frequency-like series: mean-reverting, strongly autocorrelated."""
    rng = np.random.default_rng(seed)
    innov = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = innov[0] / math.sqrt(max(1.0 - a * a, 1e-12))
    for i in range(1, n):
        x[i] = a * x[i - 1] + innov[i]
    return x


# --------------------------------------------------------------------------------------
# 1. the test that matters most
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_zero_noise_is_exact(seed):
    """Plant a clean query at a known offset; demand the exact offset and rho == 1."""
    rng = np.random.default_rng(seed)
    n_ref = int(rng.integers(3_000, 20_000))
    n_q = int(rng.integers(20, 500))
    k0 = int(rng.integers(0, n_ref - n_q))

    ref = ar1(n_ref, seed=seed)
    cache = mt.build_ref_cache(ref)
    q = ref[k0 : k0 + n_q].copy()

    res = mt.match(q, cache, metric="pearson", k_true=k0)
    assert res.k_hat == k0, f"found {res.k_hat}, planted {k0}"
    assert res.score == pytest.approx(1.0, abs=1e-12)
    assert res.rank_true == 0

    sse = mt.match(q, cache, metric="sse", k_true=k0)
    assert sse.k_hat == k0
    assert sse.score == pytest.approx(0.0, abs=1e-9)


def test_zero_noise_exact_on_a_long_reference():
    """Same guarantee at a realistic scope, where prefix-sum precision starts to bite."""
    n_ref, n_q, k0 = 604_800, 1800, 391_237
    ref = ar1(n_ref, seed=7)
    cache = mt.build_ref_cache(ref)
    res = mt.match(ref[k0 : k0 + n_q].copy(), cache, metric="pearson", k_true=k0)
    assert res.k_hat == k0
    assert res.score == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# 2. fast == slow
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_fast_matches_slow_reference_implementation(seed):
    rng = np.random.default_rng(1000 + seed)
    n_ref = int(rng.integers(500, 4000))
    n_q = int(rng.integers(5, 200))
    ref = ar1(n_ref, seed=seed)
    q = rng.normal(0, 1, n_q)

    fast = mt.match_scores(q, mt.build_ref_cache(ref), metric="both")
    slow = mt.match_scores_slow(q, ref, metric="both")

    np.testing.assert_allclose(fast.pearson, slow.pearson, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(fast.sse, slow.sse, rtol=1e-9, atol=1e-9)


def test_fast_matches_slow_when_query_comes_from_the_reference():
    """The in-distribution case: a query that really is a slice of the reference."""
    ref = ar1(3000, seed=11)
    q = ref[1234:1434] + np.random.default_rng(0).normal(0, 0.05, 200)
    fast = mt.match_scores(q, mt.build_ref_cache(ref), metric="both")
    slow = mt.match_scores_slow(q, ref, metric="both")
    np.testing.assert_allclose(fast.pearson, slow.pearson, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(fast.sse, slow.sse, rtol=1e-9, atol=1e-9)


def test_pearson_agrees_with_scipy_at_random_offsets():
    """Independent third opinion, from a library that knows nothing about this code."""
    rng = np.random.default_rng(3)
    ref = ar1(5000, seed=3)
    n_q = 120
    q = rng.normal(0, 1, n_q)
    scores = mt.match_scores(q, mt.build_ref_cache(ref), metric="pearson")
    for k in rng.integers(0, ref.size - n_q, 60):
        expected = pearsonr(q, ref[k : k + n_q])[0]
        assert scores.pearson[int(k)] == pytest.approx(expected, rel=1e-10, abs=1e-12)


# --------------------------------------------------------------------------------------
# 3. the two metrics really are different tests
# --------------------------------------------------------------------------------------


def test_pearson_is_affine_invariant_and_sse_is_not():
    """Why both criteria are worth computing: Pearson ignores a gain and offset on the
    query, squared error does not. They are not interchangeable and this pins that down.
    """
    ref = ar1(4000, seed=5)
    cache = mt.build_ref_cache(ref)
    q = ref[700:900].copy()

    base = mt.match_scores(q, cache, metric="both")
    scaled = mt.match_scores(3.0 * q + 7.0, cache, metric="both")

    np.testing.assert_allclose(base.pearson, scaled.pearson, rtol=1e-10, atol=1e-12)
    assert not np.allclose(base.sse, scaled.sse)


def test_sse_recovers_the_true_offset_despite_being_scale_sensitive():
    ref = ar1(4000, seed=6)
    cache = mt.build_ref_cache(ref)
    k0 = 1500
    res = mt.match(ref[k0 : k0 + 300].copy(), cache, metric="sse", k_true=k0)
    assert res.k_hat == k0


# --------------------------------------------------------------------------------------
# 4. numerics at length
# --------------------------------------------------------------------------------------


def test_sliding_sums_match_exact_summation_on_a_long_series():
    """Prefix sums over tens of millions of terms are where a naive cumsum fails. Checked
    against `math.fsum`, which is exact."""
    n = 10_000_000
    rng = np.random.default_rng(21)
    # millihertz-scale values with a large offset: the cancellation-prone case
    x = rng.normal(50_000.0, 20.0, n)
    cache = mt.build_ref_cache(x)
    n_q = 3600
    mean, std, _ = mt.sliding_mean_std(cache, n_q)

    for k in rng.integers(0, n - n_q, 40):
        k = int(k)
        w = cache.rc[k : k + n_q]
        exact_mean = math.fsum(w) / n_q
        exact_var = math.fsum((w - exact_mean) ** 2) / n_q
        assert mean[k] == pytest.approx(exact_mean, rel=1e-9, abs=1e-9)
        assert std[k] == pytest.approx(math.sqrt(exact_var), rel=1e-9)


def test_blocked_cumsum_agrees_with_fsum():
    rng = np.random.default_rng(22)
    x = rng.normal(1000.0, 1.0, 2_000_000)
    cs = mt._blocked_cumsum(x, block=1 << 16)
    for idx in (1, 1000, 999_999, x.size):
        assert cs[idx] == pytest.approx(math.fsum(x[:idx]), rel=1e-12)


# --------------------------------------------------------------------------------------
# 5. degenerate inputs, which real deadband grids produce
# --------------------------------------------------------------------------------------


def test_constant_query_scores_zero_rather_than_nan():
    """A short query inside a control deadband can be exactly constant. It must score 0
    rather than NaN, or the result quietly vanishes instead of being wrong."""
    ref = ar1(2000, seed=9)
    scores = mt.match_scores(np.full(50, 3.0), mt.build_ref_cache(ref), metric="pearson")
    assert np.all(np.isfinite(scores.pearson))
    assert np.allclose(scores.pearson, 0.0)


def test_constant_reference_windows_are_counted_not_hidden():
    """A held reference stretch is physically meaningful on a deadband grid, so the count
    of degenerate windows is reported."""
    ref = ar1(5000, seed=10)
    ref[1000:2000] = 0.0  # a held stretch
    scores = mt.match_scores(ref[3000:3100].copy(), mt.build_ref_cache(ref), metric="pearson")
    assert scores.n_degenerate > 0
    assert np.all(np.isfinite(scores.pearson))


def test_rejects_non_finite_reference():
    bad = ar1(1000, seed=1)
    bad[500] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mt.build_ref_cache(bad)


@pytest.mark.parametrize("n_q", [0, 1, 10_001])
def test_rejects_out_of_range_query_lengths(n_q):
    cache = mt.build_ref_cache(ar1(10_000, seed=2))
    with pytest.raises(ValueError):
        mt.match_scores(np.zeros(max(n_q, 0)), cache, metric="pearson")


# --------------------------------------------------------------------------------------
# 6. property tests
# --------------------------------------------------------------------------------------


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_ref=st.integers(min_value=200, max_value=3000),
    frac=st.floats(min_value=0.01, max_value=0.5),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_scores_are_well_formed(n_ref, frac, seed):
    n_q = max(2, int(n_ref * frac))
    rng = np.random.default_rng(seed)
    ref = rng.normal(0, 1, n_ref)
    q = rng.normal(0, 1, n_q)
    sc = mt.match_scores(q, mt.build_ref_cache(ref), metric="both")

    assert sc.pearson.size == n_ref - n_q + 1
    assert sc.sse.size == n_ref - n_q + 1
    assert np.all(np.isfinite(sc.pearson))
    assert np.all(sc.pearson >= -1.0 - 1e-9)
    assert np.all(sc.pearson <= 1.0 + 1e-9)
    assert np.all(sc.sse >= -1e-9)


# --------------------------------------------------------------------------------------
# 7. the matrix-profile identity
# --------------------------------------------------------------------------------------


def test_znorm_ed_identity_round_trips():
    rho = np.linspace(-1.0, 1.0, 101)
    n = 256
    np.testing.assert_allclose(
        mt.pearson_from_znorm_ed(mt.znorm_ed_from_pearson(rho, n), n), rho, atol=1e-12
    )


def test_znorm_ed_matches_stumpy_matrix_profile():
    """The bridge to the model-free uniqueness measurement.

    `stumpy` computes, for each window, the z-normalised Euclidean distance to its nearest
    non-trivial neighbour. Converting our Pearson landscape with the identity must give the
    same number. If this passes, the matrix profile and the matcher are measuring one
    object and can be used to cross-check each other.
    """
    stumpy = pytest.importorskip("stumpy")
    m = 64
    x = ar1(3000, a=0.95, seed=13)
    profile = stumpy.stump(x, m=m)
    mp_dist = profile[:, 0].astype(float)

    cache = mt.build_ref_cache(x)
    excl = int(np.ceil(m / 4))  # stumpy's default trivial-match exclusion zone

    for i in (100, 500, 1200, 2000):
        rho = mt.match_scores(x[i : i + m], cache, metric="pearson").pearson
        dist = mt.znorm_ed_from_pearson(rho, m)
        lo, hi = max(i - excl, 0), min(i + excl + 1, dist.size)
        dist[lo:hi] = np.inf
        assert float(np.min(dist)) == pytest.approx(mp_dist[i], rel=1e-5)


# --------------------------------------------------------------------------------------
# 8. the failure-mode instrumentation
# --------------------------------------------------------------------------------------


def test_margin_and_gap_distinguish_the_two_failure_modes():
    """`z_margin` and `gap2` must separate alias confusion from no signal. The whole
    self-similarity argument is built on this distinction, so it is asserted here before
    any grid is analysed.
    """
    rng = np.random.default_rng(31)
    n_q = 200

    # (a) no signal: the query is unrelated noise -> weak true peak
    ref = ar1(20_000, seed=31)
    cache = mt.build_ref_cache(ref)
    noise_only = mt.match(rng.normal(0, 1, n_q), cache, metric="pearson", k_true=5000)
    assert noise_only.z_margin < 4.0

    # (b) alias confusion: the reference literally repeats, so a perfect rival exists
    period = 4000
    block = ar1(period, a=0.9, seed=32)
    repeated = np.tile(block, 5)
    cache_rep = mt.build_ref_cache(repeated)
    k0 = 1000
    alias = mt.match(
        repeated[k0 : k0 + n_q].copy(), cache_rep, metric="pearson", k_true=k0, exclusion_s=50
    )
    # the true peak is as strong as it gets, yet rivals tie it
    assert alias.z_margin > noise_only.z_margin
    assert alias.gap2 <= 1e-6, f"expected a tying rival, gap2={alias.gap2}"


def test_rank_true_is_zero_for_a_clean_planted_query():
    ref = ar1(10_000, seed=33)
    cache = mt.build_ref_cache(ref)
    k0 = 4321
    res = mt.match(ref[k0 : k0 + 300].copy(), cache, metric="pearson", k_true=k0, topk=5)
    assert res.rank_true == 0
    assert res.topk_idx[0] == k0
    assert res.hit(eps=0)


def test_hit_tolerance_behaves():
    ref = ar1(10_000, seed=34)
    cache = mt.build_ref_cache(ref)
    k0 = 2000
    res = mt.match(ref[k0 : k0 + 200].copy(), cache, metric="pearson", k_true=k0)
    assert res.hit(0) and res.hit(1) and res.hit(5)
    res.k_hat = k0 + 3
    assert not res.hit(0)
    assert not res.hit(2)
    assert res.hit(3)


# --------------------------------------------------------------------------------------
# 9. golden-file regression
# --------------------------------------------------------------------------------------


def test_golden_values_are_stable():
    """Fixed seed, fixed answer. A refactor that changes the science has to change this
    number too, which makes it impossible to do by accident.

    These values were RECORDED from the implementation after every correctness test above
    passed -- they are a change detector, not an independent prediction. The tests that
    establish correctness are the zero-noise exactness, fast-vs-slow, scipy and stumpy
    checks; this one only notices if behaviour drifts afterwards.
    """
    ref = ar1(50_000, a=0.99, seed=1234)
    cache = mt.build_ref_cache(ref)
    rng = np.random.default_rng(4321)
    k0 = 31_415
    q = ref[k0 : k0 + 600] + rng.normal(0, 0.5, 600)

    res = mt.match(q, cache, metric="pearson", k_true=k0, exclusion_s=30)
    assert res.k_hat == k0
    assert res.rank_true == 0
    assert res.n_offsets == 49_401
    assert res.score == pytest.approx(0.9976207497242838, abs=1e-12)
    assert res.z_margin == pytest.approx(2.953878942801955, abs=1e-9)
    assert res.gap2 == pytest.approx(0.18697371790621353, abs=1e-12)
