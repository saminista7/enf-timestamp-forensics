"""The matcher: score a short query against every offset in a long reference.

Written to be fast, exact, and checkable against a slow reference implementation, since
everything downstream depends on it being right.

**The formulation.** For a query `q` of length `n` and a reference `r` of length `N`, the
Pearson correlation at offset `k` is

    rho(k) = ( S_qr(k)/n - qbar * rbar(k) ) / ( sq * sr(k) )

with

    S_qr(k) = sum_i q_i r_{k+i}            one FFT cross-correlation, every k at once
    rbar(k) = (C1[k+n] - C1[k]) / n        prefix sums of r
    sr(k)^2 = (C2[k+n] - C2[k])/n - rbar(k)^2

and the squared-error criterion is **algebraically the same three quantities**:

    SSE(k) = Sqq - 2 S_qr(k) + Srr(k),     Srr(k) = C2[k+n] - C2[k]

so one FFT yields both of the criteria used in the literature (minimum MSE and maximum
correlation) at no extra cost. They are genuinely different tests -- MSE is sensitive to
mean and scale offsets between query and reference, Pearson is not, so both get recorded
rather than picking one.

Complexity is `O(N log B)` with block size `B` rather than `O(N n)`. Measured: a full
all-offsets scan takes about 3 ms over a one-day reference and 20 ms over a week.

**The identity worth knowing.** For z-normalised windows,
    znormED(k)^2 = 2 n (1 - rho(k))

which is why `stumpy`'s matrix profile is literally the minimum of this matcher's
false-offset landscape, and why the model-free uniqueness measurement and the matcher can
be cross-checked against each other rather than merely compared.

**Numerical hazards, all real at these lengths.** Prefix sums over tens of millions of
squared millihertz values lose relative precision badly, so the reference is mean-centred
before accumulation, accumulation is float64, and the raw-MSE path reconstructs the
uncentred quantities exactly rather than approximately. Sliding variances can go slightly
negative from cancellation on near-constant stretches -- which real deadband grids
produce -- so they are floored, and the number of times that happens is counted and
returned rather than silently swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import oaconvolve

__all__ = [
    "RefCache",
    "MatchScores",
    "MatchResult",
    "build_ref_cache",
    "match_scores",
    "match_scores_slow",
    "match",
    "sliding_mean_std",
    "znorm_ed_from_pearson",
    "pearson_from_znorm_ed",
]

# Relative floor applied to sliding variances. Below this a window is treated as
# degenerate (constant) rather than correlated with anything.
_VAR_FLOOR_REL = 1e-12


# --------------------------------------------------------------------------------------
# reference cache
# --------------------------------------------------------------------------------------


@dataclass
class RefCache:
    """Prefix sums of a mean-centred reference, built once and reused across trials."""

    rc: np.ndarray  # reference minus its global mean, float64
    csum: np.ndarray  # prefix sums of rc,      length N+1
    csum_sq: np.ndarray  # prefix sums of rc**2,  length N+1
    mean_global: float
    n_ref: int
    var_global: float

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rc.shape


def _blocked_cumsum(x: np.ndarray, block: int = 1 << 16) -> np.ndarray:
    """Prefix sums with bounded error growth.

    A plain `cumsum` over `N` terms accumulates rounding like `O(N)`; summing within
    blocks and then offsetting by exact block totals keeps it near `O(sqrt(N))`. At
    `N = 3e7` -- a 90-day reference at one sample per second -- that is the difference
    between a usable and an unusable prefix sum.
    """
    n = x.size
    if n <= block:
        out = np.empty(n + 1, dtype=np.float64)
        out[0] = 0.0
        np.cumsum(x, out=out[1:])
        return out

    n_blocks = (n + block - 1) // block
    out = np.empty(n + 1, dtype=np.float64)
    out[0] = 0.0
    running = 0.0
    for b in range(n_blocks):
        lo = b * block
        hi = min(lo + block, n)
        np.cumsum(x[lo:hi], out=out[lo + 1 : hi + 1])
        out[lo + 1 : hi + 1] += running
        running = float(out[hi])
    return out


def build_ref_cache(reference: np.ndarray, *, block: int = 1 << 16) -> RefCache:
    r = np.ascontiguousarray(reference, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"reference must be 1-D, got shape {r.shape}")
    if r.size < 2:
        raise ValueError("reference must have at least 2 samples")
    if not np.all(np.isfinite(r)):
        raise ValueError("reference contains non-finite values; clean it before caching")

    mu = float(np.mean(r))
    rc = r - mu
    return RefCache(
        rc=rc,
        csum=_blocked_cumsum(rc, block=block),
        csum_sq=_blocked_cumsum(rc * rc, block=block),
        mean_global=mu,
        n_ref=r.size,
        var_global=float(np.mean(rc * rc)),
    )


def sliding_mean_std(cache: RefCache, n_query: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean and population std of every length-`n_query` window of the reference.

    Returns `(mean, std, n_degenerate)` in centred coordinates. `n_degenerate` counts
    windows whose variance hit the floor; on a deadband grid that count is physically
    meaningful, so it is reported rather than discarded.
    """
    n = int(n_query)
    if n < 2 or n > cache.n_ref:
        raise ValueError(f"query length {n} outside [2, {cache.n_ref}]")
    s1 = cache.csum[n:] - cache.csum[:-n]
    s2 = cache.csum_sq[n:] - cache.csum_sq[:-n]
    mean = s1 / n
    var = s2 / n - mean * mean
    floor = _VAR_FLOOR_REL * max(cache.var_global, np.finfo(np.float64).tiny)
    degenerate = int(np.count_nonzero(var < floor))
    np.maximum(var, floor, out=var)
    return mean, np.sqrt(var), degenerate


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


@dataclass
class MatchScores:
    pearson: np.ndarray | None = None
    sse: np.ndarray | None = None  # raw sum of squared errors, in the data's own units
    n_degenerate: int = 0
    n_query: int = 0

    @property
    def n_offsets(self) -> int:
        arr = self.pearson if self.pearson is not None else self.sse
        return 0 if arr is None else int(arr.size)


def match_scores(query: np.ndarray, cache: RefCache, *, metric: str = "both") -> MatchScores:
    """Score `query` at every offset. `metric` is 'pearson', 'sse', or 'both'."""
    if metric not in ("pearson", "sse", "both"):
        raise ValueError(f"unknown metric {metric!r}")
    q = np.ascontiguousarray(query, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError(f"query must be 1-D, got shape {q.shape}")
    n = q.size
    if n < 2 or n > cache.n_ref:
        raise ValueError(f"query length {n} outside [2, {cache.n_ref}]")

    # one FFT cross-correlation gives sum_i q_i * rc_{k+i} for every k
    s_qrc = oaconvolve(cache.rc, q[::-1], mode="valid")

    mean_rc, std_rc, degenerate = sliding_mean_std(cache, n)

    out = MatchScores(n_degenerate=degenerate, n_query=n)

    if metric in ("pearson", "both"):
        qbar = float(np.mean(q))
        qc = q - qbar
        sq = float(np.sqrt(np.mean(qc * qc)))
        if sq <= 0:
            # a constant query correlates with nothing; define rho = 0 and say so
            out.pearson = np.zeros_like(s_qrc)
        else:
            cov = s_qrc / n - qbar * mean_rc
            out.pearson = np.clip(cov / (sq * std_rc), -1.0, 1.0)

    if metric in ("sse", "both"):
        # Reconstruct the UNCENTRED squared error exactly. With rc = r - mu:
        #   S_qr   = s_qrc + mu * sum(q)
        #   S_rr   = (C2[k+n]-C2[k]) + 2 mu (C1[k+n]-C1[k]) + n mu^2
        mu = cache.mean_global
        s1 = cache.csum[n:] - cache.csum[:-n]
        s2 = cache.csum_sq[n:] - cache.csum_sq[:-n]
        s_qr = s_qrc + mu * float(np.sum(q))
        s_rr = s2 + 2.0 * mu * s1 + n * mu * mu
        out.sse = float(np.dot(q, q)) - 2.0 * s_qr + s_rr

    return out


def match_scores_slow(
    query: np.ndarray, reference: np.ndarray, *, metric: str = "both"
) -> MatchScores:
    """Textbook `O(N n)` implementation. For tests and tiny cases only.

    Every fast path above is validated against this. A fast matcher with no slow twin is
    a fast matcher nobody has checked.
    """
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    n, N = q.size, r.size
    k_max = N - n + 1
    pear = np.empty(k_max) if metric in ("pearson", "both") else None
    sse = np.empty(k_max) if metric in ("sse", "both") else None

    qbar = q.mean()
    qc = q - qbar
    sq = np.sqrt(np.mean(qc * qc))
    for k in range(k_max):
        w = r[k : k + n]
        if pear is not None:
            wc = w - w.mean()
            sw = np.sqrt(np.mean(wc * wc))
            pear[k] = 0.0 if sq <= 0 or sw <= 0 else float(np.mean(qc * wc) / (sq * sw))
        if sse is not None:
            d = q - w
            sse[k] = float(np.dot(d, d))
    return MatchScores(pearson=pear, sse=sse, n_query=n)


# --------------------------------------------------------------------------------------
# picking a winner, and describing how convincingly it won
# --------------------------------------------------------------------------------------


@dataclass
class MatchResult:
    k_hat: int
    score: float
    metric: str
    topk_idx: np.ndarray = field(repr=False, default_factory=lambda: np.array([], dtype=int))
    topk_score: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    k_true: int | None = None
    score_true: float | None = None
    rank_true: int | None = None
    gap2: float = float("nan")  # best minus best-outside-the-exclusion-zone
    z_margin: float = float("nan")  # (true score - null mean) / null sd
    null_mean: float = float("nan")
    null_sd: float = float("nan")
    n_offsets: int = 0
    n_degenerate: int = 0

    def hit(self, eps: int) -> bool:
        """Did the chosen offset land within `eps` samples of the truth?"""
        if self.k_true is None:
            raise ValueError("no ground truth supplied")
        return abs(self.k_hat - self.k_true) <= eps


def match(
    query: np.ndarray,
    cache: RefCache,
    *,
    metric: str = "pearson",
    topk: int = 5,
    k_true: int | None = None,
    exclusion_s: int = 0,
    resolution_s: float = 1.0,
    scores: MatchScores | None = None,
) -> MatchResult:
    """Find the best-scoring offset and characterise the win.

    `gap2` and `z_margin` separate the two ways a match fails, and recording both from the
    first trial is what makes that distinction measurable later:

    * **alias confusion** -- the true peak is strong (`z_margin` high) but a near-identical
      rival is stronger (`gap2` low). This is the self-similarity failure, expected on
      grids with repeating dispatch structure or a control deadband.
    * **no signal** -- the true peak is not strong at all (`z_margin` low). This is the
      noise failure.

    `rank_true` makes the same distinction a histogram: alias confusion piles up at ranks
    2-10, while a no-signal failure spreads uniformly.
    """
    if metric not in ("pearson", "sse"):
        raise ValueError(f"match() needs 'pearson' or 'sse', got {metric!r}")
    sc = scores if scores is not None else match_scores(query, cache, metric=metric)
    arr = sc.pearson if metric == "pearson" else sc.sse
    if arr is None:
        raise ValueError(f"scores do not contain {metric!r}")

    # Pearson is maximised, squared error is minimised; work with a "higher is better"
    # view so the ranking logic below is written once.
    better = arr if metric == "pearson" else -arr

    k_hat = int(np.argmax(better))

    # Top-k by partial selection, not a full sort. A full argsort over every candidate
    # offset is O(N log N) and dominates the cost of the FFT it follows: at a one-day
    # scope that is ~86k elements sorted per trial, for the sake of five of them.
    k = max(min(topk, better.size), 1)
    part = np.argpartition(better, better.size - k)[-k:]
    order = part[np.argsort(better[part])[::-1]]
    res = MatchResult(
        k_hat=k_hat,
        score=float(arr[k_hat]),
        metric=metric,
        topk_idx=order.astype(int),
        topk_score=arr[order],
        n_offsets=int(arr.size),
        n_degenerate=sc.n_degenerate,
    )

    excl = int(round(exclusion_s / resolution_s))
    if k_true is not None:
        res.k_true = int(k_true)
        res.score_true = float(arr[k_true])
        res.rank_true = int(np.count_nonzero(better > better[k_true]))

        # the null population: everything far enough from the truth to be a wrong answer
        mask = np.ones(arr.size, dtype=bool)
        lo = max(k_true - excl, 0)
        hi = min(k_true + excl + 1, arr.size)
        mask[lo:hi] = False
        if mask.any():
            null = better[mask]
            res.null_mean = float(np.mean(null))
            res.null_sd = float(np.std(null))
            if res.null_sd > 0:
                res.z_margin = float((better[k_true] - res.null_mean) / res.null_sd)
            res.gap2 = float(better[k_true] - np.max(null))
    else:
        mask = np.ones(arr.size, dtype=bool)
        lo = max(k_hat - excl, 0)
        hi = min(k_hat + excl + 1, arr.size)
        mask[lo:hi] = False
        if mask.any():
            res.gap2 = float(better[k_hat] - np.max(better[mask]))
    return res


# --------------------------------------------------------------------------------------
# the matrix-profile bridge
# --------------------------------------------------------------------------------------


def znorm_ed_from_pearson(rho: np.ndarray, n_query: int) -> np.ndarray:
    """`sqrt(2 n (1 - rho))` -- the z-normalised Euclidean distance.

    This identity is what lets `stumpy`'s matrix profile and this matcher be treated as
    one object: the matrix profile of a reference IS the minimum of the false-offset
    landscape, measured by an algorithm that knows nothing about our matcher.
    """
    return np.sqrt(np.maximum(2.0 * n_query * (1.0 - np.asarray(rho, dtype=np.float64)), 0.0))


def pearson_from_znorm_ed(dist: np.ndarray, n_query: int) -> np.ndarray:
    d = np.asarray(dist, dtype=np.float64)
    return 1.0 - (d * d) / (2.0 * n_query)
