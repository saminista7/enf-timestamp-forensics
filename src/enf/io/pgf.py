"""Loaders for the power-grid-frequency EDR recordings (OSF node by5hu).

Every confound control that depends on the raw data is enforced here, so that no
notebook can accidentally bypass one. Four properties of these files will silently
corrupt an analysis if mishandled, and each has a guard below.

1. **Semicolon-separated, and the values are MILLIHERTZ DEVIATION, not Hz.**
   `-139.519` in a `f50_*` column means 49.860481 Hz. Reading it as Hz would put the
   grid at 139 Hz and every downstream statistic would be nonsense-but-plausible.

2. **One file can hold several sites.** `EE01.csv` has `Time;f50_EE;QI_EE;f50_DE_KA;QI_DE_KA`
   -- Estonia *and* Karlsruhe, recorded simultaneously. A loader that takes "the
   frequency column" gets whichever came first. Columns are selected explicitly.

3. **`QI` is a quality flag and `QI == 2` means LINEARLY INTERPOLATED.** Interpolated
   stretches are locally smooth, which inflates the correlation time and therefore fakes
   matchability -- differentially per grid, because the grids have different gap rates.
   But the flag conflates four-second dropped packets with eleven-hour outages, so the
   criterion is the *length* of the interpolated run, not its mere presence. See
   `usable_mask`.

4. **"Nearly 100 % good data" does not mean contiguous, and the cause is not what you
   would guess.** EE01 is 99.9963 % `QI == 0` across 22.8 days, yet its longest strictly
   clean run is 2.0 days. Measured: the timestamp index is *gapless* on every site checked
   (`sampled_frac == 1.0`), so the fragmentation is caused entirely by 43 interpolated
   blips of at most four seconds -- not by recording interruptions. The index-contiguity
   check below is therefore cheap insurance rather than the active constraint here; it
   stays because a window straddling a hole would splice two different times together,
   leaving the matcher searching a reference that never existed, and the next dataset may
   well have holes.

Timestamps are UTC per the data descriptor (Jumar et al. arXiv:2006.01771). Local time
is used only to stratify trial offsets by hour-of-day, never for storage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RAW_BY5HU = REPO / "data" / "raw" / "osf_by5hu" / "FrequencyData"
INTERIM = REPO / "data" / "interim"

TIME_COL = "Time"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
SEP = ";"

# Interpolated runs up to this long may enter a window. Set from the measured run
# length distribution: on one site 73 flagged samples in 1.97M sit in 43 runs of at
# most 4 s, while another hides a single 11.6 hour stretch. 5 s separates them.
MAX_BAD_RUN_S = 5

# The header is not consistent across the corpus. Four variants are present, and the first
# three all break the obvious regex:
#   EE01     Time;f50_EE;QI_EE;f50_DE_KA;QI_DE_KA   two sites in one file
#   GB01     Time;f50_UK;QI_UK                      column carries the COUNTRY, not the site
#   US_CO01  Time;f60;QI                            bare column, no suffix at all
#   ZA02     ;f50;QI                                first column is UNNAMED
_F_COL = re.compile(r"^(?:f(?P<nom>50|60)|freq|frequency)(?:_(?P<suffix>.+))?$", re.IGNORECASE)
_QI_COL = re.compile(r"^QI(?:_(?P<suffix>.+))?$", re.IGNORECASE)

_NO_SUFFIX = ""  # key used for a bare `f50` / `f60` column

# Site code -> column suffix, where the upstream naming disagrees with the folder name.
_SUFFIX_ALIASES = {"GB": "UK"}


class DataIntegrityError(RuntimeError):
    """Raised when a file violates an assumption we would otherwise analyse through."""


# --------------------------------------------------------------------------------------
# column discovery
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteColumns:
    suffix: str
    f_col: str
    qi_col: str | None
    nominal_hz: int


@dataclass(frozen=True)
class Dialect:
    sep: str
    header: list[str]
    first_row: list[str]
    time_format: str | None  # None when the file carries no timestamp column
    absolute_hz: bool  # True when values are Hz, not mHz deviation


def read_header(path: Path, *, sep: str | None = None) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as fh:
        line = fh.readline().strip()
    return line.split(sep or _sniff_sep(line))


def _sniff_sep(header_line: str) -> str:
    """Semicolon everywhere except the eclipse campaign, which is comma-separated."""
    return SEP if header_line.count(SEP) >= header_line.count(",") else ","


def sniff_dialect(path: Path) -> Dialect:
    """Determine separator, timestamp format and value scale by LOOKING at the file.

    Four real variants in this corpus, and assuming any one of them would corrupt a
    different site silently:

      * `;` vs `,` separator
      * `2019-03-25 11:09:11` vs `2022-12-04 08:54:03.000` (MY01 has fractional seconds)
      * a named `Time` column vs ZA02's UNNAMED first column
      * **mHz deviation vs absolute Hz.** The eclipse file stores `50.002038695`. Treating
        that as a deviation in millihertz would place the grid 50 mHz from nominal with a
        standard deviation of microhertz -- a perfectly plausible-looking number that is
        wrong by a factor of a thousand. This is the trap in the module docstring, and it
        is real rather than hypothetical, so the scale is detected and asserted.
    """
    probe_rows = 5000
    with path.open("r", encoding="utf-8-sig") as fh:
        header_line = fh.readline().strip()
        sample = [fh.readline().strip() for _ in range(probe_rows)]
    sample = [s for s in sample if s]
    sep = _sniff_sep(header_line)
    header = header_line.split(sep)
    rows = [s.split(sep) for s in sample]
    row = rows[0] if rows else []

    # timestamp format: only if the first field parses as a date
    time_format: str | None = None
    if row and re.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", row[0]):
        time_format = "%Y-%m-%d %H:%M:%S.%f" if "." in row[0] else TIME_FORMAT

    # --- value scale: mHz deviation or absolute Hz?
    #
    # Magnitude alone CANNOT decide this, which took two wrong attempts to see. A deviation
    # series in millihertz routinely has a median of 100-200 (ZA02 sits there), and an
    # absolute series has a median near 50 -- the two ranges overlap, so any single
    # threshold on |median| mislabels one family or the other. (The first attempt was worse
    # still: it read GB01's first value of -69.7 mHz, found it inside a "looks like 50-60
    # Hz" window, and multiplied the entire file by a thousand.)
    #
    # The column NAME is the reliable signal -- `f50_*`/`f60_*` are deviations in mHz per
    # the data descriptor, while a bare `freq`/`frequency` column is absolute Hz. The
    # statistics then CROSS-CHECK that reading, and a disagreement raises rather than
    # picking a side, because a silently misread scale is the one error that yields
    # confident nonsense.
    absolute = False
    col_idx = next((i for i, name in enumerate(header) if _F_COL.match(name)), None)
    if col_idx is not None:
        named_absolute = _F_COL.match(header[col_idx]).group("nom") is None
        vals: list[float] = []
        for r in rows:
            if len(r) > col_idx:
                try:
                    vals.append(float(r[col_idx]))
                except ValueError:
                    continue
        absolute = named_absolute
        if vals:
            med = float(np.median(vals))
            spread = float(np.std(vals))
            # an absolute series sits within ~1 Hz of a nominal and varies by well under 1 Hz
            looks_absolute = min(abs(med - 50.0), abs(med - 60.0)) < 1.0 and spread < 1.0
            if looks_absolute != named_absolute:
                raise DataIntegrityError(
                    f"{path.name}: value scale is ambiguous. Column {header[col_idx]!r} "
                    f"implies {'absolute Hz' if named_absolute else 'mHz deviation'}, but "
                    f"median={med:.4f}, std={spread:.4f} implies "
                    f"{'absolute Hz' if looks_absolute else 'mHz deviation'}. Resolve "
                    f"explicitly rather than guessing."
                )
    return Dialect(
        sep=sep, header=header, first_row=row, time_format=time_format, absolute_hz=absolute
    )


def time_column(header: list[str]) -> str:
    """The timestamp column, taken POSITIONALLY.

    ZA02's header begins with an empty field rather than `Time`, so matching on the name
    would fail on a perfectly good four month record.
    """
    if not header:
        raise DataIntegrityError("empty header")
    return header[0]


def site_columns(path: Path) -> dict[str, SiteColumns]:
    """Map every site column-group present in a file, keyed by its column suffix.

    `Time;f50_EE;QI_EE;f50_DE_KA;QI_DE_KA` yields two entries; a bare `Time;f60;QI`
    yields one under the empty-string key.
    """
    header = read_header(path)
    qi_by_suffix = {(m.group("suffix") or _NO_SUFFIX): c for c in header if (m := _QI_COL.match(c))}
    out: dict[str, SiteColumns] = {}
    for col in header:
        m = _F_COL.match(col)
        if not m:
            continue
        suffix = m.group("suffix") or _NO_SUFFIX
        nom = m.group("nom")
        out[suffix] = SiteColumns(
            suffix=suffix,
            f_col=col,
            qi_col=qi_by_suffix.get(suffix),
            # a bare `freq` column does not declare its nominal; 0 means "ask the registry"
            nominal_hz=int(nom) if nom else 0,
        )
    if not out:
        raise DataIntegrityError(f"no frequency column in {path.name}; header={header}")
    return out


def code_to_suffix(code: str) -> str:
    """`EE01` -> `EE`, `ES_PM01` -> `ES_PM`, `DE_KA01` -> `DE_KA`, `GB01` -> `UK`.

    The folder code carries a campaign number, which the column suffix does not, and in
    one case the upstream column is named after the country instead of the site.
    """
    stem = re.sub(r"\d+$", "", code).rstrip("_")
    return _SUFFIX_ALIASES.get(stem, stem)


def resolve_columns(path: Path, code: str) -> SiteColumns:
    """Pick the column group belonging to `code`, and be loud when it has to guess.

    Resolution order, deliberately: an exact suffix match wins; failing that, a file with
    exactly ONE frequency column is unambiguous whatever it is called (this is what makes
    the bare-`f60` and `f50_UK` files work); only a multi-site file with no match is an
    error, because that is the one case where guessing could silently return the wrong
    grid.
    """
    groups = site_columns(path)
    want = code_to_suffix(code)
    if want in groups:
        return groups[want]
    hits = [g for s, g in groups.items() if s.upper() == want.upper()]
    if len(hits) == 1:
        return hits[0]
    if len(groups) == 1:
        return next(iter(groups.values()))
    raise DataIntegrityError(
        f"{path.name}: no column group for code {code!r} (suffix {want!r}) and the file "
        f"holds several sites: {sorted(groups)}. Pass suffix= explicitly."
    )


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def csv_path(code: str) -> Path:
    """Locate a site's CSV, tolerating the inconsistent sidecar naming upstream."""
    folder = RAW_BY5HU / code
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; run scripts/fetch_osf.py fetch")
    exact = folder / f"{code}.csv"
    if exact.exists():
        return exact
    candidates = sorted(p for p in folder.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"cannot pick a CSV for {code}: {[p.name for p in candidates]}")


def load_site(
    code: str, *, suffix: str | None = None, path: Path | None = None, nominal_hz: int | None = None
) -> pd.DataFrame:
    """Read one site's series.

    Returns a UTC-DatetimeIndexed frame with `f_mhz` (float32, deviation from nominal in
    millihertz) and `qi` (uint8). `nominal_hz`, when given, is asserted against the file's
    own column naming -- the registry and the data must agree, and if they do not, that is
    a fact to surface rather than a difference to average over.
    """
    path = path or csv_path(code)
    dia = sniff_dialect(path)
    cols = resolve_columns(path, suffix or code)

    # The file declares its nominal only via an `f50_`/`f60_` column name. Where it does
    # not (a bare `freq` column), the registry supplies it; where both speak, they must
    # agree, because a mismatch means we are about to analyse the wrong grid.
    nominal = cols.nominal_hz or nominal_hz
    if cols.nominal_hz and nominal_hz is not None and nominal_hz != cols.nominal_hz:
        raise DataIntegrityError(
            f"{path.name}: registry declares {nominal_hz} Hz for {code} but the file's "
            f"column is {cols.f_col!r} ({cols.nominal_hz} Hz)"
        )
    if not nominal:
        raise DataIntegrityError(
            f"{path.name}: column {cols.f_col!r} does not declare a nominal frequency and "
            f"none was supplied; pass nominal_hz= (from data/manifest/grids.yaml)"
        )

    if dia.time_format is None:
        raise DataIntegrityError(
            f"{path.name}: no timestamp column (header={dia.header}). Files without a time "
            f"base cannot be matched into and are out of scope for the primary study."
        )

    # Read by POSITION. ZA02's time column is unnamed, and pandas cannot select an
    # empty-string column by name.
    idx_time = 0
    idx_f = dia.header.index(cols.f_col)
    idx_qi = dia.header.index(cols.qi_col) if cols.qi_col else None
    usecols = [idx_time, idx_f] + ([idx_qi] if idx_qi is not None else [])

    df = pd.read_csv(
        path,
        sep=dia.sep,
        usecols=usecols,
        header=0,
        names=["_t", "_f"] + (["_qi"] if idx_qi is not None else []),
        dtype={"_f": "float64", **({"_qi": "float32"} if idx_qi is not None else {})},
        encoding="utf-8-sig",
    )

    t = pd.to_datetime(df["_t"], format=dia.time_format, utc=True, errors="coerce")
    if t.isna().any():
        bad = int(t.isna().sum())
        raise DataIntegrityError(
            f"{path.name}: {bad} unparseable timestamps (expected {dia.time_format!r})"
        )

    f = df["_f"].to_numpy(dtype=np.float64)
    if dia.absolute_hz:
        # absolute Hz -> deviation in mHz, so every site is on one scale downstream
        f = (f - float(nominal)) * 1000.0

    # Sanity bound on the SCALE, checked over good samples only. Real grid deviations stay
    # inside a few Hz even during a collapse, so a wilder spread means the units were
    # misread -- the one error that produces confident nonsense instead of an exception.
    # Flagged-invalid samples are excluded because sentinel garbage there is expected and
    # is dropped downstream anyway; including them would reject clean files.
    qi_probe = (
        df["_qi"].fillna(255).to_numpy(dtype=np.float64) if idx_qi is not None else np.zeros(len(f))
    )
    good = np.isfinite(f) & (qi_probe == 0)
    if not good.any():
        good = np.isfinite(f)
    if good.any() and np.nanpercentile(np.abs(f[good]), 99.9) > 10_000.0:
        raise DataIntegrityError(
            f"{path.name}: |deviation| reaches {np.nanmax(np.abs(f[good])):.0f} mHz on "
            f"QI==0 samples, which is implausible for a power grid. Value scale likely "
            f"misdetected (absolute_hz={dia.absolute_hz})."
        )

    out = pd.DataFrame(
        {
            "f_mhz": f.astype(np.float32),
            # a missing QI column is treated as "unknown", not as "good"
            "qi": (
                df["_qi"].fillna(255).to_numpy(dtype=np.uint8)
                if idx_qi is not None
                else np.full(len(df), 255, dtype=np.uint8)
            ),
        },
        index=pd.DatetimeIndex(t, name="t_utc"),
    )

    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    out.attrs.update(
        code=code,
        suffix=cols.suffix,
        nominal_hz=int(nominal),
        source_file=str(path),
        sep=dia.sep,
        time_format=dia.time_format,
        absolute_hz=dia.absolute_hz,
    )
    return out


def epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """UTC epoch seconds as int64. `.view('int64')` already yields an ndarray in pandas 2,
    so this exists to keep that detail in one place rather than in every caller."""
    return np.asarray(index.asi8, dtype=np.int64) // 1_000_000_000


def to_parquet_cache(code: str, *, overwrite: bool = False, **kwargs) -> Path:
    """Cache a site as parquet. Stores epoch seconds rather than a DatetimeIndex, which
    roughly halves the file and makes the integer gap arithmetic below exact."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / f"{code}.parquet"
    if out.exists() and not overwrite:
        return out
    df = load_site(code, **kwargs)
    flat = pd.DataFrame(
        {
            "t": epoch_seconds(df.index),
            "f_mhz": df["f_mhz"].to_numpy(),
            "qi": df["qi"].to_numpy(),
        }
    )
    flat.attrs = dict(df.attrs)
    flat.to_parquet(out, index=False, compression="zstd")
    return out


def load_cached(code: str) -> pd.DataFrame:
    path = INTERIM / f"{code}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; call to_parquet_cache({code!r}) first")
    flat = pd.read_parquet(path)
    return pd.DataFrame(
        {"f_mhz": flat["f_mhz"].to_numpy(), "qi": flat["qi"].to_numpy()},
        index=pd.DatetimeIndex(
            pd.to_datetime(flat["t"].to_numpy(), unit="s", utc=True), name="t_utc"
        ),
    )


def load_series(code: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(t_epoch_s, f_mhz, qi)` as plain arrays -- the form the matcher wants."""
    path = INTERIM / f"{code}.parquet"
    flat = pd.read_parquet(path) if path.exists() else None
    if flat is None:
        df = load_site(code)
        return (epoch_seconds(df.index), df["f_mhz"].to_numpy(), df["qi"].to_numpy())
    return (
        flat["t"].to_numpy(dtype=np.int64),
        flat["f_mhz"].to_numpy(dtype=np.float32),
        flat["qi"].to_numpy(dtype=np.uint8),
    )


# --------------------------------------------------------------------------------------
# contiguity and clean segments
# --------------------------------------------------------------------------------------


def bad_runs(qi: np.ndarray, *, max_qi: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Run-length encode the non-clean stretches. Returns `(start_idx, length)`."""
    bad = (qi > max_qi).view(np.int8)
    d = np.diff(np.concatenate(([0], bad, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


def usable_mask(
    qi: np.ndarray,
    *,
    f: np.ndarray | None = None,
    max_qi: int = 0,
    max_bad_run_s: int = 0,
    resolution_s: float = 1.0,
) -> np.ndarray:
    """Which samples may enter a reference window.

    `QI == 2` (linearly interpolated) covers two physically different things, and lumping
    them together is a real measurement error either way:

    * **Isolated blips.** EE01 carries 73 interpolated samples in 1.97 million (0.0037 %),
      in 43 runs of at most 4 seconds -- dropped packets. Excluding them outright fragments
      22.8 days of otherwise pristine data into 2-day pieces, which is why a strict
      `QI == 0` rule leaves almost no grid able to support a long reference.
    * **Genuine outages.** ES_GC02 is 42 % interpolated, including a single fabricated
      stretch of 41 808 s (11.6 hours). Admitting that would hand the matcher a reference
      that is locally smooth by construction, inflating the correlation time and faking
      matchability.

    So the criterion is the LENGTH of the interpolated run, not its mere presence:
    interpolated runs up to `max_bad_run_s` are tolerated, anything longer is excluded.
    `max_bad_run_s=0` recovers the strict behaviour, which is what makes the tolerance
    checkable rather than assumed.

    **Pass `f` whenever a reference window will be built from the result.** The tolerance
    above assumes `QI != 0` means "a plausible interpolated value is present", and on some
    sites that is false: ZA02 carries 2.57 million non-finite samples, every one of them at
    `QI != 0`. Tolerating a short run there inserts NaN into the reference. Supplying `f`
    makes finiteness a hard requirement that no tolerance can override, which is the
    correct reading of the rule -- a missing value is not a short interpolation.

    (The matcher refuses a non-finite reference outright, so this was caught rather than
    analysed through. Belt and braces: fix it here, keep the guard there.)
    """
    good = qi <= max_qi
    if f is not None:
        finite = np.isfinite(np.asarray(f))
        good = good & finite
    if max_bad_run_s <= 0:
        return good

    out = good.copy()
    starts, lengths = bad_runs(qi, max_qi=max_qi)
    tol = int(round(max_bad_run_s / resolution_s))
    for s, n in zip(starts[lengths <= tol], lengths[lengths <= tol], strict=False):
        out[s : s + n] = True
    if f is not None:
        # finiteness is never waived by the run-length tolerance
        out &= finite
    return out


def clean_segments(
    t: np.ndarray,
    qi: np.ndarray,
    *,
    f: np.ndarray | None = None,
    max_qi: int = 0,
    resolution_s: float = 1.0,
    min_len: int = 1,
    max_bad_run_s: int = 0,
) -> list[tuple[int, int]]:
    """Maximal half-open `[a, b)` index runs that are both usable and unbroken.

    "Usable" is `usable_mask(...)` -- clean, or an interpolated run short enough to
    tolerate. "Unbroken" means consecutive timestamps differ by exactly one sample period.

    Both checks are needed and for different reasons. The index check matters because a
    window straddling a hole would splice two different times together, so the matcher
    would be searching a reference that never existed. (In this corpus the index turns out
    to be gapless -- `sampled_frac == 1.0` everywhere measured -- so in practice QI does
    all the fragmenting, but the guard costs nothing and the next dataset may differ.)
    """
    if t.shape != qi.shape:
        raise ValueError("t and qi must have the same shape")
    n = t.size
    if n == 0:
        return []

    step = int(round(resolution_s))
    good = usable_mask(
        qi, f=f, max_qi=max_qi, max_bad_run_s=max_bad_run_s, resolution_s=resolution_s
    )

    # Single vectorised run-length pass: O(n), no Python loop over segments. A per-start
    # scan would be O(n * n_segments), which is minutes on the 31-million-row files.
    if n == 1:
        return [(0, 1)] if good[0] and min_len <= 1 else []

    # link[i] is True when samples i and i+1 belong to the same unbroken clean run
    link = good[:-1] & good[1:] & (np.diff(t) == step)

    starts = good.copy()
    starts[1:] &= ~link  # a good sample whose predecessor is not linked starts a run
    ends = good.copy()
    ends[:-1] &= ~link  # a good sample whose successor is not linked ends a run

    start_idx = np.flatnonzero(starts)
    end_idx = np.flatnonzero(ends)
    # each run contributes exactly one start and one end, so these pair up in order
    if start_idx.size != end_idx.size:  # pragma: no cover - guards a logic error, not data
        raise DataIntegrityError(
            f"run-length pass inconsistent: {start_idx.size} starts vs {end_idx.size} ends"
        )

    lengths = end_idx - start_idx + 1
    keep = lengths >= min_len
    return [(int(a), int(b) + 1) for a, b in zip(start_idx[keep], end_idx[keep], strict=False)]


def longest_clean_run(
    t: np.ndarray,
    qi: np.ndarray,
    *,
    f: np.ndarray | None = None,
    max_qi: int = 0,
    resolution_s: float = 1.0,
    max_bad_run_s: int = 0,
) -> tuple[int, int] | None:
    segs = clean_segments(
        t, qi, f=f, max_qi=max_qi, resolution_s=resolution_s, max_bad_run_s=max_bad_run_s
    )
    return max(segs, key=lambda s: s[1] - s[0]) if segs else None


@dataclass(frozen=True)
class RefWindow:
    code: str
    window_id: int
    i0: int
    i1: int
    t0_epoch_s: int
    length_s: int
    n_interpolated: int = 0  # tolerated QI!=0 samples inside this window
    max_bad_run_s: int = MAX_BAD_RUN_S

    @property
    def n_samples(self) -> int:
        return self.i1 - self.i0

    @property
    def interpolated_frac(self) -> float:
        return self.n_interpolated / self.n_samples if self.n_samples else float("nan")


def select_reference_windows(
    code: str,
    *,
    length_s: int,
    k: int,
    max_qi: int = 0,
    resolution_s: float = 1.0,
    disjoint: bool = True,
    max_bad_run_s: int = MAX_BAD_RUN_S,
) -> list[RefWindow]:
    """Pick up to `k` fully-clean, unbroken reference windows, spread through coverage.

    Windows are placed to maximise the minimum separation between them, so that comparing
    one window against another actually samples different parts of the record instead of
    clustering in whichever week happened to be clean.

    Returns fewer than `k` windows -- possibly zero -- when coverage does not allow more.
    A short return is information, not an error, and the caller must report it.
    """
    t, f_all, qi = load_series(code)
    need = int(round(length_s / resolution_s))
    segments = clean_segments(
        t,
        qi,
        f=f_all,  # a window that will be matched into must contain no NaN
        max_qi=max_qi,
        resolution_s=resolution_s,
        min_len=need,
        max_bad_run_s=max_bad_run_s,
    )
    bad_cum = np.concatenate(([0], np.cumsum(qi > max_qi)))

    # candidate placements: step through each long-enough segment
    candidates: list[tuple[int, int]] = []
    for a, b in segments:
        pos = a
        while pos + need <= b:
            candidates.append((pos, pos + need))
            pos += need if disjoint else max(need // 2, 1)
    if not candidates:
        return []

    chosen: list[tuple[int, int]] = [candidates[0]]
    while len(chosen) < k:
        # greedily add the candidate farthest from everything already chosen
        best, best_d = None, -1.0
        for c in candidates:
            if disjoint and any(c[0] < e and s < c[1] for s, e in chosen):
                continue
            d = min(abs(c[0] - s) for s, _ in chosen)
            if d > best_d:
                best, best_d = c, d
        if best is None:
            break
        chosen.append(best)

    chosen.sort()
    return [
        RefWindow(
            code=code,
            window_id=i,
            i0=a,
            i1=b,
            t0_epoch_s=int(t[a]),
            length_s=int(round((b - a) * resolution_s)),
            n_interpolated=int(bad_cum[b] - bad_cum[a]),
            max_bad_run_s=max_bad_run_s,
        )
        for i, (a, b) in enumerate(chosen)
    ]


def coverage_report(codes: list[str], *, resolution_s: float = 1.0) -> pd.DataFrame:
    """Per-site coverage, contiguity and QI breakdown.

    Reports clean contiguous window counts at several candidate lengths rather than
    assuming one, because which lengths are feasible is not obvious up front.
    """
    rows = []
    for code in codes:
        t, f, qi = load_series(code)
        n = t.size
        span_s = int(t[-1] - t[0] + 1) if n else 0
        segs0 = clean_segments(t, qi, max_qi=0, resolution_s=resolution_s)
        segs2 = clean_segments(t, qi, max_qi=2, resolution_s=resolution_s)
        longest0 = max((b - a for a, b in segs0), default=0)
        longest2 = max((b - a for a, b in segs2), default=0)

        row = {
            "code": code,
            "n_samples": n,
            "span_days": span_s / 86400.0,
            "sampled_frac": n / span_s if span_s else np.nan,
            "qi0_frac": float(np.mean(qi == 0)) if n else np.nan,
            "qi1_frac": float(np.mean(qi == 1)) if n else np.nan,
            "qi2_frac": float(np.mean(qi == 2)) if n else np.nan,
            "qi_unknown_frac": float(np.mean(qi == 255)) if n else np.nan,
            "n_clean_segments": len(segs0),
            "longest_clean_days": longest0 * resolution_s / 86400.0,
            "longest_qi2ok_days": longest2 * resolution_s / 86400.0,
            "nan_frac": float(np.mean(~np.isfinite(f))) if n else np.nan,
        }
        for label, scope_s in (("1h", 3600), ("6h", 21600), ("1d", 86400), ("7d", 604800)):
            need = int(round(scope_s / resolution_s))
            row[f"n_disjoint_{label}"] = sum((b - a) // need for a, b in segs0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("code")


def resample_to(f: np.ndarray, *, from_s: float, to_s: float, antialias: bool = True) -> np.ndarray:
    """Decimate a series. **Upsampling raises**, deliberately.

    Interpolating up to a finer grid would manufacture smoothness, which is precisely the
    quantity this study measures; a resampler that silently allowed it would let a
    resolution ablation produce a correlation-time artefact.
    """
    if to_s < from_s - 1e-12:
        raise ValueError(
            f"refusing to upsample {from_s} s -> {to_s} s: interpolation would "
            "manufacture the smoothness this study is measuring"
        )
    ratio = to_s / from_s
    q = int(round(ratio))
    if abs(ratio - q) > 1e-9:
        raise ValueError(f"non-integer decimation factor {ratio}; pick a multiple of {from_s}")
    if q == 1:
        return np.asarray(f)
    if antialias:
        from scipy.signal import decimate

        return decimate(np.asarray(f, dtype=np.float64), q, ftype="fir", zero_phase=True)
    return np.asarray(f)[::q]
