# enf-timestamp-forensics

Tooling for reading open power grid frequency data, and a matcher for locating a short
frequency trace inside a long one.

`ABSTRACT.md` explains what the research is for and where it has got to.

## Background in two paragraphs

Mains frequency drifts constantly by small random amounts, and the drift is shared across a
whole synchronous area at any instant. Recordings made near electrical equipment pick up a
faint hum carrying the same drift. Compare the hum against a log of the grid's real
frequency and you can work out when the recording was made, which is how disputed audio and
video get dated or shown to have been edited.

Doing that needs two unglamorous things to work properly: reading the reference data without
misinterpreting it, and searching a long reference for a short query fast enough to be
practical. That is what is in here. The analysis built on top of it is not public yet.

## What is in here

    src/enf/io/pgf.py     reader for the open grid frequency recordings
    src/enf/matcher.py    scores a query against every offset in a reference, in one pass
    scripts/fetch_osf.py  downloads the data from the published links
    data/manifest/        which files exist, where, and their checksums. committed
    tests/

The reader is the part that took longest and is probably the most useful to anyone else. The
files are not in a consistent format: four header styles, two timestamp formats, one file in
Hz where the rest are in millihertz, one with no name on the timestamp column, and a quality
flag that means two completely different things depending on the site. It sniffs all of that
and raises rather than guessing, because the failure mode otherwise is a plausible looking
number that is wrong by a factor of a thousand. Read the module docstring before touching it.

The matcher uses one FFT cross correlation plus prefix sums, so it scores every candidate
alignment at once instead of sliding a window. About 3 ms for a day of reference data at one
sample per second, 20 ms for a week. Both criteria used in this literature, maximum
correlation and minimum squared error, fall out of the same intermediates, so you get both
for the price of one.

## Setup

Needs Python 3.11. Not 3.12 or later, numba has no wheels for those yet.

    python -m venv .venv
    ./.venv/Scripts/python -m pip install -r requirements.txt
    ./.venv/Scripts/python -m pip install -e .

Get the data. Nothing is stored in this repo, only the links and checksums in
`data/manifest/osf_catalogue.yaml`, and the fetch script verifies sizes as it goes:

    ./.venv/Scripts/python scripts/fetch_osf.py discover --node by5hu
    ./.venv/Scripts/python scripts/fetch_osf.py fetch --all

That is 5.26 GB across 32 sites. Or just the ones you want:

    ./.venv/Scripts/python scripts/fetch_osf.py fetch --only ZA02,KR01,MY01,EE01,PT01

Tests. The second line needs the data, the first does not:

    ./.venv/Scripts/python -m pytest -m "not needs_data"
    ./.venv/Scripts/python -m pytest

The data tests are the ones worth running if you are reusing the reader. They check per site
standard deviations against the published values, which tells you the units, separator,
column choice and value scale are all being read correctly at once.

## Data

Recordings come from the open power grid frequency database on OSF, node `by5hu`, licensed
CC-BY. Nothing is redistributed here. If you use it, cite:

- Jumar, Maass, Schaefer, Rydin Gorjao and Hagenmeyer, Power grid frequency data base,
  arXiv:2006.01771
- Rydin Gorjao et al., Open database analysis of scaling and spatio-temporal properties of
  power grid frequencies, Nature Communications 11:6362 (2020)

## Licence

MIT, see LICENSE. Third party terms are in LICENSE-THIRD-PARTY.md, worth a look before
vendoring anything, a couple of the obvious reference implementations in this area are GPL
or carry no licence at all.
