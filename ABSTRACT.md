# How short can a recording be and still be dated from its mains hum?

**Ahnaf Samin**
Independent research, Dhaka, Bangladesh
ORCID [0009-0004-1047-6632](https://orcid.org/0009-0004-1047-6632)

**Status: work in progress, started October 2026.** Not published, not peer reviewed, no
preprint yet. The numbers below are measured and reproducible from this repository, but they
are interim results from an ongoing project and should be read as such.

---

## Abstract

The frequency of an electricity grid drifts continuously by small random amounts, and that
drift is shared across an entire synchronous area at any instant. Recordings made near
electrical equipment capture a faint mains hum carrying the same drift, so comparing the hum
against a log of the grid's true frequency can establish when a recording was made. The
technique is used in audio and video forensics to confirm or dispute a claimed recording
time, and to detect edits.

Practice follows a rule of thumb that roughly ten minutes of recording are required for a
reliable match. That threshold appears to have been established one grid at a time, on a
small number of well regulated networks. This project asks whether the threshold is a
property of the method or a property of the particular grid, and measures it across many
grids using open frequency data covering fifteen independent synchronous areas on five
continents.

Interim results indicate the threshold is strongly grid dependent. Holding recording quality
fixed, the minimum reliably matchable length varies by a factor of 4.4 across the grids
measured so far, from 71 seconds to 308 seconds, with non overlapping confidence intervals.
The variation does not follow how volatile a grid is. Controlled comparisons against
synthetic references matched exactly in volatility, but with their temporal structure
removed, change the required length by 6 to 25 times, which indicates the governing quantity
is the structure of the drift rather than its magnitude.

If these results hold, the ten minute figure is not a general property of the method, and a
forensic examiner working on one national grid cannot safely borrow a threshold measured on
another.

## Work completed so far

- Acquisition and validation pipeline for 5.26 GB of open grid frequency data: 32
  measurement sites, 15 independent synchronous areas, one sample per second, spans from
  days to one year per site. Per site standard deviations reproduce published values to
  within 3 % across 11 sites, which validates the reader against four incompatible file
  formats in the corpus.
- An offset matching engine scoring every candidate alignment in one pass, about 3 ms per
  full scan over a day of reference data. Verified against an independent slow
  implementation, against `scipy` correlation, and against a published matrix profile
  implementation through an exact algebraic identity.
- A statistical estimator for the minimum matchable length, fitted rather than read off a
  sweep, with confidence intervals from a bootstrap clustered on trials. Recovers known
  parameters from simulated data to within 2 %.
- Partial reproduction of the published study this work builds on: two of three synthetic
  results reproduce, and the third differs by 5 dB in a way traceable to a parameter the
  source does not state.
- 120 automated tests, each validating a measurement against a process whose answer is
  known analytically.
- A written plan with thresholds fixed before measurement, and a dated record of every
  change to those thresholds. Two of the original thresholds were not met and are recorded
  as failures rather than revised.

## Remaining

Extraction of the frequency trace from real audio rather than from clean reference data; a
cross sensor validation using simultaneous GPS synchronised recordings from several sites on
one grid; the full sweep across all fifteen areas; and a predictive model for the threshold.

## Data

Grid frequency recordings come from the open power grid frequency database, licensed CC-BY.
Nothing is redistributed here; the repository holds the download links and checksums only.

- Jumar, Maass, Schaefer, Rydin Gorjao and Hagenmeyer, *Power grid frequency data base*,
  arXiv:2006.01771
- Rydin Gorjao et al., *Open database analysis of scaling and spatio-temporal properties of
  power grid frequencies*, Nature Communications 11:6362 (2020)

Audio validation uses the ENF-WHU dataset (MIT licensed).

The study this work builds on and tests itself against:

- Hua, Wang, Ye, Zhang, Wang and Xia, *Factors affecting forensic electric network frequency
  matching*, Digital Communications and Networks 10(4):1121-1130 (2024)

## Code

Python, MIT licensed. Everything in this repository is reproducible from the committed code
and the data manifest. See `README.md` for setup.
