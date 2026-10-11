# Third-party code and data

Our own code is MIT (see `LICENSE`). This file records everything else, because the licence
situation in this field is genuinely mixed and one wrong copy would make the repository
undistributable.

## Code we may vendor

| Component | Location | Licence | Status |
|---|---|---|---|
| ENFormant (Bellingcat community), ENF extraction | not vendored yet | BSD-2-Clause | Fine to vendor. Keep the original copyright headers and record the upstream commit. |
| ENF-WHU dataset tooling | not vendored; used as a data dependency | MIT | May vendor if useful. Some baselines ship as P-coded MATLAB (run-only, not readable). |

## Code we must NOT copy

| Component | Licence | Why not |
|---|---|---|
| `openenf/openenf` | **GPLv3** | Copyleft would propagate to this entire repository. We read it for its reference-data endpoints and its score convention, and cite it as prior art. **No code copied.** Note the project explicitly excludes the grid *data* from GPLv3, so the Zenodo `.freqdb` files remain CC-BY and are fine to use. |
| `robert/enf-matching` | **none stated** | No licence means all rights reserved. The accompanying blog post is the clearest published explanation of the pipeline; we reimplement from the prose. **No code copied.** |

Record the reason for any future addition here.

## Data

All redistributed or derived data must carry its attribution. Per-file provenance lives in
`data/manifest/data_manifest.yaml`; this is the summary.

| Dataset | Licence | Attribution requirement |
|---|---|---|
| power-grid-frequency.org / OSF `by5hu` (32 measurement sites) | CC-BY-4.0 | Cite Jumar et al. *Power grid frequency data base*, arXiv:2006.01771, and Rydin Gorjão et al. *Nat. Commun.* 11:6362 (2020). |
| OSF `m43tg`, TSO mirror | mixed | **TransnetBW data is published with explicit permission but WITHOUT an open licence.** It must be excluded from any derived corpus we redistribute, or separately cleared. |
| OpenENF prebuilt `.freqdb` (Zenodo 7741427, 7809233) | CC-BY-4.0 | Cite the Zenodo records. |
| NESO System Frequency (GB) | NESO Open Licence | OGL-style, attribution required. |
| Fingrid dataset 339 | CC-BY-4.0 | Attribution to Fingrid. |
| ENF-WHU | MIT | Cite the ENF-WHU papers (Hua et al.). |


