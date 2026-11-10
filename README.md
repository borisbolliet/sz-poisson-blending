# sz-poisson-blending

Poisson full-sky simulations of SZ galaxy clusters (Tinker08 mass function, z < 3) answering two questions:

1. What is the probability of finding two (or N) clusters within an aperture?
2. Given a detection SNR threshold, how often are detections blended?

- **Two-page summary:** [`twopager/twopager.pdf`](twopager/twopager.pdf)
- **Report:** [`REPORT.md`](REPORT.md) (readable), [`REPORT_detailed.md`](REPORT_detailed.md) (all numbers and tests)
- **Library:** [`szsim/`](szsim/) (mass function, full-sky sampler, SZ signal and matched-filter noise; see `szsim/README.md`)
- **Analyses:** `analysis/`; independent verification scripts: `verify/`; figures: `figures/`

Simulated catalogues and maps (`data/`, ~7 GB) are not included; regenerate them with `szsim` and the scripts in `analysis/`.

Requirements: Python 3.12 with numpy, scipy, colossus, numba, healpy, matplotlib, astropy.
