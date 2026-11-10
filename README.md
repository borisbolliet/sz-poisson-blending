# Close pairs and blending of SZ galaxy clusters

Full-sky Poisson simulations of galaxy clusters seen through the thermal Sunyaev–Zel'dovich (SZ) effect, used to answer two questions:

1. **What is the probability of finding two (or N) clusters within an aperture?**
2. **Given a detection signal-to-noise threshold, how often are detections blended?**

Every number below was computed by a script in this repository. Each one was also checked by an independent re-implementation, or re-run on fresh random seeds.

📄 **Two-page summary:** [`twopager/twopager.pdf`](twopager/twopager.pdf) · 📘 **Report:** [`REPORT.md`](REPORT.md) · 🔬 **All numbers and tests:** [`REPORT_detailed.md`](REPORT_detailed.md)

---

## Key results

- **Close pairs follow Poisson statistics exactly.** The number of clusters in an aperture depends only on the number of clusters per square degree and on the aperture's area. The same law holds around a detected cluster.
- **Detected clusters rarely sit together.** At signal-to-noise q > 5, only **0.7%** of detections have another detection within 2′.
- **Blending is common anyway.** About **15%** of detections have their SZ signal inflated by more than 10%. About **8%** are detected only because of that boost, and about **10%** of real clusters are lost. The cause is mostly faint, undetectable halos at *other redshifts* along the line of sight.
- **So blending is a confusion problem, not a pair problem.** It has to be computed from the summed signal of all faint neighbours (a "P(D)" calculation), not from pairs within a radius.
- **Two traps.**
  - Not removing the map's mean SZ signal roughly *triples* every blending fraction.
  - Not computing fractions on the *detected* sample underestimates blending by 1.3–1.5×.

---

## The model

| ingredient | choice |
|---|---|
| Halo abundance | Tinker08 mass function (M500c), Planck18 cosmology |
| Range | 10¹³–10¹⁶ M☉, 0 < z < 3, full sky (≈ 29 million halos) |
| Positions | independent and uniform on the sphere (Poisson) |
| SZ signal | Planck 2013 Y500–M500 relation, 17% log-normal scatter, Arnaud pressure profile |
| Telescope | SO-like: 1.4′ beam, uniform white noise, no CMB or foregrounds |
| Detection | matched filter; detected if signal-to-noise q > q_cut (4–7) |

| q_cut | 4 | 5 | 6 | 7 |
|---|---|---|---|---|
| detections, full sky | 132,000 | 86,000 | 62,000 | 47,000 |
| per square degree | 3.2 | 2.1 | 1.5 | 1.1 |

---

## Question 1: two or more clusters in an aperture

Positions are independent, so any subset of clusters is also Poisson: detected clusters, massive halos, or a redshift slice. The number N inside an aperture of radius θ is

$$P(N=n)=\frac{e^{-\mu}\mu^n}{n!},\qquad \mu=\Sigma\times\pi\theta^2 ,$$

where Σ is the number of clusters of that kind per unit area. Use the exact cap area 2π(1 − cos θ) above a few degrees.

- **Around a detected cluster** the same law gives the number of *other* clusters. The chance of at least one neighbour is 1 − e^(−μ) ≈ μ.
- **Over the whole sky** the expected number of pairs closer than θ is N_total·μ/2.

<p align="center"><img src="figures/q1_aperture/PN_readable.png" width="560"></p>

**How to read the plot:**
- Draw a circle of radius θ around a detected cluster. Each curve is the probability that it holds exactly n *other* detected clusters.
- Dots are 120 simulated full skies; lines are the Poisson formula, with nothing fitted.
- Small circles are almost always empty.
- The chance of one neighbour grows as θ².
- Near 23′ there is one neighbour on average.

**Chance that a detected cluster (q > 5) has at least one neighbour:**

| neighbour within | 1′ | 2′ | 5′ | 10′ |
|---|---|---|---|---|
| another detected cluster | 0.2% | 0.7% | 4% | 17% |
| any halo > 10¹⁴ M☉ | 0.8% | 3% | 19% | 57% |
| any halo > 3×10¹³ M☉ | 9% | 32% | 91% | ~100% |
| close pairs of detections, full sky | 78 | 312 | 1,950 | 7,800 |

*Caveat:* real clusters are clustered. Using halo bias and the Limber approximation, this raises close-pair probabilities by about 15–25% at 1–5′. Multiply μ by (1 + w̄(θ)), where w̄ is the angular correlation function averaged over the aperture.

---

## Question 2: blending at a detection threshold

| type | what happens |
|---|---|
| **Merged pair** | two clusters that would each be detected are so close the pipeline sees one |
| **Contaminated** | a real detection whose signal (and so mass) is inflated by neighbours |
| **Boosted** | too faint to be detected alone; pushed over the threshold by neighbours |
| **Lost** | detectable alone but missed: swallowed by a brighter peak, or pulled below threshold |

These were measured in two independent ways:
- **Catalogue calculation:** the compound-Poisson P(D) method below, checked against direct sums over 60 simulated skies.
- **Simulated maps:** 32,000 deg² with every halo painted in, noise added and a multi-scale matched filter run. These are the ground truth.

| q_cut | merged pairs | contaminated >10% | contaminated >20% | boosted | lost |
|---|---|---|---|---|---|
| 4 | 0.6% | 14–19% | 8–12% | 8–9% | 11–12% |
| 5 | 0.5% | 14–16% | 8–11% | 8% | 9–10% |
| 6 | 0.4% | 14–15% | 8–9% | 8% | 8–9% |
| 7 | 0.3% | 13% | 7–8% | 7–8% | 7–8% |

Ranges span the catalogue calculation and the maps. Lost is a percentage of clusters that would be detectable on their own; the other columns are percentages of detections.

<p align="center"><img src="figures/q2_maps/blending_fractions_ALL.png" width="900"></p>

**Why pairs are the wrong picture.**
- Each cluster sits on a background of many faint halos. Each is too faint to detect, but together they add a sizeable, fluctuating signal.
- Most of this signal comes from halos of 10¹⁴ M☉ and above at other redshifts.
- No single "blending radius" describes it: the radius you would need grows from 8′ to 12′ as the threshold rises.

**How to compute it (P(D), as in radio source confusion):**
1. Every halo near the cluster adds signal, weighted by the filter's response at its distance.
2. The halos are Poisson, so the total added signal has a compound-Poisson distribution, computable from the mass function with one FFT.
3. Remove the map mean, add the noise, apply the threshold, and count fractions *on the detected sample*.

This reproduces the simulated maps to within about 2 percentage points.

<p align="center"><img src="figures/blending/PD_ccdf_random_points.png" width="900"></p>

---

## Robustness

| change | effect on contamination / boosting |
|---|---|
| noise ×0.5 / ×2 | ×1.3 / ×0.7 |
| 5′ beam instead of 1.4′ | ×1.4 |
| different mass function (Bocquet16, Despali16) | ×0.95 |
| Y–M scatter 0 or 0.3 | no change |
| clustering (2-halo term) | ×1.1 |
| neighbour mass floor 10¹³ vs 3×10¹³ M☉ | converged |

- **Close pairs stay exactly Poisson** in every configuration; only Σ_det changes.
- **3D overlap is negligible.** Physically overlapping halos are about 0.1% of close pairs, so blending is a projection effect.
- **Not modelled:** CMB, foregrounds, compensated filters, deblending.

---

## Recipe for another survey

1. **Count detections:** Σ_det = ∫ HMF × completeness, per unit area.
2. **Close pairs:** P(N) = Poisson(Σ_det πθ²), times about 1.2 for clustering at arcminute scales.
3. **Merged pairs:** 1 − exp(−Σ_det π θ_b²), with θ_b ≈ the filter scale. This is small.
4. **Contamination and boosting:**
   - Compute the compound-Poisson P(D) from all halos down to about 10¹³ M☉, using your filter's response.
   - Remove the map mean.
   - Apply the cut and count on the detected sample.
5. **Calibrate once:** inject into maps and run your own pipeline, since absorption by bright neighbours depends on the peak finder.

---

## Repository layout

```
szsim/        library: mass function tables, full-sky Poisson sampler, SZ signal, matched-filter noise (see szsim/README.md)
analysis/     analysis scripts and logs, one folder per topic (build, q1_aperture, blending, q2_maps, robustness)
verify/       independent verification scripts (two lenses per topic: derivation, reproduction)
figures/      all figures
twopager/     two-page PDF summary and LaTeX source
REPORT.md     readable report;  REPORT_detailed.md: every number, test and open disagreement
```

Simulated catalogues and maps (`data/`, ~7 GB) are not included. Regenerate them with `szsim` and the scripts in `analysis/`; each figure's script is listed in `REPORT.md`.

**Requirements:** Python 3.12 with numpy, scipy, colossus, numba, healpy, matplotlib and astropy.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy colossus numba healpy matplotlib astropy
```
