# Close pairs and blending of SZ galaxy clusters

Full-sky Poisson simulations of galaxy clusters seen through the thermal Sunyaev–Zel'dovich (SZ) effect, used to answer three questions:

1. **What is the probability of finding two (or N) clusters within an aperture?**
2. **Given a detection signal-to-noise threshold, how often are detections blended?**
3. **Does ignoring blending bias cosmology from cluster counts, and can a likelihood account for it?**

Every number below was computed by a script in this repository. Each one was also checked by an independent re-implementation, or re-run on fresh random seeds.

📄 **Two-page summary:** [`twopager/twopager.pdf`](twopager/twopager.pdf) · 📘 **Report:** [`REPORT.md`](REPORT.md) · 🔬 **All numbers and tests:** [`REPORT_detailed.md`](REPORT_detailed.md), [`REPORT_cosmo_detailed.md`](REPORT_cosmo_detailed.md)

---

## Key results

- **Close pairs follow Poisson statistics exactly.** The number of clusters in an aperture depends only on the number of clusters per square degree and on the aperture's area. The same law holds around a detected cluster.
- **Detected clusters rarely sit together.** At signal-to-noise q > 5, only **0.7%** of detections have another detection within 2′.
- **Blending is common anyway.** About **15%** of detections have their SZ signal inflated by more than 10%. About **8%** are detected only because of that boost, and about **10%** of real clusters are lost. The cause is mostly faint, undetectable halos at *other redshifts* along the line of sight.
- **So blending is a confusion problem, not a pair problem.** It has to be computed from the summed signal of all faint neighbours (a "P(D)" calculation), not from pairs within a radius.
- **Ignoring blending biases cosmology.** A cluster-count fit that ignores it shifts S8 by about **1σ** for an SO-like survey and **2σ** on the full sky. It usually passes goodness-of-fit, and mass-calibration nuisances do not absorb it. Modelling it exactly is cheap: about one minute per P(D) table, and it does not need recomputing at every step.
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

## Question 3: impact on cosmology

We put blending into a binned (z, q) Poisson count likelihood and fitted the usual *no-blending* model to blended data.

**What blending does to the counts.**
- Nearby, faint detections are lost: down to −12% at z < 0.1.
- Distant ones gain 2–5%.
- The total changes by only −1.6%, but the tilt in redshift looks like a change of cosmology.

<p align="center"><img src="figures/cosmo/blended_vs_naive_zq.png" width="900"></p>

<p align="center"><img src="figures/cosmo/bias_bars_v2.png" width="900"></p>

**What this figure shows.** It answers one question: *if you analyse cluster counts with the standard model, which ignores blending, how wrong are the cosmological parameters you get?*

**The experiment.**
- We generate the counts that a real survey would see, which *do* include blending.
- We then fit them with the standard model that assumes no blending.
- The data are the exact expected counts, with no random noise. So any offset between the fit and the true cosmology is a systematic bias caused only by ignoring blending.

**Layout.**
- **Two blocks.** The top block is an SO-like survey covering 40% of the sky (16,500 deg²). The bottom block is the full sky. Same experiment; the bigger survey has smaller error bars, so the same offset counts for more σ.
- **Three columns**, one per parameter: Ωm, σ8, and S8 = σ8 (Ωm/0.3)^0.5.
- **x-axis:** the bias, best fit minus truth, in units of the statistical error σ. Zero means no bias. The grey band marks |bias| < 1σ, where the error is smaller than the statistical uncertainty. Points outside it are significant biases.
- **Five rows, a–e.** Each row lets more parameters of the mass–SZ (Y–M) scaling relation float in the fit. Parameters not listed are fixed at their true values.
  - **a:** only Ωm and σ8 are fitted. The scaling relation is assumed perfectly known.
  - **b:** also fit the mass bias (1−b), with no prior.
  - **c:** also fit (1−b) and the scatter σlnY.
  - **d:** also fit (1−b), the slope α and σlnY, with a 4% prior on (1−b). This is the most realistic case, like a weak-lensing mass calibration.
  - **e:** same as d, but with no prior on (1−b).

**Three versions of the "data".**
- **Blue dots:** counts predicted by our blending (P(D)) model.
- **Orange squares:** counts measured in simulated maps, with spurious detections (cluster fragments and noise peaks) removed. The error bars come from resampling map patches.
- **Green triangles:** the same maps, keeping every detection. This mixes blending with defects of the detection pipeline.

**How to read it.**
- Orange sits close to blue in every row, so the blending model captures what the maps show. Orange is about 20–30% smaller.
- **Row a:** S8 comes out low by about 4σ (SO-like) or 6σ (full sky). The fit can only adjust cosmology, so it bends cosmology to mimic blending.
- **Rows b and c:** freeing the mass bias brings S8 back into the grey band. However, Ωm is still off by 1–2σ.
- **Rows d and e:** once the slope α is also free, the fit tilts the scaling relation to imitate blending. This pushes σ8 and S8 *high*, by about 1–2σ (SO-like) and 2–3σ (full sky).
- So freeing nuisance parameters does not remove the bias; it changes which parameter absorbs it, and in which direction.
- **Green** points are often far off, in either direction. They show that the detection pipeline itself (fragments, noise peaks) must be calibrated on simulations, and that this is a bigger effect than blending.

**Bottom line.** If blending is ignored, a realistic SO-like analysis (row d) gets S8 too high by about 1σ, and a full-sky one by about 2σ.

| fitted | SO-like 16,500 deg² | full sky |
|---|---|---|
| Ωm, σ8 (Y–M fixed) | S8 −4.0σ | S8 −6.3σ |
| + (1−b) | S8 −0.5σ, Ωm −1.2σ | S8 −0.8σ, Ωm −1.8σ |
| + (1−b) 4% prior, α, σlnY | S8 +1.1σ, σ8 +1.8σ | S8 +2.1σ, σ8 +3.1σ |
| all Y–M params, Planck-like priors | S8 +1.1σ | S8 +2.2σ |
| same + 4 free confusion params | ≤ 0.5σ | ≤ 0.7σ |

**How to handle it.**
- **Exact:** convolve the selection kernel with the compound-Poisson P(D). Remove the map mean and count each peak once.
  - Compute it at the fiducial cosmology, then once more at the first best fit. This leaves ≤ 0.05σ.
  - Alternatively, use an 81-point emulator, which leaves ≤ 0.02σ.
- **Cheap:** use a Gaussian offset plus scatter that depends on angular size, with 4 free parameters.
- **Not enough:** a constant extra scatter, or a shifted (1−b) prior.
- **Bigger than blending:** peak-finding artefacts (fragments, noise peaks, scale maximisation) change counts by +4–6%. Calibrate the pipeline with injected maps.

**Open points.**
- Maps show 0.77 ± 0.11 of the predicted blending deficit, so the biases may be about 20% smaller.
- There is no CMB, foreground or mask, and no weak-lensing calibration. See [`REPORT.md`](REPORT.md) §6.

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
analysis/     analysis scripts and logs, one folder per topic (build, q1_aperture, blending, q2_maps, robustness, cosmo)
verify/       independent verification scripts (two lenses per topic: derivation, reproduction); verify_cosmo/ for Q3
figures/      all figures
twopager/     two-page PDF summary and LaTeX source
REPORT.md     readable report;  REPORT_detailed.md: every number, test and open disagreement
```

Simulated catalogues and maps (`data/`, ~7 GB, and `data_cosmo/`, ~0.5 GB) are not included. Regenerate them with `szsim` and the scripts in `analysis/`; each figure's script is listed in `REPORT.md`.

**Requirements:** Python 3.12 with numpy, scipy, colossus, numba, healpy, matplotlib, astropy (and emcee for the MCMC).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy colossus numba healpy matplotlib astropy emcee
```
