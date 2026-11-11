# Close pairs and blending of SZ clusters on a Poisson sky

We simulated a full sky of galaxy clusters to answer three questions:

1. How likely is it to find two (or N) clusters inside a given aperture?
2. For a given detection threshold, how often are detections blended?
3. Does blending bias cosmology from cluster counts, and can a likelihood account for it?

## In short

- **Close pairs follow simple Poisson counting.** The number of clusters in an aperture depends only on how many clusters there are per square degree and on the aperture's area. The same holds around a detected cluster. The simulations confirm this exactly.
- **Close pairs of detected clusters are rare.** At a signal-to-noise cut of 5, only 0.7% of detections have another detection within 2 arcmin.
- **Blending is common anyway.** About 15% of detections have their SZ signal inflated by more than 10%. The cause is mostly faint, undetectable clusters along the same line of sight at other redshifts.
- **So blending is a confusion problem, not a pair problem.** It has to be computed from the summed signal of all faint neighbours. A "two clusters within some radius" picture misses it.
- **Ignoring blending biases cosmology.** A cluster-count analysis that ignores it gets S8 wrong by about 1σ (SO-like) to 2σ (full sky), and usually passes goodness-of-fit. Modelling it exactly is cheap (Section 6).
- **Two traps can triple the answer.** Forgetting to remove the mean SZ signal of the map does this. Ignoring that a signal-to-noise cut prefers boosted objects also biases the answer low.

---

## 1. What we simulated

| ingredient | choice |
|---|---|
| Halo abundance | Tinker08 mass function, Planck18 cosmology |
| Halo range | masses 10¹³–10¹⁶ M☉, redshift 0–3 |
| Positions | random on the full sky, independent of each other (Poisson) |
| SZ signal | Planck 2013 Y–M relation, 17% scatter, Arnaud pressure profile |
| Telescope | 1.4′ beam, uniform white noise (roughly SO-like), no CMB |
| Detection | matched filter; a cluster is detected if its signal-to-noise q exceeds q_cut = 4–7 |

The full sky holds about 29 million halos above 10¹³ M☉. Of these, 86,000 are detected at q > 5, or about 2 per square degree.

| q_cut | 4 | 5 | 6 | 7 |
|---|---|---|---|---|
| detections, full sky | 132,000 | 86,000 | 62,000 | 47,000 |
| per square degree | 3.2 | 2.1 | 1.5 | 1.1 |

![dN/dz and counts](figures/build/dNdz_Nabove.png)

---

## 2. Question 1: two or more clusters in one aperture

### The answer

Halos are placed independently, so any subset is also placed independently. That holds whether the subset is "detected clusters", "halos above 10¹⁴" or "clusters at z > 1". The number N in an aperture of radius θ then follows a Poisson law:

$$P(N=n)=\frac{e^{-\mu}\mu^n}{n!},\qquad \mu=\Sigma\times\text{aperture area}$$

Here Σ is the number per unit area of the population you care about. On the sphere, the area is 2π(1 − cos θ). For apertures under a few degrees, πθ² is fine.

Three practical points:

- **Around a detected cluster the law is the same.** Count only the *other* objects. Knowing there is a cluster at the centre tells you nothing about its neighbours when positions are independent.
- **The probability of at least one neighbour is 1 − e^(−μ).** For small μ this is ≈ μ.
- **For the whole sky, the expected number of close pairs is N_total × μ / 2.** The chance that at least one such pair exists anywhere is 1 − e^(−expected number).

### The numbers (q > 5 detection at the centre)

| neighbour type | within 1′ | 2′ | 5′ | 10′ |
|---|---|---|---|---|
| another detected cluster | 0.18% | 0.72% | 4.4% | 17% |
| any halo above 10¹⁴ M☉ | 0.8% | 3.3% | 19% | 57% |
| any halo above 3×10¹³ M☉ | 9% | 32% | 91% | ~100% |
| two or more other detections | 0.0002% | 0.003% | 0.1% | 1.5% |
| close pairs of detections on the full sky | 78 | 312 | 1,952 | 7,807 |

So a detected cluster rarely has a detected neighbour nearby. It very often has a group-sized halo nearby, almost always at a different redshift.

### Did the simulations agree?

Yes. We ran 120 full-sky realisations, plus thousands of extra skies for rare configurations. Every probability above matched the formula within the statistical noise.

How to read the figure:
- Draw a circle of radius θ around a detected cluster.
- Each curve gives the probability that the circle holds exactly n *other* detected clusters.
- Dots are the simulations; lines are the Poisson formula, with nothing fitted.
- Small circles are almost always empty (black curve ≈ 1). The chance of one neighbour grows as θ² (blue): 0.7% at 2′.
- Near 23′ the circle holds one other cluster on average, and beyond that several neighbours become typical.

![P(N) vs aperture](figures/q1_aperture/PN_readable.png)

### Caveat

Real clusters are clustered, and our simulated ones are not. Using the measured clustering strength of massive halos, close-pair probabilities go up by about 15–25% at 1–5′. To include it, multiply μ by (1 + w̄), where w̄ is the angular correlation function averaged over the aperture.

---

## 3. Question 2: blending at a detection threshold

### Four ways a detection can be blended

| name | what happens |
|---|---|
| **Merged pair** | two clusters that would each be detected sit so close that the pipeline sees one |
| **Contaminated** | the detection is real, but neighbours add to its signal, so its Y (and inferred mass) is too high |
| **Boosted** | the cluster would be below threshold on its own; neighbours push it over |
| **Lost** | the cluster would be detected on its own, but is not: it is swallowed by a brighter neighbour's peak, or pushed down by the mean subtraction |

### How big each effect is

We measured these in two independent ways:
- **Catalogue calculation:** a formula, explained below, checked against direct sums over simulated catalogues.
- **Simulated maps:** 32,000 deg² of maps with every halo painted in, noise added and a real detection pipeline run. These are the "ground truth".

| q_cut | merged pairs | contaminated by >10% | contaminated by >20% | boosted | lost |
|---|---|---|---|---|---|
| 4 | 0.6% | 14–19% | 8–12% | 8–9% | 11–12% |
| 5 | 0.5% | 14–16% | 8–11% | 8% | 9–10% |
| 6 | 0.4% | 14–15% | 8–9% | 8% | 8–9% |
| 7 | 0.3% | 13% | 7–8% | 7–8% | 7–8% |

Each range spans the catalogue calculation and the maps. Contamination, boosting and loss are percentages of detections. Loss is a percentage of clusters that would be detectable on their own.

What this shows:

- **Merged pairs are a small effect** (about 0.5%). They follow the Q1 formula.
- **Contamination, boosting and loss are 15–30 times larger.** They hardly drop as the threshold rises.
- **The catalogue calculation agrees with the maps** to within about 2 percentage points.

![blending in maps](figures/q2_maps/blending_fractions_ALL.png)

### Why the pair picture fails

A cluster sits on a background of many faint halos. Each is too faint to detect, but together they add a sizeable, fluctuating signal. Most of this extra signal comes from halos of 10¹⁴ M☉ and above at different redshifts. There is no single radius that captures it: the radius you would need grows from about 8′ to 12′ as the threshold rises. The pair formula does describe one thing: clusters lost by being swallowed by a brighter neighbour, with an effective radius of about 5–8′ that depends on the peak finder.

### How to compute it: sum over all neighbours

The right tool is the one used for source confusion in radio and far-IR surveys, often called **P(D)**. The idea:

1. Every halo near the cluster adds a little signal, weighted by how the filter responds at that distance.
2. The halos are Poisson-distributed, so the total added signal has a known distribution (a "compound Poisson" distribution). It can be computed with one Fourier transform from the halo mass function.
3. Add the noise, apply the threshold, and count how often the added signal exceeds 10% (contaminated), or makes the difference between passing and failing (boosted or lost).

The distribution computed this way matched direct sums over 60 simulated skies across its full range, including the rare bright tail.

![neighbour signal distribution](figures/blending/PD_ccdf_random_points.png)

### Two traps

- **Remove the mean SZ signal of the map.** The faint halos give the whole sky a small mean y (about 6×10⁻⁷). A plain matched filter is not insensitive to a constant offset, so it reads this mean as signal: +0.4σ for small clusters, up to +3σ for large ones. If you do not subtract the mean, every blending fraction roughly triples and never converges as you add fainter halos. Real pipelines remove it implicitly, through filtering or CMB-like large-scale power; a simulation has to do it explicitly.
- **Compute fractions on the detected sample.** Clusters with extra signal from neighbours are more likely to pass the cut. So blending among detected clusters is higher than among all clusters: by about ×1.3 at q > 5 and ×1.5 at q > 7. Formulas that ignore this underestimate blending.

---

## 4. How robust are these numbers?

| change | effect on blending (contamination, boosting) |
|---|---|
| noise halved | ×1.3 more |
| noise doubled | ×0.7 |
| beam 5′ instead of 1.4′ | ×1.4 more |
| Planck-like (7′ beam, more noise) | ×0.8–1.0 |
| different mass function | ×0.95 |
| more or less Y–M scatter | no change |
| only halos up to z = 1 | ×0.9 |
| clustering included | ×1.1 more |
| halo floor 10¹³ vs 3×10¹³ M☉ | no change (converged) |

Question 1 stays exactly Poisson in every configuration; only the number of detections per square degree changes.

Two more checks:
- **Physical overlap is negligible.** Clusters overlapping in 3D are about 0.1% of close pairs. Blending is a projection effect.
- **The detected catalogue itself is not Poisson.** Because neighbours boost each other over the threshold, detected clusters at 2–5′ separation are 1.5–2.7× more common than Poisson. Very close pairs (under about half a beam) are suppressed because they merge.

![robustness](figures/robustness/robustness_scaling.png)

The figure re-runs the simulation in 13 configurations. Each configuration changes one thing (noise, beam, neighbour mass floor, maximum redshift, Y–M scatter or mass function) and uses 40 full skies. Each point is one configuration.

**Top left: close pairs follow the same Poisson curve in every setup.**
- The x-axis is λ = Σ_det × aperture area, the expected number of other detections in the circle for that configuration. The y-axis is how often the simulations find at least 1 (circles) or at least 2 (squares) other detections.
- The lines are the Poisson predictions, 1 − e^(−λ) and 1 − e^(−λ)(1 + λ). At small λ these become λ and λ²/2, so they have slopes 1 and 2.
- All configurations lie on the lines (χ² = 124 for 124 points). Noise, beam, scatter and mass function only change Σ_det, which moves a point along the curve.
- To answer question 1 for any survey, you only need Σ_det.

**Top right: what sets the number of detections.**
- This panel shows detections per deg² at q > 5 when the noise (blue) or the beam (orange) is scaled relative to the baseline (black dot, 2.08 deg⁻²). Lines are analytic; points are simulations.
- Noise matters most. Counts scale as noise^(−1.6 to −1.7) near the baseline: halving the noise gives about three times more detections.
- The beam matters less. Counts scale as beam^(−0.3 to −1.1), because typical detected clusters (about 2′) are as large as the 1.4′ beam or larger.
- The red star is a Planck-like setup: a 7.3′ beam and about three times more noise give 0.094 deg⁻². That is about 20 times fewer detections, so close-pair probabilities are about 20 times smaller.

**Bottom left: blending fractions are fairly stable.**
- At q_cut = 5, the panel shows contaminated detections (Y inflated by more than 10%) and boosted detections. Circles and squares have the map mean removed (the physical case); triangles are the raw map.
- With the mean removed, contamination stays at 11–22% and boosting at 6–11%, even though the number of detections changes about 60-fold.
- More sensitive surveys blend more (×1.3 at half the noise), because they detect fainter clusters, closer to the confusion background.
- A larger beam blends more (×1.4 at 5′), because the filter collects more neighbours.
- Scatter and the choice of mass function hardly matter.
- On the raw map the fractions are 2–5 times higher and unstable, up to 90% at a 5′ beam. This is the mean-y trap.

**Bottom right: the mean boost is exactly predictable.**
- The y-axis is the average extra signal-to-noise that neighbours add to a detection on the raw map, from the simulations.
- The x-axis is the analytic prediction: mean sky y × the filter's response to a uniform background, in noise units (Campbell's theorem for Poisson sums).
- All configurations lie on the 1:1 line. The boost runs from 0.4σ (neighbour floor 10¹⁴) to 2σ (5′ beam), and is about 1σ at baseline.
- This is the offset that mean removal subtracts. Because it can be computed analytically, you can correct for it without simulations, or remove it with a compensated filter.

---

## 5. Recipe for another survey

1. **Count your detections.** Integrate the mass function times the probability of detection to get detections per square degree, Σ_det.
2. **Close pairs (Q1).** Use Poisson with μ = Σ × area. Multiply by about 1.15–1.25 for clustering at arcminute scales.
3. **Merged pairs.** 1 − exp(−Σ_det π θ_b²), with θ_b about the filter size. This is a small effect.
4. **Contamination and boosting.**
   - Compute the summed-neighbour distribution (P(D)) from all halos down to about 10¹³ M☉, using your filter's response.
   - Remove the map mean.
   - Apply the threshold and count on the detected sample.
5. **Check once against maps.** Inject clusters into maps and run your own pipeline, because the number of clusters swallowed by brighter ones depends on the peak finder.

---

## 6. Does blending bias cosmology from cluster counts?

We added blending to a standard cluster-count likelihood. The likelihood is binned in redshift and observed signal-to-noise (q > 5). We then asked what happens to Ωm, σ8 and S8 if blending is ignored. Full numbers are in `REPORT_cosmo_detailed.md`.

### Short answer

- **Yes, it can be modelled exactly.** Add the summed-neighbour signal (P(D)) to the usual noise in the selection function. Two rules are essential:
  - remove the map mean;
  - count each peak once, under its brightest cluster.
- **It is cheap.** One P(D) table costs 0.5–2.5 minutes. You do not need to recompute it at every likelihood step. Either one recomputation at the first best fit, or a small emulator, brings the residual bias below 0.05σ.
- **If you ignore it, expect a silent bias of about 1σ in S8 for an SO-like survey (16,500 deg²), and about 2σ on the full sky.** The exact size depends on how freely the mass–observable relation is fitted (table below).
- **The fit would usually look fine.** A goodness-of-fit test catches the problem in only 7–54% of realisations.

### What blending does to the counts

<p align="center"><img src="figures/cosmo/blended_vs_naive_zq.png" width="900"></p>

- **Left:** percentage change in counts per (z, q) bin.
  - Nearby, faint detections are lost, up to −12% at z < 0.1 and 5 < q < 6. They are big on the sky, so the mean subtraction lowers them, and the extra scatter pushes many below the cut.
  - Distant detections gain +2–5%. They are small, sit on top of the confusion, and get boosted over the cut.
- **Right:** the same change in units of Poisson noise on the full sky. It reaches 3σ per bin at low z.
- **The total change is only −1.6%.** The danger is the *shape*: fewer low-z clusters and more high-z clusters look like a change of cosmology.

### Bias in cosmological parameters if blending is ignored

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

| what is fitted | SO-like, 16,500 deg² | full sky |
|---|---|---|
| Ωm, σ8 only (scaling relation fixed) | S8 −4.0σ | S8 −6.3σ |
| + mass bias (1−b) | S8 −0.5σ, Ωm −1.2σ | S8 −0.8σ, Ωm −1.8σ |
| + (1−b) with 4% prior, slope α, scatter σlnY | S8 +1.1σ, σ8 +1.8σ | S8 +2.1σ, σ8 +3.1σ |
| all four relation parameters, Planck-like priors | S8 +1.1σ | S8 +2.2σ |
| same, plus 4 free "confusion" parameters | ≤ 0.5σ | ≤ 0.7σ |

**What the nuisance parameters do:**
- **They do not absorb the bias.** The shift in S8 stays the same size or grows when nuisances are freed. It only looks smaller in σ units because the error bars grow.
- **(1−b) takes up the overall amplitude, but not the redshift tilt.** The truth is still excluded at about 5σ jointly on the full sky.
- **The slope α mimics the tilt.** It rises by about 2σ and drags σ8 up. This is why the sign of the bias flips between rows b and d.
- **Raising the cut to q > 6** roughly halves every bias.

<p align="center"><img src="figures/cosmo/mcmc_contours_v2.png" width="900"></p>

*MCMC for the SO-like survey, row d.* Grey shows the naive model fitted to data without blending, centred on the truth. Blue shows the naive model fitted to blended data. σ8 moves up by about 1.5σ, and α moves to compensate.

### What to do in practice

1. **Best:** put the P(D) convolution into the selection function. Compute it once at a fiducial cosmology, then once more at the first best fit; or use an 81-point emulator (about 40 CPU-minutes, residual bias ≤ 0.02σ).
2. **Cheap fallback:** add an extra Gaussian scatter and offset that both depend on the cluster's angular size, with 4 free parameters. This removes most of the bias (≤ 0.7σ remains) and widens σ(Ωm) by about 40%.
3. **Not good enough:** a single constant scatter, a shift in the (1−b) prior, or a log-normal rescaling of Y. These leave about half the misfit.
4. **Whatever you do, calibrate the detection pipeline on injected maps.** Peak-finding (fragments, noise peaks, the maximum over filter scales) changes counts by +4–6%. That is more than blending does.

### Open points

- **Maps vs model.** Simulated maps show about 0.77 ± 0.11 of the blending deficit the model predicts. If this shortfall is real, the biases above are about 20% too large.
- **Highest-q bin (q > 50).** Two codes disagree at the 1–2% level. This affects σ8 biases by about 0.2σ.
- **Missing ingredients.** The setup has no CMB, foregrounds or masks, no weak-lensing mass calibration, and no halo clustering. h, Ωb, ns and the mass function are fixed.

---

## 7. Things not fully settled

- At q > 7 the boosted fraction from maps is 7.2–7.9%, depending on the run (a 2–3σ spread).
- The "pushed below threshold" loss fraction at q > 5 is 4.8% in one run and 5.2% in another.
- Not modelled: CMB, foregrounds, compensated filters and deblending. CMB and compensated filters would reduce the mean-signal effect.

---

## Appendix: files and validation

- **Code:** library `szsim/` (mass function, sampler, signal model); analyses in `analysis/`; independent checks in `verify/`. Python: `.venv/bin/python`.
- **Detailed technical report:** every number, test statistic and verifier disagreement is in `REPORT_detailed.md`.
- **Two-page summary:** `twopager/twopager.pdf`.
- **Validation:** each analysis was checked by two independent re-implementations, and disputed numbers were re-run on fresh random seeds.

| figure | script |
|---|---|
| figures/build/dNdz_Nabove.png | analysis/build/validate_sampler.py |
| figures/q1_aperture/PN_readable.png | analysis/q1_aperture/q1_twopager_figure.py |
| figures/blending/PD_ccdf_random_points.png | analysis/blending/b06_figures.py |
| figures/q2_maps/blending_fractions_ALL.png | analysis/q2_maps/analysis_blend.py (PIDS=0:324 TAG=_ALL) |
| figures/robustness/robustness_scaling.png | analysis/robustness/aggregate.py |
| figures/cosmo/blended_vs_naive_zq.png | analysis/cosmo/likelihood/c07_figures.py |
| figures/cosmo/bias_bars_v2.png | analysis/cosmo/bias/c17_figures_v2.py |
| figures/cosmo/mcmc_contours_v2.png | analysis/cosmo/bias/c17_figures_v2.py |
