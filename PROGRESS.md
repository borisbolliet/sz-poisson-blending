# Progress report (12 November 2026)

This page summarises what has been done so far, what we found and what comes next.
- **Idealised telescope (Q1–Q3):** details in [`REPORT.md`](REPORT.md).
- **Realistic Simons Observatory setup:** details in [`analysis/realistic/step1/RESULTS.md`](analysis/realistic/step1/RESULTS.md), [`analysis/realistic/step2/RESULTS.md`](analysis/realistic/step2/RESULTS.md), [`analysis/realistic/step3/RESULTS.md`](analysis/realistic/step3/RESULTS.md) (step 3a) and [`analysis/realistic/step3b/RESULTS.md`](analysis/realistic/step3b/RESULTS.md).

Every number below comes from a script in this repository. Logs are committed for the realistic setup, the patch pictures and the finder timing test; the Q1–Q3 logs are kept locally. Every Q1–Q3 and step-1 result was checked independently before being reported. So was every step-2, step-3a and step-3b result, except a few that only one reviewer computed or checked:
- step 2: the arbiter's extra analyst setups, and the map check's Poisson realisations;
- step 3a: the count-change preview with its selection calibration, and the class fractions and count change with SZiFi's default settings (the arbiter);
- step 3b: the final headline on all tiles and the arbiter's targeted tests (the arbiter; for this report, its q > 5 biases in configs a, d and e were refitted with the fits check's code on the arbiter's data), and the fits with our re-implementation of SZiFi's SO-script setting (the builder only).

Each RESULTS.md says, claim by claim, which check confirmed it. The patch pictures and the timing test carry only their own internal checks.

## Status

| part | question | status |
|---|---|---|
| Q1 | How likely are two or more clusters inside an aperture? | done, verified |
| Q2 | How often is a detection blended at a signal-to-noise (S/N) threshold? | done, verified |
| Q3 | Does ignoring blending bias cosmology from cluster counts? | done, verified (idealised telescope) |
| Realistic step 1 | Does blending survive with SO's forecast noise, the CMB and a six-channel filter? | done, verified: **yes; Y boosts and losses about halved** |
| Realistic step 2 | Does the cosmology bias survive with the SO filter? | done, verified: **with the default analysis choices, Q3's S8 bias disappears when the Y–M relation is fitted (at most +0.6σ, SO-like, over all choices tried); with the relation fixed, +1.33σ to +1.85σ (SO-like), the opposite sign to Q3** |
| Realistic step 3a | What does a real cluster finder do with blended peaks? | done, verified: **it credits each peak to its brighter cluster (like f_dom ≈ 0.7) and adds some blending by searching over position and size. It matches the step-2 cases whose S8 bias is +0.1σ to +0.4σ with the Y–M relation fitted and −0.6σ to −3.7σ with it fixed (SO-like; a preview read off step 2's table, for our finder's way of estimating the noise after masking)** |
| Realistic step 3b | Cosmology fits with the finder's own catalogue | done, verified: **the catalogue holds 1.5% (baseline) and 2.1% (goal) fewer clusters at q > 5 than a no-blending analysis expects. S8 is biased by +0.32σ / +0.68σ with the Y–M relation fitted and −2.3σ / −4.2σ with it fixed (SO-like), and a goodness-of-fit test would rarely notice** |
| SZiFi itself | Does SZiFi's own code confirm step 3a's re-implementation? | running (see the last section) |

---

## What a simulated patch looks like

<p align="center"><img src="figures/visual/patch_idealised.png" width="950"></p>

One 10°×10° patch of the idealised setup used for Q2 and Q3. It contains 71,291 halos above 10¹³ M☉.
- **(a) The true SZ sky.** Most of the "background" is faint halos that are never detected individually.
- **(b) What the telescope sees.** Noise hides almost everything; only the brightest clusters stand out.
- **(c) The matched-filter S/N map,** with 237 detections above q = 5:
  - 186 are clean;
  - 23 pass on their own but get more than 10% of their signal from other halos;
  - 23 are detected only thanks to other halos;
  - 5 are not a real cluster.
- **Bottom: close-ups of three detections.** The cyan circle is the main halo; dashed circles are the other halos that add the most signal.
  - **A** (q = 11.9): other halos add 13% to a z = 0.47 cluster. The largest are neighbours at z = 0.52, 0.27 and 1.14, adding 9%, 6% and 3%.
  - **B** (q = 6.7): the cluster alone (z = 0.90) gives only 4.6, below the threshold. A halo at z = 0.63, almost exactly in front of it, adds 56% of the cluster's own signal; all other halos together add a net 47%. It is a projection, not a physical pair.
  - **C** (q = 6.4): no single halo gives more than S/N 1.9. Twenty-six other halos each add more than 1% of that signal (the largest at z = 0.53, 0.86 and 1.38), plus a noise fluctuation of +1.6. No real cluster is behind this peak.

<p align="center"><img src="figures/visual/patch_so.png" width="950"></p>

The same halos seen by the SO large-aperture telescope (baseline noise, lensed CMB, atmosphere).
- **Top:** a 2°×2° close-up of the brightest cluster (z = 0.29) in the six channels.
  - The CMB is the large-scale pattern shared by all six channels. At 27 GHz, and partly at 280 GHz, the instrument noise is larger.
  - The cluster is the dark dot at 93 and 145 GHz and vanishes at 225 GHz, close to the SZ null at 217 GHz. Above the null it should be bright, but at 280 GHz the noise hides it.
- **Bottom:** the SO multi-frequency filter against the idealised filter. They give 228 and 237 detections, 168 in common.
  - The SO filter is better for compact clusters and much worse for large, nearby ones, because the CMB and atmosphere dominate at large scales.
  - Examples: the brightest cluster drops from S/N 47 to 29, and a z = 0.07 cluster from 34 to 9.
- Both S/N maps show the maximum in each 2′ block, so compact peaks stay visible.

Checks:
- The idealised map was rebuilt with the pipeline's own code and noise seed. It gives the same 613 candidate peaks as the stored run, with S/N matching to 3×10⁻⁶.
- The SO filter's noise on this pixel grid matches the library value to better than 10⁻⁴.

Script: [`analysis/visual/v01_patches.py`](analysis/visual/v01_patches.py) (about 20 s; set `PID` to draw another patch). Log: [`v01_patches.log`](analysis/visual/v01_patches.log).

---

## Results so far

### Idealised telescope (Q1–Q3)

**Setup:**
- **Halos:** Tinker08 mass function, Planck18 cosmology, 10¹³–10¹⁶ M☉ and 0.01 < z < 3. That is about 29 million halos on the full sky, placed independently (Poisson).
- **SZ signal:** Planck 2013 Y–M relation with 17% log-normal scatter, Arnaud pressure profile.
- **Telescope:** 1.4′ beam, white y-noise 4×10⁻⁶ arcmin, no CMB or foregrounds.
- **Detection:** matched filter. At q > 5 this gives about 86,000 detections on the full sky.

**Q1, close pairs.**
- Positions are Poisson, so the number of clusters in an aperture follows P(N) = Poisson(Σ πθ²) exactly, also around a detected cluster.
- Only 0.7% of q > 5 detections have another detection within 2′.

**Q2, blending at a threshold.** Blending is a confusion problem, not a pair problem.
- About 15% of detections have their Y inflated by more than 10%.
- About 8% are detected only because of that boost, and about 10% of real clusters are lost.
- Merged pairs, the "pair" effect, are only about 0.5%.
- Most of the extra signal comes from faint halos at *other* redshifts along the line of sight.
- **How to compute it:** use the compound-Poisson distribution of the summed neighbour signal ("P(D)", as for source confusion in radio and far-IR surveys). It agrees with 324 simulated 100 deg² map patches to within about 2 percentage points.
- **Two traps:**
  - not removing the map mean roughly triples every blending fraction;
  - not counting on the detected sample underestimates blending by 1.3–1.5×.

**Q3, cosmology.** We generated the counts a survey would see, binned in redshift and S/N, including blending. We then fitted them with the standard model that ignores blending.
- **Bins:** 16 × 9; the separate analysis behind the last two table rows uses 12 × 10.
- **Fitted parameters:** Ωm, σ8 and up to four Y–M parameters: mass bias (1−b), slope α, redshift evolution β and scatter σlnY.
- **Fits:** maximum likelihood on noiseless data, with Fisher errors.
- **Cross-checks:**
  - MCMC for the SO-like survey (third table row);
  - Poisson realisations of the SO-like counts: 100 each for the first two rows, 50 for the third;
  - resampling of map patches.

<p align="center"><img src="figures/cosmo/bias_bars_v2.png" width="900"></p>

| what is fitted | SO-like, 16,500 deg² | full sky |
|---|---|---|
| Ωm, σ8 only (scaling relation fixed) | S8 −4.0σ | S8 −6.3σ |
| + mass bias (1−b) | S8 −0.5σ, Ωm −1.2σ | S8 −0.8σ, Ωm −1.8σ |
| + (1−b) with 4% prior, slope α, scatter σlnY | S8 +1.1σ, σ8 +1.8σ | S8 +2.1σ, σ8 +3.1σ |
| all four relation parameters, Planck-like priors | S8cl +1.1σ | S8cl +2.2σ |
| same, plus 4 free "confusion" parameters | ≤ 0.5σ | ≤ 0.7σ |

*S8 = σ8(Ωm/0.3)^0.5; the separate analysis behind the last two rows uses S8cl = σ8(Ωm/0.3)^0.3.*

- **Silent and unabsorbed:** a goodness-of-fit test flags the problem in only 7–54% of realisations. Freeing nuisance parameters does not absorb the bias; it changes which parameter takes it.
- **It can be modelled exactly and cheaply.**
  - One P(D) table costs 0.5–2.5 minutes.
  - Computing it at the fiducial cosmology and once more at the first best fit leaves ≤ 0.05σ within Planck ±3σ (≤ 0.22σ at the edges of the tested grid).
  - An 81-point emulator leaves ≤ 0.02σ.
- **The detection pipeline matters more.** Peak-finding artefacts (fragments, noise peaks, picking the best filter size) change counts by +4–6%, more than blending itself.
- **Open:** the simulated maps show 0.77 ± 0.11 of the deficit the P(D) model predicts.

### Realistic SO telescope, step 1: does blending survive?

**Setup:**
- **Instrument:** the six SO large-aperture-telescope channels (27–280 GHz), baseline and goal noise including atmosphere (Ade et al. 2019).
- **Sky:** the lensed CMB from CAMB, and the same Poisson halos.
- **Filter:** a multi-frequency matched filter (MMF).
- **Still missing:** CIB, radio, kSZ and Galactic foregrounds.

<p align="center"><img src="figures/realistic/step1_sigmaY.png" width="850"></p>

*Noise of the cluster measurement, σ_Y, versus cluster size θ500.*
- The SO filter beats the idealised one for compact clusters: 0.80× (baseline) and 0.60× (goal) at 0.5′.
- It is much worse for large ones: 3.3× (baseline) and 2.9× (goal) at 10′, because the CMB and atmosphere dominate at large scales.

| q > 5, full sky, neighbours above 10¹³ M☉ | idealised | SO baseline | SO goal |
|---|---|---|---|
| detections without blending | 85,912 | 81,705 | 128,167 |
| Y boosted by > 10% | 16.2% | **8.8%** | **10.4%** |
| Y boosted by > 20% | 10.5% | 4.7% | 5.6% |
| detected only because of the boost | 8.1% | 6.2% | 6.9% |
| lost (of clusters detectable alone) | 9.0% | 4.9% | 5.4% |
| mean Y bias of detections | +3.4% | +2.2% | +2.7% |
| net change in the number of detections | −1.6% | +1.3% | +1.4% |
| change in counts from z < 0.1 to z = 2–3 (P(D) expectation) | −5.3% → +4.0% | +0.7% → +4.8% | +0.8% → +5.0% |
| Y boosted by > 10%, attribution rules f_dom = 0.5–2 (default 1) | 12.5–19.6% | 7.4–10.1% | 8.7–11.9% |

<p align="center"><img src="figures/realistic/step1_blending_fractions.png" width="950"></p>

*Blending fractions at q > 5: idealised (blue), SO baseline (orange), SO goal (green).*
- **Left:** all neighbours.
- **Middle:** only massive neighbours (above 10¹⁴ M☉).
- **Right:** only neighbours that are themselves undetectable.

With the SO filter, most of the effect comes from undetectable neighbours.

**What step 1 shows:**
- **As a noise term, faint-halo SZ is small** with the SO filter. The undetectable halos add only 5–8% to σ_Y, in line with Raghunathan et al. (2022), who called it negligible for SO.
- **Per detected cluster, blending is still large.**
  - Y boosts and losses are about half the idealised values (×0.45–0.64).
  - Boosts over the threshold fall by only 15–23%.
  - It changes the redshift shape of the counts.
- **The SO filter needs no mean removal.** It is compensated: it ignores a uniform SZ background.
- **The numbers depend on the attribution rule,** i.e. on which cluster a blended peak is assigned to. Our default rule (f_dom = 1) is meant to count each peak once, under its brightest contributor; step 3a found that it credits both clusters in 25–69% of the peaks where a second cluster adds at least 10% of the signal. A real cluster finder pins this down (step 3a).
- **The fractions are probably upper bounds.** There are no foregrounds yet, and the halo model's SZ power is 2.65× the value SPT measures.
- **Verification:**
  - Two independent checks, a re-derivation with separate code and map simulations, reproduce the headline fractions to within about 0.1 percentage point.
  - An arbitration step tested every disputed claim.
  - Two small offsets remain open, neither of which changes the conclusions: up to 0.13 percentage points between methods, and 0.1–0.5 points in the z = 2–3 bin.

### Realistic SO telescope, step 2: does the cosmology bias survive?

**Setup:**
- **Telescope and sky:** as in step 1 (the SO filter, baseline and goal noise, no foregrounds).
- **Fits:** as in Q3. Counts in 16 redshift × 9 S/N bins above q = 5, for an SO-like survey (16,500 deg²) and the full sky. The standard model, which ignores blending, is fitted to the expected blended counts (maximum likelihood, Fisher errors).
- **Blended counts:** from step 1's P(D) with the SO filter. Each cluster is measured at its true position and size, with the summed signal of all other halos added.

<p align="center"><img src="figures/realistic/step2_bias_bars.png" width="900"></p>

*Bias in Ωm, σ8 and S8 for each set of fitted parameters (a–e, as in the table below), q > 5. Top: SO-like survey; bottom: full sky. Diamonds: the idealised filter (Q3 setup). Circles and squares: SO baseline and goal noise with the default attribution rule and the noise-only σ; the bars span the rules f_dom = 0.5–2 (other noise estimates are not shown). Triangles: SO baseline with a fainter halo population whose SZ power matches SPT's measurement. Grey band: |bias| < 1σ.*

S8 bias, SO-like survey, q > 5, default attribution rule (f_dom = 1), S/N divided by the noise-only σ:

| what is fitted | idealised filter (Q3 setup) | SO baseline | SO goal |
|---|---|---|---|
| a: Ωm, σ8 only (scaling relation fixed) | −3.95σ | **+1.33σ** | **+1.85σ** |
| b: + mass bias (1−b), no prior | −0.53σ | +0.91σ | +1.23σ |
| c: + (1−b) (no prior), scatter σlnY | −0.47σ | +0.57σ | +0.80σ |
| d: + (1−b) with 4% prior, slope α, σlnY | +1.11σ | **−0.01σ** | **+0.01σ** |
| e: as d, without the prior | +1.73σ | +0.07σ | +0.11σ |

*The idealised column is Q3's setup run through the step-2 code (Q3 itself gave −3.97σ in a and +1.11σ in d). Full sky, baseline / goal: d +0.04σ / +0.08σ, a +2.11σ / +2.92σ.*

**What step 2 shows:**
- **Fit the scaling relation (d), and Q3's S8 bias is gone** (−0.01σ / +0.01σ, SO-like, baseline / goal, default choices). The same code with the idealised filter gives +1.11σ (SO-like) and +2.07σ (full sky), as Q3 did. So the change comes from the filter, not from the code.
- **Why.** With the SO filter, the default attribution rule and the noise-only σ, blending raises the counts smoothly: +1.3% (baseline) and +1.4% (goal) in total, growing from +0.7% at z < 0.1 to +4.7% at z = 2–3 (baseline; +4.8% for clusters with q_A ≥ 0.5, as in the step-1 table) and largest near the threshold. A small shift of the slope α (−0.4σ), with Ωm −0.1σ and σ8 +0.1σ, absorbs this, and the Ωm and σ8 shifts cancel in S8. This needs α free: with α fixed (configs b and c) the change is not absorbed. In Q3, config d's bias came from how the count change varied with S/N (a loss at q = 5–7, a gain at q = 15–50); the fit mimicked it by moving the slope and the scatter, which pulled σ8 up. (If the noise level is instead the plain standard deviation of the map, the count change takes back Q3's S/N shape and config d rises to +0.3σ to +0.5σ.)
- **Fix the scaling relation (a), and a bias of the opposite sign appears:** +1.33σ to +1.85σ (SO-like) and +2.11σ to +2.92σ (full sky). Only Ωm and σ8 can absorb the extra counts. Configs b and c keep the slope α fixed at its true value and fit (1−b) without a prior; an analysis that takes the slope from external calibration would be closer to them than to d. With the default choices they give +0.6σ to +1.2σ (SO-like; +0.90σ to +1.95σ on the full sky) and, like config a, they depend strongly on the attribution rule (up to +3.2σ for b, goal, f_dom = 2).
- **The analyst's choices matter a lot when the relation is fixed, and little when it is fitted:**
  - *Attribution rule,* f_dom = 0.5–2 (SO-like survey): config a from −0.75σ to +3.15σ (baseline) and from −1.41σ to +4.72σ (goal).
  - *Noise level measured on the map,* which contains faint-halo signal, instead of the noise-only value: config a changes sign. A robust or clipped width gives −0.3σ to −1.4σ; the plain standard deviation gives −2.9σ (baseline) and −4.4σ (goal) (SO-like survey).
  - *Config d over all 20 combinations computed* of noise estimate, attribution rule and instrument noise (8 of them, including the worst case, checked by the arbiter only): −0.19σ to +0.61σ (SO-like) and −0.14σ to +0.89σ (full sky). The worst case is goal noise with the plain standard deviation and the strict rule f_dom = 0.5. None comes back to Q3's +1.1σ and +2.1σ.
- **A fit would rarely notice.** A goodness-of-fit test has power of at most 0.073 in configs b–e (SO-like survey; 0.05 means no power), and 0.153 even in the worst analyst setup (full sky: 0.12 and 0.43). Config a is easier to catch: up to 0.38 (SO-like) and 0.91 (full sky) over the setups tried.
- **Correcting once works, if the blending model is right.** A correction computed at the fiducial cosmology (SO baseline, default rule) leaves ≤ 0.17σ in S8 when the true S8 is 0.75 or 0.87 (at fixed Ωm); ignoring blending there gives up to 2.8σ.
- **A fainter SZ sky gives less.** With SZ power matching SPT's measurement ((1−b) = 0.6), config a drops to +0.57σ and config d stays at −0.02σ (SO-like, baseline).
- **A higher threshold,** q > 6, lowers config a by about a quarter (+0.98σ / +1.40σ, SO-like, baseline / goal).
- **Poisson noise does not change it.** In 200 noisy realisations (SO-like, baseline, default choices), blending shifts S8 by +1.334 ± 0.007σ in config a and −0.001 ± 0.004σ in config d, the same as the noiseless fits.

**Verification:**
- **Catalogue check,** with its own code (11 fresh full skies, direct sums over each cluster's neighbours, its own count model and fitter): config d −0.02σ / +0.00σ, config a +1.31σ / +1.84σ (SO-like, baseline / goal).
- **Map check** (500 fresh 10°×10° patches, 50,000 deg², through the SO filter; plus the Poisson realisations): config d +0.01 ± 0.05σ / +0.02 ± 0.06σ, config a +1.39 ± 0.10σ / +1.90 ± 0.10σ (SO-like, baseline / goal).
- **Arbitration:**
  - The map check found configs c and e 0.1–0.3σ higher (SO-like). The arbiter measured the blending kernel directly on 60 more patches: it agrees with the P(D) kernel within the test's errors of about 10⁻⁴, so the offset was a fluctuation in the map check's sample of clusters.
  - It confirmed σ(S8) in configs c and e to within 1.2%, added the plain-standard-deviation case raised by the map check, and computed the attribution-rule × noise-estimate corners, which only it checked (the worst cases quoted above). No published number changed.
  - The correction off the fiducial was confirmed by the catalogue check only, on fresh skies at the two other S8 values.
- **Open:**
  - which attribution rule and noise estimate a real SO pipeline uses (step 3a, below, measures them for a SZiFi-like finder);
  - freeing the redshift evolution of the Y–M relation in the fit, not yet tried;
  - two small per-bin offsets between methods (z = 1–1.5 and z = 2–3), worth less than 0.1σ in S8 if real;
  - no foregrounds, halo clustering, masks, bandpasses or relativistic SZ; the fiducial sky's SZ power is 2.65× SPT's;
  - the map-measured noise level is assumed known exactly from the fiducial sky;
  - Poisson realisations for one case only; the correction tested only along S8 at fixed Ωm (two truths); the fainter population and the robust width tried only with the default rule.

### Realistic SO telescope, step 3a: what does a real cluster finder do?

**Setup:**
- **Finder:** our own lean version of SZiFi's iterative multi-frequency matched filter (Zubeldia et al. 2023a,b), the finder being tested for SO, in [`szsim/finder.py`](szsim/finder.py). SZiFi's code was read, not copied. Like SZiFi, the finder:
  - estimates the noise from each tile, so the SZ of other halos enters its noise level;
  - finds peaks in position and in size (15 sizes, 0.5′–15′);
  - masks its q > 4 detections, re-estimates the noise and filters again once (the re-estimate differs from SZiFi's; see below).
- **Why our own version:** SZiFi is built for real data (sky tiling, masks, foreground removal), which our periodic simulated tiles do not need. Applying the filter the way SZiFi does took 11.4 s per tile in a timing test, against 0.3 s for the same algebra in Fourier space ([`analysis/finder/f00_bench_filtering.py`](analysis/finder/f00_bench_filtering.py)).
- **Sky:** 324 tiles of 10°×10° per noise level (32,400 deg², about twice the SO-like area), SO baseline and goal noise. Each tile holds all halos above 10¹³ M☉, the lensed CMB, white noise and atmosphere. Noise-only tiles and tiles with injected isolated clusters measure the finder's own false detections and selection.
- **Exact bookkeeping:** every detection is split into the signal of each halo, the CMB and the noise, so each can be labelled clean, boosted, a fragment or a false detection.

<p align="center"><img src="figures/realistic/step3_r_theta.png" width="900"></p>

*The finder's noise level relative to the noise-only value, r = σ_finder/σ_Y, against filter size (left: baseline; right: goal). Blue: before the masking step. Orange: our finder after it. Red: our re-implementation of SZiFi's masking step with its default settings, written from its source code. Green: the same with the setting of SZiFi's SO example script. Black lines: step 2's three noise estimates.*

| q > 5, 32,400 deg² per noise level | finder, SO baseline | finder, SO goal | step 1 (true positions), baseline / goal |
|---|---|---|---|
| detections | 69,496 | 103,739 | – |
| Y boosted by > 10% | **10.6%** | **12.0%** | 8.8% / 10.4% |
| detected only because of neighbours | 6.0% | 6.3% | 6.2% / 6.9% |
| lost (of clusters detectable alone) | 6.2% | 6.6% | 4.9% / 5.4% |
| fragments of a brighter detection (of all) | 5.6% | 5.4% | – |
| no real cluster behind the peak (of all) | 1.7% | 1.5% | – |
| size off by more than a factor 2 | 21.9% | 20.6% | – |

*Unless marked "of all", fractions are of the real detections that are not fragments.*

**What step 3a shows:**
- **Up to the masking step, the finder matches SZiFi.** Given the same noise model, its noise level equals step 1's filter to 4×10⁻⁶. An emulation of SZiFi's own noise estimate, templates and peak rules, written from its source code, gives the same catalogue before the masking step: 100% of q > 5 detections matched. SZiFi itself was not run.
- **Which noise level?** Before masking, the noise level holds the SZ of all halos, detected clusters included: the CMB, the noise and the tile's own SZ power predict it to 0.2–0.7%. That is close to step 2's noise estimate (iii), the plain standard deviation of the filtered map: r − 1 is 84–92% of (iii)'s excess. Masking the detections removes most of it. Our finder ends at r = 1.02–1.07, about 0.7 of the way from the noise-only value to step 2's noise estimate (i), which adds to the noise-only σ the scatter from halos below the detection threshold. How the masked noise is estimated matters:
  - Our re-implementation of SZiFi's masking step with the setting of its SO example script (dividing by the unmasked fraction only above ℓ = 12000) lands within 0.02 of our finder in r, but its noise level is about 2.5% low on maps without halos.
  - With SZiFi's default settings (dividing above ℓ = 3000; set for Planck), our re-implementation overestimates the noise at SO resolution. It blends the masked discs back towards the original map but still divides the small-scale power by the unmasked fraction. At small sizes r ends at 1.14 (baseline) and 1.18 (goal), close to (iii), and the noise level is 2.6% and 3.4% too high even on maps without halos.
- **Which attribution?** The finder credits a blended peak to the brighter cluster (97–98% of the time when the second one contributes less than half as much) and never counts a peak twice. Step 2's default rule (f_dom = 1) credits both clusters in 25–69% of the peaks where a second cluster adds at least 10% of the signal. The finder behaves like the rule with f_dom ≈ 0.7: a peak is credited to a cluster only if no single neighbour adds more than 0.7 of that cluster's own S/N (independent check: 0.71 ± 0.05 and 0.70 ± 0.03).
- **The search adds blending.** At the true positions, the finder's own filter reproduces step 1's numbers to within about 1 percentage point. Searching over position and size pulls peaks towards bright neighbours and grows the size to include them: compared with the same filter at the true positions (Y boosted for 8.3% / 9.3%, lost 4.8% / 5.1%), 2.3 / 2.6 points more Y boosts and 1.4 / 1.5 points more losses (baseline / goal).
- **Do not merge at 10′.** SZiFi's Planck scripts merge detections within 10′. At SO depth that throws away 7% (baseline) and 10–11% (goal) of real clusters, because real clusters often sit that close. Merging within 1–2′ removes the fragments and only 0.1–0.2% of real clusters.
- **Count change (a preview).** We compare the clusters the finder credits with what an analysis expects from the finder's own selection, calibrated on simulated clusters without neighbours and applied to the finder's S/N at face value: **−0.8 ± 0.4% (baseline) and −1.5 ± 0.3% (goal)**, with no trend in redshift (errors of 1–4 points above z = 1.5).
  - The finder's noise level gives −1.2% / −1.7%: an isolated cluster sees less than unit noise in the finder's units, so fewer scatter above q = 5. Neighbours and the search give back +0.5% / +0.3%.
  - The ± errors are statistical only. Calibrating the selection on noise-only tiles instead of injection tiles gives −1.6% / −2.7%, and injection sets of the same size scatter by up to 1–2 points.
  - With our re-implementation of SZiFi's default settings, and the same selection model: −4.7% / −5.8%, a larger deficit than any case in step 2's table (at most −3.6% / −4.4%). Part of it is the noise bias those settings have even without halos, so not all of it is blending.
- **What this means for cosmology (a preview read off step 2's table, not a fit).** The step-2 cases that match the finder's total count and redshift shape, attribution f_dom = 0.5–1 with noise estimate (i), gave an S8 bias of (SO-like survey; the f_dom = 0.5 ends were checked by step 2's arbiter only):
  - **+0.12σ to +0.22σ (baseline) and +0.22σ to +0.40σ (goal) with the scaling relation fitted** (config d);
  - **−0.64σ to −2.39σ and −1.00σ to −3.66σ with it fixed** (config a): negative, as in Q3, not positive as with step 2's default choices.
  - Caveats: these hold for our finder's masking step. The cases were matched on count totals and redshift shape only, not on the S/N shape that drives config d. With the noise-only calibration, the goal count change (−2.7%) lies beyond the matched cases.
  - With SZiFi's default masking, the nearest step-2 case (f_dom 0.5, noise estimate (iii)) gave config d +0.35σ / +0.61σ and config a −4.21σ / −6.23σ.

**Verification:**
- **SZiFi check:** reproduced SZiFi's conventions from its source code and applied them to the same tiles. SZiFi itself was not run: three packages it imports (NaMaster, pixell, scikit-learn) are not installed in our environment, and our sandbox did not allow running its code.
- **Independent check:** its own code for every step (painting, CMB and noise, noise estimate, filter, peaks, masking, merging, bookkeeping). It ran on the build's tiles and on 100 new skies, 200 noise-only tiles and two sets of 270 injection tiles per noise level. On the same tiles it matches the finder to machine precision. On new skies it reproduces the blending fractions, attribution and noise levels within about 2σ.
- **Arbitration:**
  - The build's first count change, −3.4% / −4.0% with a steep drop at z = 2–3, is withdrawn. The independent check found that the selection model let undetectable faint halos count (0.7–0.9 points of it); the arbiter's 4× larger injection set removed most of the rest. The redshift trend was not real.
  - The SZiFi check found that our masking step is not SZiFi's. The arbiter wrote its own implementation of SZiFi's masking step, confirmed the effect and ran it on 100 tiles per noise level.
- **Open:**
  - which masked-noise estimate an SO pipeline will use: it moves the count change from −0.8% to −4.7% (baseline);
  - the selection calibration: calibrating on injection tiles or on noise-only tiles differs by about 1 point (settled in step 3b, below), and injection sets of the same size scatter more than expected;
  - SZiFi itself not yet run;
  - no foregrounds, halo clustering or masks; the halo population's SZ power is 2.65× SPT's.

### Realistic SO telescope, step 3b: cosmology fits with the finder's own catalogue

**Setup:**
- **The analyst.** Step 3a read its cosmology off step 2's table. Here the finder's own catalogue is fitted by an analyst who models every cluster as isolated. The analyst knows the finder's noise level and how it selects isolated clusters: the best a no-blending analysis can do.
- **The analyst's selection** comes from 1.3 million *virtual* isolated clusters per noise level and finder variant. Each one's filtered signal is added to the finder's final S/N maps of the same tile without halos (same CMB and noise). Then the finder's peak rules apply, and the cluster is credited if its own signal at the peak is at least 2σ. Sparse clusters painted into the same halo-free maps and found end to end agree with them: +0.00 ± 0.27% (baseline) and +0.27 ± 0.28% (goal) at q > 5.
- **The finder's catalogue** comes from 2,348 (baseline) and 1,948 (goal) tiles of 100 deg²: step 3a's 324, 1,224 new tiles from 34 new full-sky skies, and 800 (baseline) or 400 (goal) tiles made with the sims check's generators. A halo counts once, at the S/N of its highest credited detection.
- **Fits.** In each of step 2's 16 redshift × 9 S/N bins, the ratio of the catalogue's counts to the analyst's expectation multiplies the analyst's model counts. This gives noiseless mock data, as in step 2, which are fitted with step 2's configs a–e. Three finder variants are compared: our finder, and our re-implementation of SZiFi's masked-noise step with its default settings and with its SO script's setting (ℓ_sep = 12000). The SZiFi variants use step 3a's 324 tiles only, so their errors are 2–3 times larger.

<p align="center"><img src="figures/realistic/step3b_bias_bars_goal.png" width="900"></p>

*Bias in Ωm, σ8 and S8 for each fit configuration (rows a–e), SO goal noise, q > 5. Top: SO-like survey; bottom: full sky. Diamonds: step 2's default analysis. Grey bars: step 3a's preview. Blue: our finder (open circles: merged within 10′). Orange and green: our re-implementation of SZiFi's masking with its default and SO-script settings. Whiskers: error from the finite simulations. Our finder's points use the builder's 1,548 tiles, the SZiFi variants step 3a's 324. The final values for our finder on all tiles (table below) differ from the filled blue points by at most 0.06σ at SO-like size and 0.09σ on the full sky.*

| S8 bias, q > 5, SO-like survey (full sky) | SO baseline | SO goal |
|---|---|---|
| count change against the no-blending expectation | **−1.46 ± 0.05%** | **−2.10 ± 0.05%** |
| config d: Ωm, σ8, (1−b) with a 4% prior, α, σlnY | **+0.32 ± 0.04σ** (+0.50σ) | **+0.68 ± 0.06σ** (+1.06σ) |
| config e: as d, no prior on (1−b) | +0.38 ± 0.07σ (+0.60σ) | +0.72 ± 0.09σ (+1.15σ) |
| config a: Ωm and σ8 only | **−2.33 ± 0.09σ** (−3.68σ) | **−4.23 ± 0.10σ** (−6.69σ) |
| q > 6 instead: config d / config a | +0.10σ / −0.88σ | +0.23σ / −1.56σ |

*Our finder, no merging, all tiles. The ± is the error from the finite simulations; the choice of estimator adds at most 0.04σ at SO-like size and 0.07σ on the full sky.*

**What step 3b shows:**
- **S8 is biased even when the scaling relation is fitted.** The catalogue holds 1.46% (baseline) and 2.10% (goal) fewer clusters at q > 5 than the analyst expects. The loss is almost all at q = 5–7.3 (−2.4% / −3.5%). The fit absorbs it by steepening the Y–M slope α (+1.1σ / +2.0σ), which drags σ8 up. S8 is off by +0.32σ / +0.68σ in config d for an SO-like survey and +0.50σ / +1.06σ on the full sky. A goodness-of-fit test would rarely notice. At SO-like size its power is 0.07 / 0.10 in config d and at most 0.21 in any configuration.
- **Why fewer clusters?** The finder measures S/N against its own noise level, which still holds the SZ of the halos it does not mask: 1.02–1.07 times the noise-only value. In these units a cluster's own CMB and instrument noise is less than one, while the analyst's isolated clusters scatter by one unit. So fewer faint clusters scatter above q = 5 than the analyst expects, and neighbours give back only part of the loss.
- **With the relation fixed, the sign is negative, as in Q3.** Only Ωm and σ8 can absorb the deficit: S8 is off by −2.33σ / −4.23σ (SO-like). Step 2's default analysis had +1.33σ / +1.85σ, the opposite sign.
- **The masked-noise estimate matters, most at goal depth.** Config d at goal is +0.68σ with our finder. With our re-implementation of SZiFi's masking it is +0.90σ (SO-script setting) and +1.02σ (default settings). Config a is −4.2σ, −5.7σ and −8.0σ. At baseline config d is +0.32σ, +0.31σ and −0.04 ± 0.15σ. The last value comes from the unsmoothed estimator, because the pooled one over-smooths for this variant.
- **Do not merge at 10′.** Merging within 1′ changes config d by at most 0.03σ (SO-like). Merging within 10′ removes another 7.4% / 11.2% of real clusters. Then config d is +1.04σ / about +2.6σ to +2.8σ and config a −13σ / −26σ. A goodness-of-fit test would flag it in config a (power 1.00), less reliably in config d (0.58–1.00).
- **The right no-blending reference is isolated clusters.** This settles step 3a's open question of which calibration to use. Isolated clusters on skies without halos, virtual or painted, give the same selection at q > 5.
  - Any injection tile, dense or sparse, selects slightly fewer clusters near q = 4, below the q > 5 cut.
  - Using step 3a's injection sets instead (expected counts 0.3% / 0.9% lower) would move config d by at most 0.06σ, but config a by +0.4σ / +1.5σ (± 0.6σ / ± 0.8σ).
  - Step 3a's 0.8 / 1.2-point gap between its two calibrations was mostly statistical, plus its different detection criterion.
  - A real analysis that calibrates by injecting clusters into the real maps would fold part of the confusion into its selection.
- **The preview had the right sign but too small a config-d bias.** Config d lies 0.10σ (baseline) and 0.28σ (goal) above the upper end of step 3a's preview (+0.22σ / +0.40σ). It is close to the step-2 case with attribution f_dom = 0.5 and noise estimate (iii) (+0.35σ / +0.61σ). That case's count change has the finder's S/N shape, though not its total.
- **A simple analytic selection function can be a bigger risk than blending.** Suppose the analyst uses Φ(√(q′² + k) − 5) instead of the measured selection, set to zero below q′ = 2.
  - q′ is a cluster's true Y over the finder's noise level. k = 0 is steps 1–2's model; k = 2 or 3 corrects for the search over position and size, as in ACT-, SPT- and SZiFi-type analyses.
  - This analyst sees almost the same blending effect, within 0.23σ in configs a and d.
  - But the function adds its own error, before any blending. In config a it is +15.6σ / +17.8σ for k = 0 and −6.3σ / −8.9σ for k = 3, against about −2σ / −4σ from blending itself. k = 2 comes closest, at +1.8σ / +0.9σ.

**Verification:**
- **Fits check:** rebuilt the data vector from the build's per-halo tables with its own binning, kernel interpolation and estimator. It re-measured the selection with exactly painted clusters, and refitted with its own count model and fitter. Every headline count and bias was reproduced within its error.
  - Its own estimator gave config d 0.04σ more. The arbiter traced this to the check's own discretisation, and the estimator choice is now quoted as an error of at most 0.04σ.
  - It also found that the pooled estimator over-smooths for SZiFi's defaults at baseline, so that value now comes from the unsmoothed one.
- **Sims check:** its own pipeline on 400 fresh tiles per noise level, from the halo sampler to the fits, plus its own injections. It reproduced the selection, the goal results and the SZiFi-default results (with its own re-implementation of SZiFi's masking, on 60 tiles). At baseline its count change was larger: −1.71 ± 0.13% against −1.36 ± 0.06%.
- **Arbitration:**
  - The builder's code on the sims check's own maps credits the same clusters (11,911 of 11,912). So the codes agree, and the gap comes from the tile samples.
  - The arbiter made 400 more tiles with the sims check's generators and pooled all tiles for the final headline. This moved config d at baseline from +0.36σ to +0.32σ.
  - It also confirmed the sims check's correction of the dense-injection explanation, and re-optimised the goal fits merged within 10′.
- **For this report:** the q > 5 biases of configs a, d and e were refitted with the fits check's count model and fitter on the arbiter's pooled data ([`verify_real/step3b_report/`](verify_real/step3b_report/)).
  - Configs a and d agree within 0.01σ.
  - Config e comes out 0.01–0.03σ lower at SO-like size and up to 0.06σ lower on the full sky, the known difference between the two count models in config e.
  - Weighted with the check's model counts, the count change is unchanged.
- **Open:**
  - The builder's tiles and the tiles made with the sims check's generators differ at baseline by 2.6σ (−1.36% against −1.64%), for an unknown reason. If the difference is real, the baseline biases carry about ±0.05σ (config d) and ±0.25σ (config a) more.
  - Which masked-noise estimate an SO pipeline will use. Step 3b uses step 3a's re-implementation of SZiFi's masking; SZiFi itself is being run now.
  - Config e at goal is at the 0.1σ precision target.
  - The pooled estimator over-smooths for the SZiFi variants. It also assumes unclustered halos; with clustered halos (Websky) the per-bin estimator and many more tiles will be needed.
  - At goal the per-bin and pooled estimators differ bin by bin more than expected (χ² 150 for 106 bins), although their config-d biases agree within 0.03σ.
  - The goal fits merged within 10′ are uncertain by ±0.2σ (SO-like) and ±0.4σ (full sky).
  - Not traced: why injection tiles lose clusters near q = 4, and the extra loss at z < 0.1 (−2.5% against about −1.5% elsewhere at baseline).
  - No foregrounds, halo clustering or masks; the halo population's SZ power is 2.65× SPT's.

---

## What we have done

### External codes used
Only general scientific and cosmology libraries; everything SZ-specific is our own code.

| code | used for |
|---|---|
| colossus 1.4 | Planck18 cosmology, distances, Tinker08 mass function (σ(M) from Eisenstein & Hu); Bocquet16 and Despali16 for robustness |
| numpy, scipy | FFTs, Hankel transforms, special functions, likelihood maximisation (L-BFGS-B) |
| numba | fast loops: filter cross-response tables, neighbour sums, painting halos into maps |
| healpy | sky pixelisation (jackknife regions, sampler uniformity tests) and one neighbour search in a catalogue check |
| CAMB 2.0.5 | lensed CMB spectrum for the SO filter |
| CLASS (classy 3.4.2) | independent check of the CMB spectrum |
| emcee 3.1 | MCMC cross-check of the cosmology fits |
| matplotlib, astropy | figures; small use in two checks |

### What we implemented

- **[`szsim/`](szsim/README.md), the core library:**
  - `cosmo_hmf.py`: mass-function tables.
  - `sampler.py`: a full-sky Poisson halo sampler (29 million halos in about 8 s).
  - `signal.py`:
    - the Y–M relation with scatter and the Arnaud profile;
    - the matched-filter noise σ_Y;
    - the cross-response ρ_AB (how much of cluster B's signal enters cluster A's filter);
    - detection probabilities and expected counts.
  - `instrument.py`: the SO noise model, the tSZ spectrum and the multi-frequency matched filter.
- **Blending calculations:**
  - **Catalogue level** ([`analysis/blending/`](analysis/blending/)): the compound-Poisson P(D) by FFT, with mean removal and the attribution rule. The SO version, for a filter whose response goes negative around the cluster, is in [`analysis/realistic/step1/pdlib1.py`](analysis/realistic/step1/pdlib1.py).
  - **Map level** ([`analysis/q2_maps/`](analysis/q2_maps/)): 10°×10° patches with painted halos, a matched filter at 12 sizes (0.5–10′), peak finding, merging within 2′, and an exact split of each detection into the contributions of individual halos.
- **Cosmology** ([`analysis/cosmo/`](analysis/cosmo/)):
  - a binned counts model and blended data from the P(D) and from the maps;
  - maximum-likelihood fits with Fisher errors, Poisson realisations and MCMC;
  - the P(D) correction on an 81-point grid of cosmologies, with an emulator;
  - corrected models (fixed, emulated, iterated) and cheap effective models.
- **Realistic SO, step 1** ([`analysis/realistic/step1/`](analysis/realistic/step1/)):
  - the instrument and filter;
  - noise and confusion versus filter size;
  - the P(D) with the SO filter, and the go/no-go table.
- **Realistic SO, step 2** ([`analysis/realistic/step2/`](analysis/realistic/step2/)):
  - blended detection kernels from the P(D) with the SO filter, for each attribution rule and noise estimate;
  - the counts model with the SO filter's noise;
  - the cosmology fits (configs a–e), a linear split of the main biases into amplitude, redshift-tilt and S/N-shape parts, and the correction tested off the fiducial.
- **Realistic SO, step 3a** ([`szsim/finder.py`](szsim/finder.py), [`analysis/realistic/step3/`](analysis/realistic/step3/)):
  - the SZiFi-like finder, the tile simulator and the exact split of each detection into halos, CMB and noise;
  - noise-only and injection runs for the finder's own false detections and selection;
  - SZiFi's masked-noise step as an option, for comparison.
- **Realistic SO, step 3b** ([`analysis/realistic/step3b/`](analysis/realistic/step3b/)):
  - the analyst's selection from virtual isolated clusters, and its check with clusters painted into maps;
  - three estimators of the finder's count change, two of them paired between each halo and its isolated copy (one per bin, one pooled over redshift);
  - 1,224 more tiles, the cosmology fits (configs a–e) and an analyst with an analytic selection function.
- **Pictures of patches** ([`analysis/visual/`](analysis/visual/)): the two figures at the top of this page.

### How every result is checked
- **One step at a time.** Each step is built once, then checked by two independent re-implementations written without the builder's new code. They reuse only earlier-validated pieces such as the halo sampler, and the map checks deliberately run the build's filter on independently simulated skies. A typical pair:
  - a re-derivation with different numerics;
  - a simulation that measures the same thing on maps or catalogues.
- **Arbitration.** A final pass tests every disputed claim with targeted runs, fixes the bugs it finds and lists what is still open. It writes a results file that says, claim by claim, which check confirmed it.
- **Where the checks live:**
  - `verify/`: Q1 and Q2;
  - `verify_cosmo/`: Q3;
  - `verify_real/`: the realistic setup.
- **Compute:** everything runs on a 10-core laptop with 16 GB of RAM.

---

## Next: SZiFi itself on our tiles

Steps 3a and 3b used our own finder and our re-implementation of SZiFi's masking. SZiFi's own code now runs in a separate environment. It is being run on up to 30 of step 3a's tiles per noise level and on noise-only tiles, with four settings:
- no inpainting;
- its defaults (30 tiles);
- its defaults with the SO example script's ℓ_sep = 12000;
- the SO example script's full settings, as is and with an isotropic covariance.

The settings other than the defaults use 2–10 tiles. The run checks:
- whether SZiFi gives the same catalogue as our emulation before the masking step;
- whether our re-implementation of its masking reproduces its noise level, its false detections and its number of q > 5 detections;
- what SZiFi's SO example settings give.

It has two independent checks and an arbitration, as before. Its answer bears on step 3b's open question: which masked-noise estimate an SO pipeline will use.

**After that:**
- foregrounds (CIB, radio, kSZ);
- clustered halos from the Websky simulation;
- the P(D) inside a full cluster-count likelihood code;
- a paper draft.
