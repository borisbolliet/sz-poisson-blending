# Current conclusions (12 November 2026)

What we can say so far about the blending of SZ galaxy clusters and its effect on cluster-count cosmology. This covers the idealised telescope (Q1–Q3) and the realistic Simons Observatory (SO) steps done so far: steps 1, 2, 3a, 3b and the run of SZiFi itself.

- **Sources.** Every number here is quoted from [`PROGRESS.md`](PROGRESS.md), [`README.md`](README.md) or a step's `RESULTS.md`, except the absolute S8 shifts in section 7, which [`analysis/conclusions/c01_s8_shifts.log`](analysis/conclusions/c01_s8_shifts.log) prints from step 3b's stored fits. Those files name the script, log and checks behind each number.
- **Checks.** Each number was checked by at least one independent re-implementation, or re-run on fresh random seeds, except the few listed under [How sure we are](#how-sure-we-are). For the realistic steps two checks ran per step, an arbiter tested every disputed claim, and each `RESULTS.md` says which check confirmed which number.
- **Terms.**
  - *Idealised telescope* (Q1–Q3): one map of the SZ signal itself (Compton y), instead of maps at several frequencies.
    - It has a 1.4′ Gaussian beam, the same as SO's 145 GHz channel, and uniform white noise of 4×10⁻⁶ arcmin in y: an rms of 4×10⁻⁶ in a 1′ × 1′ pixel. At 145 GHz that is 11 μK′, close to SO baseline's 10 μK′ there. So it is roughly SO baseline's 145 GHz channel on its own, read directly as y, with its white noise only.
    - Nothing else is on the sky: no CMB, atmosphere, CIB, radio sources, kSZ or Galactic emission.
    - Clusters are found with a matched filter built for this beam and white noise. Its noise model leaves out the SZ of other halos, and unlike SO's filter it responds to the sky's mean y, which Q2 and Q3 subtract.
    - Q1, Q2's calculations and Q3's fits need no map. They use the filter's noise σ_Y, computed analytically, with the filter at each cluster's true position and size. Q2's map results, and a check of Q3 that reuses them, come from a 12-scale version of the filter (0.5′–10′) run on simulated 10° × 10° patches.
    - Its depth is close to our SO baseline model's: 85,912 expected detections at q > 5 on the full sky without blending, against 81,705 for SO baseline and 128,167 for SO goal.
    - For compact clusters (θ500 = 0.5′), SO's six-channel filter measures Y with 0.80× (baseline) and 0.60× (goal) the idealised telescope's noise, because SO's 93 and 145 GHz channels together reach a lower y-noise on small scales. For extended ones (θ500 = 10′) it has 3.3× and 2.9× the noise. The idealised telescope's map noise is white, the same on all scales, while SO's rises steeply on large scales, where the CMB and the atmosphere together dominate.
  - *Realistic SO setup* (steps 1–3b and the SZiFi runs): the six channels of SO's large-aperture telescope (27–280 GHz, beams 7.4′ to 0.9′) with SO's forecast white and atmospheric noise at uniform depth, the lensed CMB, and a multi-frequency matched filter, called "the SO filter" below. Like the idealised telescope, it has no CIB, radio sources, kSZ or Galactic emission yet. One SZiFi check used only SZiFi's own four SO channels (93–280 GHz).
  - *SO-like*: a survey of 16,500 deg² (40% of the sky).
  - *Baseline / goal*: SO's two forecast noise levels.
  - *q*: signal-to-noise; detections have q > 5 unless stated.
  - *σ*: the statistical error of the cosmology fit.
  - *Config a*: fits only Ωm and σ8, with the mass–SZ (Y–M) relation fixed.
  - *Config d*: also fits the mass bias (1−b) with a 4% prior, the slope α and the scatter σlnY.
  - *Configs b, c, e*: b also fits (1−b) without a prior, c also the scatter, and e is d without the prior. In b and c the slope α stays fixed.
- **SZiFi.** The problems found in the SZiFi cluster finder are written up separately in [`SZIFI_REVIEW.md`](SZIFI_REVIEW.md).

## In one paragraph

Blending is common, and it comes mostly from faint halos along the line of sight, not from close pairs of clusters. SO's six-channel filter roughly halves the boosts of the clusters' signal and the losses, but blending does not go away. A real cluster finder makes its own choices about blended peaks, about its noise level and about merging. These choices set the size of the bias in cosmology, and with the Y–M relation fixed even its sign. Suppose the clusters our SZiFi-like finder credits at q > 5 are fitted by an analysis that ignores blending but knows the finder's noise level and how it selects isolated clusters. Here each real halo counts once, and peaks with no real cluster behind them are left out. For an SO-like survey, S8 then comes out +0.32σ (baseline) and +0.68σ (goal) too high with the Y–M relation fitted, and −2.3σ / −4.2σ too low with it fixed. Other masked-noise estimates move the relation-fitted bias by up to about 0.35σ. A goodness-of-fit test would rarely notice. Keeping the peaks with no real cluster behind them, mostly confusion peaks of several faint halos, would cancel most of this. SZiFi's own code confirms our finder up to its masking step, and our re-implementation of SZiFi's masking once a round-off problem in SZiFi is avoided. Foregrounds, clustered halos and a full likelihood are not yet included, and the simulated halos give 2.65 times the SZ power that SPT measures.

## The S8 bias at each stage

Bias in S8 from ignoring blending, SO-like survey, q > 5, baseline / goal noise (one value: the idealised telescope, which has a single noise level).

| analysis | config a: Y–M relation fixed | config d: Y–M relation fitted |
|---|---|---|
| Q3, idealised telescope | −4.0σ | +1.1σ |
| Step 2, SO filter, step 2's default choices (each cluster at its true position and size) | +1.33σ / +1.85σ | −0.01σ / +0.01σ |
| Step 3a preview: the step-2 cases that match the finder (read off step 2's table, not a fit) | −0.64σ to −2.39σ / −1.00σ to −3.66σ | +0.12σ to +0.22σ / +0.22σ to +0.40σ |
| **Step 3b: our finder's own catalogue** | **−2.33 ± 0.09σ / −4.23 ± 0.10σ** | **+0.32 ± 0.04σ / +0.68 ± 0.06σ** |
| Step 3b, our re-implementation of SZiFi's masking: its defaults with ℓ_sep = 12000 | −2.7σ / −5.7σ | +0.31σ / +0.90σ |
| Step 3b, our re-implementation of SZiFi's masking: its defaults | −4.8σ / −8.0σ | −0.04 ± 0.15σ / +1.02σ |
| Step 3b, our finder's catalogue with the peaks that have no real cluster behind them kept | +0.26σ / −1.57σ | −0.04σ / +0.17σ |
| Step 3b, our finder's catalogue merged within 10′ | −13σ / −26σ | +1.04σ / about +2.6σ to +2.8σ |

- On the full sky the same fits give:
  - Q3: −6.3σ (a) and +2.1σ (d);
  - step 2: +2.11σ / +2.92σ (a) and +0.04σ / +0.08σ (d);
  - step 3b: −3.68σ / −6.69σ (a) and +0.50σ / +1.06σ (d).
- The ± in step 3b is the error from the finite simulations. The choice of estimator adds at most 0.04σ at SO-like size for our finder. For SZiFi's defaults at baseline the pooled estimator over-smooths, so config d there comes from the unsmoothed one (−0.04 ± 0.15σ; the pooled one gives +0.17σ). The rows with SZiFi's masking use step 3a's 324 tiles only, so their errors are 2–3 times larger. The row with the peaks that have no real cluster behind them uses the build's 1,548 tiles and was computed by the builder only.

## Conclusions

### 1. Close pairs are rare and follow Poisson statistics exactly (Q1)
- Halo positions are independent. So the number of clusters in an aperture of radius θ is Poisson with mean Σπθ², where Σ is the number of clusters per unit area. This holds around a detected cluster as well.
- At q > 5 only 0.7% of detections have another detection within 2′ (idealised telescope).
- Halo clustering, which these simulations leave out, would raise close-pair probabilities by about 15–25% at 1–5′ (Limber estimate).
- Details: [`README.md`](README.md#question-1-two-or-more-clusters-in-an-aperture), [`REPORT.md`](REPORT.md).

### 2. Blending is a confusion problem, not a pair problem (Q2)
- With the idealised telescope, about 15% of detections have their Y inflated by more than 10%. About 8% are detected only because of that boost, and about 10% of real clusters are lost. Merged pairs are only about 0.5%.
- Most of the extra signal comes from faint, undetectable halos at *other* redshifts along the line of sight.
- So blending has to be computed from the summed signal of all faint neighbours (a compound-Poisson "P(D)" calculation). This reproduces 324 simulated map patches of 100 deg² to within about 2 percentage points.
- Two traps:
  - not removing the map's mean SZ signal roughly triples every blending fraction;
  - not counting on the *detected* sample underestimates blending by 1.3–1.5×.

### 3. With an idealised telescope, ignoring blending biases cosmology (Q3)
- S8 comes out −4.0σ low with the Y–M relation fixed and +1.1σ high in config d (SO-like). On the full sky it is −6.3σ and +2.1σ.
- Freeing nuisance parameters does not absorb the bias. It only changes which parameter takes it.
- A goodness-of-fit test flags it in only 7–54% of realisations.
- Modelling it exactly is cheap:
  - one P(D) table takes 0.5–2.5 minutes;
  - computing it at the fiducial cosmology and once more at the first best fit leaves ≤ 0.05σ within Planck ±3σ;
  - an 81-point emulator leaves ≤ 0.02σ.
- The detection pipeline's own artefacts (fragments, noise peaks, picking the best filter size) change the counts by +4–6%, more than blending does.

### 4. SO's six-channel filter roughly halves the Y boosts and losses, but does not remove blending (step 1)
- At q > 5:
  - **Y boosted by more than 10%:** 8.8% (baseline) and 10.4% (goal) of detections, against 16.2% with the idealised telescope.
  - **Detected only because of the boost:** 6.2% / 6.9% (idealised: 8.1%).
  - **Real clusters lost:** 4.9% / 5.4% (idealised: 9.0%).
- So Y boosts and losses are about halved (×0.45–0.64), but boosts over the threshold fall by only 15–23%.
- As a noise term, faint-halo SZ adds only 5–8% to the filter noise, in line with Raghunathan et al. (2022). Yet for each detected cluster, blending is still large.
- The filter ignores a uniform SZ background, so it needs no mean removal.
- These fractions are probably upper bounds. There are no foregrounds yet, and the simulated halos give 2.65× the SZ power that SPT measures.

### 5. With the SO filter, the sign of the bias depends on the analysis choices (step 2)
- These fits measure each cluster at its true position and size, with the neighbours added through the P(D), and fit noiseless expected counts.
- With step 2's default choices, config d is unbiased (−0.01σ / +0.01σ, SO-like). Config a is biased by +1.33σ / +1.85σ, the opposite sign to Q3.
  - The default choices are: a peak is credited to every cluster that is not outshone by a single neighbour (f_dom = 1), and the noise level is the noise-only value.
- With the relation fixed, the choices matter a lot:
  - the attribution rule (f_dom = 0.5–2) moves config a from −0.75σ to +3.15σ (baseline) and from −1.41σ to +4.72σ (goal);
  - with the default attribution rule, a noise level measured on the map turns it negative: −0.6σ to −4.4σ for the widths (i)–(iii) of step 2's model, and −0.3σ / −0.8σ for a 3σ-clipped width measured on the map check's maps. With f_dom = 2 and width (i) it stays positive (+0.9σ / +1.4σ).
- With the relation fitted, config d stays between −0.19σ and +0.61σ over all 20 combinations of choices computed (SO-like). This needs the slope α free. With α fixed and (1−b) free (configs b and c, as with a slope from external calibration), the default choices give +0.6σ to +1.2σ.
- A goodness-of-fit test has power of at most 0.073 in configs b–e, and 0.153 even in the worst analyst setup (SO-like; 0.05 means no power).

### 6. A real cluster finder makes its own choices, and they matter (step 3a)
- Our finder is a lean version of SZiFi's iterative matched filter. It ran on 324 simulated SO tiles of 100 deg² per noise level.
- **Attribution.** It credits each blended peak to its brighter cluster and never counts a peak twice. This is like step 2's rule with f_dom ≈ 0.7. Step 2's default rule credits both clusters in 25–69% of the peaks where a second cluster adds at least 10% of the signal.
- **Noise level.** After masking its detections, the finder's noise level is 1.02–1.07 times the noise-only value, because the SZ of the halos it does not mask stays in it.
- **The search adds blending.** Searching over position and size pulls peaks towards bright neighbours. Y is boosted by more than 10% for 10.6% (baseline) and 12.0% (goal) of real, non-fragment detections, against 8.3% / 9.3% for the same filter at the true positions.
- **Do not merge at 10′.** SZiFi's Planck scripts merge detections within 10′. At SO depth this throws away 7% (baseline) and 10–11% (goal) of real clusters. Merging within 1–2′ removes the fragments and only 0.1–0.2% of real clusters.
- **How the masked noise is estimated matters.** SZiFi's default settings were made for Planck: they fill the masked discs, blend them back towards the original map, and still divide the power above ℓ_sep = 3000 by the unmasked fraction. At SO resolution this overestimates the noise at the smallest filter size (0.5′) by 2.6% (baseline) and 3.4% (goal), even on maps without halos (our re-implementation; SZiFi itself gives 2.5% / 3.4% once its round-off problem is avoided). At 15′ it is about 1% low. Raising ℓ_sep to 12000, the value in SZiFi's SO example script, makes the noise about 2.5% low instead. The SO example's own settings compute the spectra only up to ℓ = 10000, so they never divide, and with an isotropic covariance they are unbiased on maps without halos.

### 7. Fitting the finder's own catalogue biases S8, even with the Y–M relation fitted (step 3b)
- **The analyst.** It models every cluster as isolated, but knows the finder's noise level and how the finder selects isolated clusters. That is the best a no-blending analysis can do.
- **What is fitted.** The clusters the finder credits: each real halo once, at the S/N of its highest credited detection and at its true redshift. Peaks with no real cluster behind them (1.75% / 1.50% of the q > 5 detections, about 80% of them confusion peaks of several faint halos) are left out, and the analyst's model has none either.
- **The deficit.** On 2,348 (baseline) and 1,948 (goal) tiles of 100 deg², the finder credits 1.46 ± 0.05% and 2.10 ± 0.05% fewer clusters at q > 5 than this analyst expects. Almost all of the loss is at q = 5–7.3.
- **Why.** The finder measures S/N against its own noise level, which still contains the SZ of unmasked halos. In these units a cluster's own CMB and instrument noise scatters by less than one, so fewer faint clusters scatter above q = 5 than the analyst expects. Neighbours give back only part of the loss.
- **The bias.** Config d is +0.32σ / +0.68σ (SO-like). The fit steepens the slope α to absorb the S/N-shaped loss, and that drags σ8 up. Config a is −2.33σ / −4.23σ, the same sign as Q3. With α fixed (configs b and c) it is −0.9σ to −2.6σ.
- **In S8 itself the relation-fixed shift is smaller.** It is −0.002 / −0.004, against +0.006 / +0.012 with the relation fitted (SO-like). It counts for more σ because that fit's error is about 18–20 times smaller.
- **A higher threshold reduces it.** At q > 6 the bias is +0.10σ / +0.23σ (config d) and −0.88σ / −1.56σ (config a).
- **Hard to detect.** A goodness-of-fit test has power of at most 0.21 in any configuration (SO-like).
- **The masked-noise estimate matters.** With our re-implementation of SZiFi's masking, config d is +0.31σ (its defaults with ℓ_sep = 12000) and −0.04 ± 0.15σ (its defaults; unsmoothed estimator) at baseline, against +0.32σ with our finder. At goal it is +0.90σ and +1.02σ, against +0.68σ. Config a moves more at goal: −4.2σ with our finder, −5.7σ and −8.0σ with the two SZiFi variants (baseline: −2.3σ, −2.7σ, −4.8σ).
- **Merging within 10′** removes another 7.4% / 11.2% of real clusters and pushes config d to +1.04σ / about +2.6σ to +2.8σ.
- **The peaks with no real cluster behind them matter.** Keeping them in the catalogue cancels most of the deficit (+0.51% / −0.54%) and moves config d to −0.04σ / +0.17σ (build's tiles, builder only).
- **The right no-blending reference is isolated clusters.** Virtual clusters added to halo-free maps and clusters painted into them give the same selection at q > 5.
- **A simple analytic selection function can do more harm than blending.** Consider Φ(√(q′² + k) − 5), with q′ a cluster's true Y over the finder's noise level, set to zero below q′ = 2. It sees almost the same blending effect (within 0.23σ in configs a and d). But in config a at baseline its own error is +15.6σ for k = 0, +1.8σ for k = 2 and −6.3σ for k = 3, against about −2σ from blending itself. Without the zero below q′ = 2, the k = 2 and 3 fits break down.

### 8. SZiFi's own code confirms our finder before masking, and our re-implementation of its masking once a round-off problem in SZiFi is avoided
- SZiFi's unchanged code ran on 30 of step 3a's tiles per noise level.
- **Before masking.** SZiFi finds the same objects as our finder: 99.97–99.99% are matched within 1′.
- **Default masking.** On 12 tiles per noise level, SZiFi's noise level after its default masking, in units of the noise-only value, ends within 0.006 of step 3a's re-implementation. This holds once two things are dealt with:
  - **the round-off problem**, avoided with an ℓ range that leaves none of SZiFi's spectral bins wholly outside it, or with SZiFi's inverse covariance set to zero outside the range;
  - **the masked discs at our test tiles' edges**, which have to be filled periodically.

  Then SZiFi finds 22% (baseline) and 24% (goal) fewer q > 5 clusters than our finder. The re-implementation finds 20% / 24% fewer on the same tiles.
- **ℓ_sep = 12000.** With SZiFi's defaults but the SO example script's ℓ_sep, SZiFi counts the same q > 5 clusters as our finder within 2%.
- **The SO example's own settings: its covariance ('mixed_gaussian') breaks down on our test tiles.** With an isotropic covariance the example works and finds 3.5% / 7.7% fewer q > 5 clusters than our finder.
- **Seven possible bugs in SZiFi** turned up along the way, the round-off problem among them. They are written up in [`SZIFI_REVIEW.md`](SZIFI_REVIEW.md).
- **What this supports.** SZiFi itself supports the masked noise levels behind step 3b's two SZiFi variants. The count change and the S8 fits were not recomputed with SZiFi itself.

## What this means for an SO cluster-count analysis

- **Blending belongs in the counts model.** An analysis that fits the clusters our finder credits, knows the finder's noise level and selection, and fits the Y–M relation, is still biased (+0.32σ / +0.68σ in config d, SO-like). With the relation fixed the bias is more significant, and a goodness-of-fit test would rarely reveal either. How the peaks with no real cluster behind them are handled matters as much: kept in the catalogue, they cancel most of the deficit.
- **The model has to be calibrated on the pipeline actually used.**
  - The finder's attribution and noise level decide even the sign of the relation-fixed bias: step 2's default analysis gives +1.33σ, the finder's own catalogue −2.33σ (baseline).
  - Its masked-noise estimate and merging radius change the size of the bias.
  - Clusters injected into simulated maps with a realistic halo population, and into their halo-free versions, then run through the real pipeline, can measure this. Isolated clusters on the halo-free maps are the right no-blending reference.
  - Injecting into the real maps instead folds part of the confusion into the selection.
- **The finder's settings matter for SO.**
  - Do not merge at 10′; 1–2′ is enough.
  - SZiFi's default masked-noise estimate, set for Planck, is biased high at small filter sizes at SO resolution. Raising ℓ_sep to 12000 makes it biased low.
  - SZiFi's ℓ range must leave none of its spectral bins wholly outside it, or SZiFi amplifies round-off ([`SZIFI_REVIEW.md`](SZIFI_REVIEW.md)). In practice it must reach the top of the map's multipoles.
- **The selection function needs as much care as blending.** A simple analytic one can bias S8 more than blending does.
- **A higher threshold reduces the bias.** At q > 6 the step-3b bias is +0.10σ / +0.23σ (config d) and −0.88σ / −1.56σ (config a), at the cost of fewer clusters.
- **A cheap blending model exists for the idealised telescope, not yet for a real finder.** For the idealised telescope a P(D) table takes 0.5–2.5 minutes, and a correction computed at the fiducial cosmology and once more at the first best fit leaves ≤ 0.05σ. With the SO filter at the true positions, a correction computed once leaves ≤ 0.17σ, if the blending model is right. For a real finder, step 3b measured the blending effect on simulated tiles at the fiducial cosmology only.

## How sure we are

- **How the checks work.** Each realistic step was built once, then checked by two independent re-implementations, and an arbiter tested every disputed claim; each `RESULTS.md` says, claim by claim, which check confirmed it. For Q1–Q3, disputed numbers were re-run on fresh random seeds.
- **Checked by one reviewer only** (from [`PROGRESS.md`](PROGRESS.md)):
  - step 2: the arbiter's extra analyst setups, and the map check's Poisson realisations;
  - step 3a: the count-change preview with its selection calibration, and the class fractions and count change with SZiFi's default settings;
  - step 3b: the final headline on all tiles and the arbiter's targeted tests. For the report, its q > 5 biases in configs a, d and e were refitted with the fits check's code. Also two sets of fits that only the builder made and nobody checked: those with our re-implementation of SZiFi's masking with ℓ_sep = 12000, and those with the peaks that have no real cluster behind them kept;
  - SZiFi itself: two sets of runs that only the arbiter made, the periodic edge filling and ℓ_sep = 12000 with the round-off removed; their numbers were recomputed from the stored runs with our own code. Also the re-implementation's q > 5 counts on the same tiles, and the run with SZiFi's own four SO channels.
- **Main open differences** (the full list is in [`PROGRESS.md`](PROGRESS.md)):
  - **Step 3b tile samples.** At baseline the builder's 1,548 tiles and the 800 made with the sims check's generators differ by 2.6σ (−1.36% against −1.64%), for an unknown reason. If the difference is real, the baseline biases carry about ±0.05σ (config d) and ±0.25σ (config a) more.
  - **Q3 maps.** The simulated maps show 0.77 ± 0.11 of the count deficit the P(D) model predicts, so Q3's biases may be about 20% smaller.
  - **SZiFi at goal.** With the round-off problem removed and our test tiles' edges filled periodically, SZiFi's final noise level at 0.5′ is within 0.002 of the re-implementation. But that is only because its sharper template already adds +0.010 before masking: the masking step itself raises the noise level 0.012 ± 0.003 less. At 15′, SZiFi ends 0.003–0.004 higher than its template explains, at both noise levels. Neither difference was traced.
  - **SZiFi on its own setup.** The edge effect of our periodic test tiles has a tail (up to +0.017 in r on one tile). SZiFi's own apodised tiles, and whether its round-off problem affects its Planck runs, were not tested.

## What is not included yet

- **Sky.**
  - No foregrounds (CIB, radio sources, kSZ, Galactic emission).
  - Halos are placed independently (Poisson). In the idealised robustness test, clustering raised contamination and boosting by about 10% (×1.1). We treat this as second order and leave clustered halos (Websky) out of this paper.
  - The simulated halos give 2.65× the SZ power that SPT measures. With a fainter population matching SPT, step 2's config a drops from +1.33σ to +0.57σ (SO-like, baseline).
- **Instrument and maps.**
  - No survey masks, bandpasses or relativistic SZ.
  - The SO map simulations use flat, periodic 10° × 10° tiles.
- **Analysis.**
  - The fits use noiseless expected counts. Q3 cross-checked its fits with Poisson realisations. Among the realistic steps only one case was tested this way (step 2, SO-like, baseline, default choices), where the shift from blending matched the noiseless fit; steps 3a and 3b were not tested this way.
  - The redshift evolution of the Y–M relation is not freed in steps 2–3b.
  - In config d, mass calibration enters only as a 4% prior on (1−b).
  - The analyst knows the finder's selection of isolated clusters exactly.

## Open questions and next steps

- **Open:**
  - which masked-noise estimate an SO pipeline will use (with SZiFi's defaults it moves config d by about 0.35σ, down at baseline and up at goal);
  - how to model the peaks with no real cluster behind them, which cancel most of the deficit if they are kept;
  - the count change and S8 with SZiFi itself, which need a selection calibrated with SZiFi (about 100 s per tile run);
  - the step-3b baseline tile-sample difference;
  - whether SZiFi's 'mixed_gaussian' covariance works on its own apodised SO tiles.
- **Next:**
  - foregrounds (CIB, radio, kSZ);
  - the P(D) inside a full cluster-count likelihood code;
  - a paper draft.
