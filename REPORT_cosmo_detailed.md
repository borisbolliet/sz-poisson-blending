# Blending in the cluster-count likelihood: full technical results

Raw output of the verified cosmology analysis (three analyses, two verifiers each, then repaired). Readable summary: Section 6 of `REPORT.md`.

```text
BLENDING IN SZ CLUSTER-COUNT LIKELIHOODS: CAN IT BE MODELLED, AND DOES IT BIAS COSMOLOGY?

Terms used below:
- q is the matched-filter signal-to-noise. Detections have q_obs > 5.
- The "naive" likelihood is the standard one: each cluster is seen in isolation, with Gaussian unit noise.
- The "blended" likelihood adds the confusion from all other halos (M > 1e13).
- P(D) is the probability distribution of that confusion signal.
- Bias is quoted as (best fit − truth) / 1σ, where σ is the statistical error of the correct analysis with the same free parameters.
- All fits are Asimov fits: noise-free expected counts, with the naive model fitted to blended "data".
- "Full sky" is 41,000 deg². "40% sky" is about 16,500 deg², SO-like.
- h, Ωb and ns are fixed throughout, and the mass function is Tinker08 with no nuisance parameter.

Three independent analyses (likelihood-model, cosmo-bias, cosmo-dependence) were each checked by two verifiers and then repaired. Only confirmed or re-supported numbers are used here.

------------------------------------------------------------
1. SHORT ANSWER
------------------------------------------------------------
- **Can it be modelled? Yes, exactly, at modest cost.**
  - Blending is a convolution of the usual selection kernel with the compound-Poisson P(D) of neighbours.
  - The two essential ingredients are the "one detection per peak" (dominance) rule and the map-mean subtraction. Leaving out the dominance rule gives +85% counts; leaving out the mean subtraction gives +36%.
  - The kernel takes 26–150 s per cosmology, depending on the implementation. A quadratic emulator or a single recomputation at the best fit makes cosmology dependence a non-issue (≤0.05σ).
  - The model reproduces catalogue Monte Carlos statistically, and the map simulations to within about 20% of the size of the effect (see caveats).
- **The effect on counts is small in total but strongly shaped.** Blending changes the total q>5 counts by −1.6%. Per bin it ranges from −12% (z<0.1, 5<q<6) to +3–5% (z>2), which mimics a change of cosmology.
- **Bias if blending is ignored depends mostly on how freely the Y–M scaling relation is fitted:**
  - **Scaling relation fixed:** Ωm about −5.8σ and σ8 about +4.7σ on the full sky; about −3.7σ and +3.0σ at 40% sky. All three analyses agree.
  - **Mass bias (1−b) free, 5% prior:** Ωm −2.0σ (full) and −1.4σ (40%); σ8 about +0.5σ.
  - **All four scaling-relation parameters free under Planck-like priors:** σ8 +1.8σ, S8 +2.1–2.2σ (full); σ8 +0.9σ, S8 +1.1σ (40%).
  - **Without blending modelled, a realistic SO-like 40% survey should expect a silent bias of about 1σ in σ8/S8 (or Ωm).** On the full sky it is about 2σ.
  - **The free nuisances do not absorb the bias.** The absolute shifts stay as large or grow. The bias looks smaller in σ units only because the errors grow 4–22×. With all parameters taken together, the truth is still excluded at about 5.4σ (full sky).
- **Would the misfit be noticed? Mostly no.**
  - A 95% goodness-of-fit test flags it in about half of full-sky realisations with the scaling relation fixed or nearly fixed (power 0.40–0.54).
  - It flags it only 7–29% of the time at 40% sky, or once the scaling relation has free nuisances.
- **The bigger danger is the detection pipeline.** Map peak-finding (maximum over filter scales and position, fragments, noise peaks) adds +4–6% to the counts, more than blending does. If fragments and noise peaks are kept, the resulting cosmology bias is large (S8 up to ±4σ at 40% sky) but detectable. This must be calibrated by injecting clusters into simulated maps whatever is done about blending.

------------------------------------------------------------
2. HOW TO INCLUDE BLENDING IN THE LIKELIHOOD
------------------------------------------------------------
**The model.** In the standard count model,
dN/dz dq_obs = ∫ dlnM dN/dz dlnM ∫ dlnY p(lnY|M,z) K(q_obs|q).
Normally K is a unit Gaussian. With blending it becomes:
- **Observed signal:** q_obs = q_A + n + δ − μ(θ_A).
  - δ is the summed signal of all neighbours seen through the cluster's filter. It is a compound-Poisson variable, and its rate follows from the same halo mass function and scaling relation.
  - μ is the map-mean subtraction.
- **Dominance rule:** each peak is counted once, under its brightest contributor. This truncates δ at neighbours fainter than the primary and multiplies by P_dom = exp(−Λ(x ≥ q_A)).
- **Shape:** the kernel depends on the filter scale θ500(M,z). The result is a positively skewed extra scatter with a negative median, and it is mass- and redshift-dependent. It is wide and shifted negative at low z (large θ, large μ) and has a positive tail at high z.
- **Clusters boosted over threshold** come in automatically, because the integral runs down to q_A ≈ 0.
- **Noise-only peaks** add a small cosmology-independent term: +1.05% in the 5<q<6 bin, 1.6σ on the full sky. This term was not in the cosmology fits.

**Validation.**
- Full-sky catalogue Monte Carlo (8 skies), q bins: χ² = 11.2/10. (z,q) bins: 93.3/120.
- Paired blended-minus-naive difference: −1450±36 (Monte Carlo) vs −1400 (model).
- An independent direct Monte Carlo reproduces the blending fractions at σ8 = 0.75, 0.81 and 0.87 within 3 Monte Carlo σ.

**Cost and cosmology dependence.**
- One kernel table costs about 150 s on 3 threads in the high-resolution implementation, or about 26–30 s on 1 core in the fast one. A count evaluation then takes about 0.1 s.
- The kernel depends on the same parameters being fitted. Mean y scales as σ8^5.0 Ωm^2.3 (1−b)^1.8, and the confusion variance as σ8^8.9.
- A kernel held fixed at a fiducial cosmology is almost good enough:
  - Over Planck18 ±3σ it leaves count errors of about 0.5%.
  - The resulting biases are 0.5–1.6σ with the scaling relation fixed, ≤0.6σ with nuisances under priors, and ≤0.5σ without priors.
  - At the grid edges (σ8 = 0.87, Ωm = 0.35) they grow to 2–4σ with the scaling relation fixed.
  - In a separate test, a kernel computed at a fiducial off by Δσ8 = 0.03 biases a fixed-scaling-relation fit by −2.2σ in Ωm and +1.5σ in σ8.
- Two cheap remedies, both verified:
  - **Recompute once at the first best fit and refit:** residual bias ≤0.05σ over Planck ±3σ, ≤0.22σ even at the grid edges.
  - **Quadratic emulator of ln C(z,q) on a 3^4 = 81-point grid:** costs about 41 core-minutes once and leaves ≤0.02σ bias.
- Recomputing the kernel at every likelihood call is not necessary.

**Cheap approximations: when they fail and when they work.**
- **One-number or fixed-shape corrections do not reproduce the blended counts.** A constant Gaussian bias+scatter, a re-fitted lognormal (log A, σlnY), a skew-normal, or a Gaussianised P(D) all leave Δχ² = 34–46 on the full sky (14–18 at 40% sky), against 72.5 for doing nothing.
- **A 4-parameter Gaussian whose bias and variance vary with ln θ500 does reproduce them:** q + b0 + b1 L + N(0, 1 + v0 + v1 L), with L = ln(θ/2'). Δχ² = 1.1, max pull 0.6.
  - It works only if the total variance may drop below the unit noise variance at large θ, so it is a phenomenological description, not a physical scatter model.
  - An earlier conclusion that it fails (Δχ² = 17.7) came from a parameterisation that forced the variance to stay ≥ 1. That conclusion was refuted.
- **Used as free nuisance parameters with no priors**, these 4 confusion parameters:
  - reduce the Ωm bias to about 0.5σ of their own (wider) errors;
  - leave +0.7σ relative to a correct P(D) analysis when (1−b) or the full scaling relation is free, and +2.5σ when it is fixed;
  - widen σ(Ωm) by about 40% (×3.5 with the scaling relation fixed).
- **Shifting the (1−b) prior centre does not help.** Centring it where a weak-lensing calibration of the blended observable would put it (0.7896 instead of 0.80, −0.26 prior σ) does not remove the bias. S2 S8cl goes from +2.21σ to +2.42σ on the full sky.

------------------------------------------------------------
3. RESULTS TABLE: BIAS IF BLENDING IS IGNORED (q_obs > 5)
------------------------------------------------------------
Entries are bias in σ for (Ωm, σ8, S8). Every parameter not listed as free is fixed at its true value.

Two S8 definitions are in use:
- S8 = σ8(Ωm/0.3)^0.5 in the cosmo-bias analysis.
- S8cl = σ8(Ωm/0.3)^0.3 in the likelihood-model analysis.

With the scaling relation fixed, the shift lies almost along the cluster-count degeneracy. So S8 (0.5) is biased by −6.3σ while S8cl (0.3) is biased by only −0.4σ.

| Free parameters (prior) | Full sky | 40% sky | Source |
|---|---|---|---|
| Ωm, σ8 only (scaling relation fixed) | −5.8, +4.7; S8 −6.3; S8cl −0.4 | −3.7, +3.0; S8 −4.0; S8cl −0.25 | all three agree to ≤0.3σ |
| + (1−b) (prior ±0.04 about 0.80) | −2.0, +0.5; S8cl −0.5 | −1.4, +0.5; S8cl −0.3 | likelihood-model (c05b) |
| + (1−b) (no prior) | −1.8, +0.3; S8 −0.8 | −1.2, +0.2; S8 −0.5 | cosmo-bias (c12) |
| + (1−b), σlnY (priors 0.04, 0.05) | −1.7, +0.35 | −1.3, +0.4 | cosmo-dependence (r02) |
| + (1−b), σlnY (no priors) | −1.4, +0.05 | −0.9, +0.03 | cosmo-dependence (r02) |
| + (1−b), α, β_E, σlnY (Planck-like priors ±0.04, ±0.08, ±0.5, ±0.023) | +1.2, +1.8; S8cl +2.2 (profile likelihood: +1.1, +1.7) | +0.4, +0.9; S8cl +1.1 | likelihood-model (c05b) |
| + (1−b) (4% prior), α, σlnY (no priors on α, σlnY) | +0.9, +3.1; S8 +2.1 | +0.2, +1.8; S8 +1.1 (emcee σ8 +1.5, S8 +0.9; 50 Poisson realisations S8 +1.0±0.13) | cosmo-bias (c12, c14, c13) |
| same, no prior on (1−b) | +2.0, +3.5; S8 +2.7 | +1.25, +2.2; S8 +1.7 | cosmo-bias (c12) |
| (1−b) case or full Planck-like case, + 4 free confusion params | Ωm +0.7σ (vs correct analysis), +0.5σ (own errors); σ8 −0.2 to −0.3 | Ωm +0.4 to +0.5σ | likelihood-model (c05b) |

What the nuisances absorb:
- **(1−b)**
  - It moves by only +0.012 to +0.019 (about 0.5σ). That matches the shift implied by the 3.4% mean Y overestimate of detected clusters (+0.015).
  - It takes up the amplitude but not the (z,q) shape. The fit χ² falls only from 24.3 to 23.7–23.9, against 62.6 at the truth.
  - The absolute S8 shift grows when (1−b) is freed: −0.0037 → −0.0107 (40% sky).
  - Δ(−2lnL) between the truth and the best fit stays 31–39 on the full sky (about 5.4σ for 4 parameters jointly) and 12–15 at 40% sky (about 2.9σ).
- **α and σlnY**
  - The slope α rises by about +0.026 (+2σ), and σlnY is pushed to its physical bound of 0.
  - Tilting the mass dependence mimics the redshift tilt of blending, and in doing so drives σ8 up.
  - This is why the sign of the Ωm/σ8 bias changes between the "(1−b) only" and "full scaling relation" rows.
- **Raising the threshold** to q_cut = 6 roughly halves all biases (40% sky, catalogue: S8 −2.1σ fixed relation, +0.65σ with α free).

**Map-based check.** The same fits were run with the "truth" measured from 32,400 deg² of map simulations, with fragments and noise peaks removed. They give the same sign and pattern, about 20–30% smaller:
- Fixed scaling relation: S8 −2.9±0.4σ against −4.0σ from the catalogue, at 40% sky.
- (1−b) 4% prior with α and σlnY free: S8 +0.93±0.14σ against +1.1σ.

------------------------------------------------------------
UNRESOLVED DISAGREEMENTS (flagged)
------------------------------------------------------------
- **Size of the blending effect in maps.** A noise-paired map test measures 0.77±0.11 of the model's predicted blending deficit (2.05σ low), mostly in the shift/scatter part; the dominance part agrees at 0.94±0.05. The total blending change is −0.8% in the cleaned maps against −1.6% in the catalogue model. Possible causes are pixelisation, patch-mean vs global-mean removal, or profile truncation. If the shortfall is real, the bias numbers above are about 20% (up to about 2×) too large.
- **Highest-q bin (q>50).** The pipeline gives C = 0.990–0.996, while an independent Monte Carlo gives 1.006–1.011. This is unresolved. It moves σ8 biases in the free-nuisance cases by 0.2–0.3σ, so σ8 biases there are not physically accurate below about 0.2σ.
- **Method uncertainty with all scaling-relation parameters free.** Fisher, profile-likelihood and MCMC errors differ by about ±0.2σ in the quoted bias. Where σlnY sits on its bound, the ML bias overstates the posterior shift: S8 +1.11 (ML) vs +0.87 (emcee) at 40% sky.
- **Superseded results.** Earlier, larger numbers came from an under-resolved count model and an optimiser that accepted uphill steps. Examples: config d S8 +1.8σ and config e S8 +4.5σ at 40% sky (corrected to +1.1σ and +1.7σ above), and "(1−b) absorbs the bias". These are replaced by the v2 results above.

------------------------------------------------------------
4. CAVEATS: WHAT THIS TOY SETUP LACKS
------------------------------------------------------------
- **No CMB, no foregrounds, white noise only.** Primary CMB, CIB and radio sources would add their own (partly correlated) confusion to the matched filter. Compensated filters were not tested.
- **Real pipelines.**
  - The cosmology fits use a catalogue-level filter at the true position and true θ500.
  - Real peak-finding (maximum over scales and position, merging, deblending, masks, edges) changes the counts by +4–6%. This is more than blending and of the opposite sign.
  - A 2-parameter ad hoc pipeline term lets the blended model fit the map counts (χ² = 125.7/109; naive 140.5/109). Residuals remain at z 2–3.
  - Its cosmology dependence was not computed.
  - Masks and edges are not modelled; 40% sky is a simple 0.4 rescaling of the counts.
- **Mass calibration.**
  - Priors are Gaussian, Planck-like and centred on the truth (one crude test shifted the centre).
  - No simulated weak-lensing calibration on the blended observable was done.
  - h, Ωb, ns and the mass function are fixed.
  - The bias in σ units depends strongly on these choices. The fixed-relation case is an extreme bound.
- **Clustering.** The 2-halo term is not included (earlier work: blending fractions about ×1.1, with extra cosmology dependence through halo bias).
- **Redshifts.** Each detection takes its brightest contributor's redshift. Counterpart misidentification is not modelled.
- **Noise-only peaks** are excluded from the cosmology fits (+1% of the lowest-q bin).
- **Statistics.** All fits are Asimov, so biases are deterministic shifts. Poisson realisations reproduce them where tested (fixed relation: S8 −3.99±0.09σ against −3.97σ Asimov).

------------------------------------------------------------
5. FIGURES AND KEY SCRIPTS
------------------------------------------------------------
Figures (all confirmed to exist with ls). Paths are relative to /Users/boris/GitHub/sz-poisson-blending/figures/cosmo/:
- blended_vs_naive_zq.png: blended/naive − 1 in (z,q)
- kernel.png: blending kernel vs θ
- map_vs_model.png: map counts vs model
- cosmo_bias_v2.png: likelihood-model biases, current (cosmo_bias.png is superseded)
- bias_bars_v2.png: cosmo-bias biases per configuration, current (bias_bars.png is superseded)
- mcmc_contours_v2.png: current (mcmc_contours.png is superseded)
- count_residuals.png: catalogue and map count changes vs z
- blend_dependence_params.png: cosmology scaling of the confusion signal
- blend_dependence_Cq.png: C(q) as cosmology varies

Key scripts. Paths are relative to /Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/:
- **Likelihood model:**
  - likelihood/cclib.py: kernel and count library
  - likelihood/c02_counts_compare.py
  - likelihood/c03d_map_paired.py
  - likelihood/c04b_effective_E5E6.py
  - likelihood/c05b_cosmo_bias_v2.py
  - likelihood/c06_pd_cosmology.py
  - likelihood/c08_prior_centre.py
- **Cosmology bias:**
  - bias/cosmolib2.py: converged count model
  - bias/c12_fits_v2.py
  - bias/c13_poisson_v2.py
  - bias/c14_mcmc_v2.py
  - bias/c15_table_v2.py: full table in c15_table_v2.log
  - bias/c16_boot_ml_v2.py
- **Cosmology dependence:**
  - blend_dependence/cosmolib.py
  - blend_dependence/c02_grid.py
  - blend_dependence/c03_analysis.py
  - blend_repair/replib.py
  - blend_repair/r02_fits.py
  - blend_repair/r03_report.py
  - blend_repair/r06_joint.py
  - blend_repair/r07_mc_compare.py

Data products are in /Users/boris/GitHub/sz-poisson-blending/data_cosmo/: fits_v2_final.json, r02_fits.json, cosmo_bias v2 outputs, and others. The analyses report that nothing in data/ was modified.
```
