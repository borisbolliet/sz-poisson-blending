# Review of SZiFi: possible bugs and areas of improvement (12 November 2026)

[SZiFi](https://github.com/inigozubeldia/szifi) is the iterative multi-frequency matched-filter cluster finder of Zubeldia et al. (2023a,b), and the finder being tested for the Simons Observatory (SO). We ran its unchanged code on simulated SO tiles to test the cluster finder used in our blending study ([`CONCLUSIONS.md`](CONCLUSIONS.md)). This page collects what we learned about SZiFi itself:
- seven possible bugs;
- smaller issues with documentation, defaults and packaging;
- where the method and the speed could be improved at SO resolution.

It is written so that SZiFi's authors, or SO users of SZiFi, can check each point. **None of it has been reported to SZiFi's authors yet.**

- **Code reviewed:** SZiFi at commit `73626fb` (28 August 2026), run unchanged.
- **How the findings were made.** Details are in [`analysis/realistic/step3_szifi_real/RESULTS.md`](analysis/realistic/step3_szifi_real/RESULTS.md).
  - A build ran SZiFi on 30 tiles per noise level.
  - Two checks, each with its own survey file and driver, reproduced and extended the build.
  - An arbiter settled the disputed points with targeted runs.
  - For this review we also:
    - read the code at every location cited below;
    - profiled one run;
    - ran SZiFi to test two points (D1, E6);
    - tried a fix for E5.

    A script ([`p03`](analysis/realistic/szifi_review/p03_code_sites.log)) checks that each cited line holds the expected identifier. It confirms the location, not what the code does there. The scripts and logs are in [`analysis/realistic/szifi_review/`](analysis/realistic/szifi_review/).
- **Independent review of this page.** A first version was checked by independent reviewers against the logs and SZiFi's source, and a corrected version once more. Their corrections are included. They found E6, and traced E7, an offset the config check had measured, to a bug.
- **Environment:** Python 3.12.9, numpy 2.5.4, scipy 1.18.1, healpy 1.20.1, astropy 8.0.2, numba 0.68.1, scikit-learn 1.9.1, pymaster (NaMaster) 3.0.1, pixell 0.32.9, orphics (git d062f39), setuptools 80.10.2. A 10-core macOS laptop, one thread per SZiFi run.
- **SZiFi's code is not reproduced here.** The repository has no licence, so we only import and call it. Code locations are given as `file:line` at commit `73626fb`. Fixes are described in words.

## What works

- **Same detections as an independent implementation.** Before masking, SZiFi's catalogue on our tiles matches our own implementation of the same filter. 99.97–99.99% of objects are matched within 1′ at any filter scale (85–89% also within one scale step), and their S/N differs by +0.01 ± 0.20–0.25.
- **Correct response to injected clusters.** Clusters of known Y were injected into maps without halos (195 of 196 found; SZiFi's default iteration settings in step 3a's ℓ range).
  - At the injected scale, y0 is recovered to 1.003 ± 0.008.
  - The S/N lies within −0.05 ± 0.08 of the value expected from SZiFi's own σ.
  - This tests the response to signal, not σ itself, which is biased ([A1](#the-method-at-so-resolution)).
  - At θ500 = 4.5–9′ the S/N is 0.25 ± 0.12 low.
  - The injected clusters sat on pixel corners, where the shift of [E7](#e7-positions-are-shifted-by-half-a-pixel) puts the S/N peak exactly on a pixel.
- **The merging tools agree with ours.** Once adapted to current packages (E3), SZiFi's 'closest' and 'fof' merging keep the same detections as our own code in 438 of 480 tile runs. The others differ only for pairs within 0.4% of the merging radius (flat against spherical distance).
- **The SO example's settings work with an isotropic covariance.** The noise level on maps without halos is unbiased (0.993–1.003 of the true value, 3 tiles per noise level).

## Summary

| # | finding | effect | found by | severity (our view) |
|---|---|---|---|---|
| [E4](#e4-round-off-is-amplified-outside-the-ℓ-range) | an ℓ range that leaves whole spectral bins outside it makes SZiFi amplify round-off | silently changes the catalogues; with the default ℓ_sep = 3000, raises the noise level after masking by 2.1–2.6% at 0.5′ on our tiles | config check; arbiter | high |
| [E5](#e5-the-mixed_gaussian-covariance-is-not-positive-definite) | the 'mixed_gaussian' covariance, the SO example's choice, is not positive definite on our tiles *(our set-up)* | the run breaks down: from no detection (σ = NaN) to 133,309 peaks per tile | build; both checks | high where it is used |
| [E6](#e6-more-than-one-masking-iteration-never-runs) | more than one masking iteration never runs: the convergence test is always true after the second pass | `max_it` ≥ 2 silently has no effect | independent review of this page (by reading); confirmed by a run (p07) | low to medium |
| [E3](#e3-merging-fails-with-current-numpy-and-scikit-learn) | 'closest' and 'fof' merging fail with current numpy and scikit-learn | merging cannot run | build; independent check | medium |
| [E2](#e2-a-misspelled-key-in-the-default-parameters) | the default dictionary misspells 'snr_weighting' | a run with the defaults stops; by reading, so do SZiFi's four Planck example scripts | build; this review | low to medium |
| [E7](#e7-positions-are-shifted-by-half-a-pixel) | positions are shifted by half a pixel | catalogue positions half a pixel off along both axes (0.25′ on our tiles, 0.43′ on SZiFi's Planck tiles) | config check (measured); independent review of this page (traced to a bug) | low |
| [E1](#e1-σ-at-the-first-filter-scale-is-reported-as-0) | σ at the first filter scale is reported as 0 with 'maxima_lomem' and more than three scales | wrong σ in the output; catalogues unaffected | build | low |
| [D1–D6](#documentation-and-defaults) | documentation and defaults | lrange = None crashes; unknown covariance types stop without a message; `import szifi` switches off all warnings; the author's paths and template files in the defaults; inpaint = False makes the iteration do nothing | build; this review | low |
| [A1–A2](#the-method-at-so-resolution) | the method at SO resolution | the masked-noise estimate is biased either way at SO resolution; 10′ merging loses real clusters | step 3a; arbiter | medium (affects cosmology) |
| [A3](#the-method-at-so-resolution) | periodic simulated tiles *(our set-up)* | spurious edge peaks; holes at the edge filled from one side | build; independent check | – (not a bug for real data) |
| [S1–S6](#speed-and-memory) | speed | 82–111 s per 100 deg² tile run on one thread; about three quarters of it in the filtering; 1,692 2D FFTs (numpy and scipy) per run | this review | – |
| [P1–P5](#packaging) | packaging | no licence; no declared dependencies; "optional" packages needed at import; `pkg_resources`; `szifi.utils` is pixell's | this review | low to medium |

## Scope: what we tested, and what we did not

- **Our tiles.**
  - 10° × 10° flat, periodic squares of 1200 × 1200 pixels of 0.5′.
  - The six SO large-aperture-telescope channels (27–280 GHz), SO baseline and goal noise including atmosphere, the lensed CMB, and Poisson halos above 10¹³ M☉ with Arnaud profiles.
  - No foregrounds.
  - All masks are ones, except that peak finding and selection leave out an 8′ edge buffer.
- **SZiFi's own setting is different.** It cuts HEALPix or CAR tiles from a larger sky map, with apodised masks and catalogue-level tiling. Findings that may depend on this are marked *(our set-up)*.
- **Settings tested:**
  - SZiFi's defaults for the noise iteration (ℓ_sep = 3000, diffusive inpainting), with 15 filter scales from 0.5′ to 15′;
  - the same with ℓ_sep = 12000;
  - its SO example script's settings where they apply to flat tiles.

  Each was run with step 3a's ℓ range, [100, 20000], or with an ℓ range that leaves no spectral bin out. Full table: [`RESULTS.md`, Configurations](analysis/realistic/step3_szifi_real/RESULTS.md#configurations).
- **Not tested:**
  - SZiFi on its own CAR tiles for SO, or on Planck maps;
  - the spectrally constrained filter;
  - point-source finding;
  - float64 maps, beyond one diagnostic run.

  More than one masking iteration cannot run (E6).

## Possible bugs

### E4. Round-off is amplified outside the ℓ range

**What happens.** On our grid, `lrange` = [100, 20000] leaves 32.7% of the Fourier modes outside the range. It also leaves 22 of SZiFi's 299 spectral bins wholly outside it.
1. SZiFi zeroes the Fourier modes outside `lrange` in every map (`szifi/mmf.py:179–186`).
2. It still bins the spectra over the whole grid, up to the Nyquist ℓ (`szifi/spec.py:24–25`), and gives the corner modes the value of the nearest bin (`szifi/spec.py:398`). A bin wholly above ℓ = 20,000 therefore holds only float32 round-off, 10⁻¹⁶–10⁻¹⁴ of the in-band level.
3. It inverts the covariance at every mode (`szifi/spec.py:317`). Above ℓ = 20,000 the inverse is 8×10¹³ times its value at ℓ = 15,000–20,000 (145 GHz).
4. The cluster templates are band-limited by an FFT round trip (`szifi/mmf.py:664–667`), so out of band they are zero only up to round-off (2×10⁻³² of the in-band energy). Multiplied by that inverse covariance, the filters' out-of-band energy is 7×10⁻⁴ (0.5′) to 0.2 (15′) times their in-band energy.
5. This part of each filter is round-off noise. It enters SZiFi's zero-padded convolution (`szifi/mmf.py:1155`), which mixes Fourier modes, and moves q by up to 0.2. A periodic convolution would pair it only with the maps' own out-of-band round-off. That last point is our inference; we did not test it.

The same binning also dilutes the two bins that straddle the edges of `lrange`, by 0.67 and 0.78 on our grid, because each bin is averaged over all its modes, including the zeroed ones.

**Effect on our tiles** (12 tiles per noise level; values without the problem in brackets):
- **Catalogues change before masking.** A third (baseline) and a quarter (goal) of SZiFi's (pixel, scale) entries have no identical entry once the problem is removed, mostly through a different scale or extra maxima along the scale axis. The objects are the same: 99.96–100% still match our finder within 1′. q changes by 0.01 rms (at most 0.2), and σ is unchanged (≤ 3×10⁻⁸).
- **Spurious maxima at large scales.** At baseline there are 23.6 entries per tile at θ ≥ 7.24′ (15.4; our finder has 14.2); at goal 23.1 (15.0; finder 16.5).
- **More of the tile masked, so more noise.** Masking these entries leaves 74% of the tile unmasked (78%) at baseline. With the default ℓ_sep = 3000, SZiFi divides the small-scale power by that fraction. So its final noise level at 0.5′ ends 2.1% (baseline) and 2.6% (goal) higher (about 0.1% at 15′), and the q > 5 count falls by 8.1% through the iteration (0.9%) at baseline. With ℓ_sep = 12000 (7 tiles per noise level) the problem works the other way, and weakly: it lowers the noise level by 0.2% at 0.5′ and 0.4–0.5% at 15′.
- **Results depend on the last bit of the inputs** (4 tile runs). Changing 6 of the 15 scale values by at most 5.6×10⁻¹⁶ had two effects:
  - before masking, it moved q by up to 0.1 and moved 6% of the detections to another scale;
  - through the iteration, it changed the unmasked fraction by up to 0.024, σ at 0.64′ by up to 1.4% and the final q > 5 count by up to 2.7% per tile.

  Without the problem, the same change left both iterations' catalogues identical (one tile).
- **Float64 maps.** In one diagnostic run with float64 maps and the same ℓ range, SZiFi made 2.1 million detections. Its bins above ℓ = 20,016 held 10⁻³¹–10⁻³⁰ of the in-band power, so we attribute this to the same mechanism. We did not run float64 maps with no bin out of band.

**Two independent ways around it agree:**
- an ℓ range that leaves no spectral bin wholly outside it ([100, 30600] on our grid);
- SZiFi's inverse covariance set to zero outside the range.

On 6 tiles per noise level they agree to 3×10⁻⁴ in r, the noise level in units of its value for the true CMB and noise, at 0.5′, 4.45′ and 15′. Their q > 5 counts, in the interior at least 9′ from the edges, are 939 against 941 (baseline) and 1,340 against 1,340 (goal).

**Does it affect SZiFi's own runs?**
- **SO example script: no.** It sets `lrange` = [1, 300000] (`test_files/run_szifi_so_car.py:30`), which leaves no bin out.
- **Planck script: open.**
  - It keeps the default `lrange` = [100, 2500] (`szifi/params.py:24`). On the tiles of SZiFi's Planck survey file (1024 pixels over 14.8°, `surveys/data_planck.py:40–41`), that leaves 96.8% of the modes and 205 of the 255 spectral bins wholly outside ([`p03`](analysis/realistic/szifi_review/p03_code_sites.log)).
  - But there the spectra are taken through an apodised mask (`surveys/data_planck.py:75–76`) with MASTER decoupling (the default, `szifi/params.py:13`). So the empty bins need not hold only round-off, as they did on our ones-masked tiles.
  - Whether the amplification follows there is untested. So is the ℓ range used for the published Planck catalogues. This is the first thing we would check.
  - As shipped, the Planck example scripts stop at E2 first.

**Suggested fix.**
- **Zero what lies outside `lrange`.** Outside `lrange` the templates and the maps are zero. So setting the inverse covariance, or the filter, to zero there, or dropping the bins that lie wholly outside, removes only amplified round-off and leaves the in-band filter unchanged.
- **Optionally, average over in-band modes only.** Averaging each bin over its in-band modes would also remove the dilution of the two edge bins, and would change the results slightly there.
- **At the least, warn** when a spectral bin lies wholly outside `lrange`.

*Evidence:* `RESULTS.md` §2, §3 and §4(b); [`zA02_checks.log`](analysis/realistic/step3_szifi_real/zA02_checks.log) (2); [`zA05_analysis.log`](analysis/realistic/step3_szifi_real/zA05_analysis.log) (1), (2), (4b), (5); the config check's [`c04_compare.log`](verify_real/step3_szifi_real_config/c04_compare.log) (3), (4); [`r01_spot.log`](verify_real/step3_szifi_real_report/r01_spot.log) (6); [`p03`](analysis/realistic/szifi_review/p03_code_sites.log).

### E5. The 'mixed_gaussian' covariance is not positive definite

**What happens.** For 'mixed_gaussian', SZiFi handles each channel pair's 2D cross-periodogram in three steps:
1. it divides it by that pair's isotropic cross-spectrum (`szifi/spec.py:363`);
2. it smooths the ratio with a Gaussian of 10 pixels in the SO example (`spec.py:373`);
3. it multiplies back (`spec.py:382`).

Each element is rescaled differently before and after the smoothing, so the 6 × 6 result need not be positive semi-definite. 'anisotropic_gaussian' runs through the same pair-by-pair loop (`spec.py:333`) but smooths every element with the same kernel, and stays positive semi-definite.

On our tile 0 (baseline):
- 45.5% of the 2D modes have a non-positive eigenvalue (the independent check: 48.0% on its realistic tile 9 at baseline, 25.0% on its noise-only tile 9 at goal);
- 4.1% of the correlation coefficients lie outside [−1, 1];
- the matched filter's weight aᵀC⁻¹a is negative in 6.1% of the modes.

When the filter's normalisation, a sum over modes, comes out negative, σ = 1/√norm is NaN (`szifi/mmf.py:1162–1168`). The run continues without a warning (see D3).

**Effect.**
- With the SO example's settings, SZiFi finds anything from no detection to 133,309 peaks with q > 4 per tile, with median q from about 20 to 3,600 per tile. In one run σ was NaN at 14 of the 15 scales.
- It also happens on maps without halos, and with SZiFi's own four SO channels.
- Masking about 32,000 spurious detections took 342 s of a 685 s run.

**A fix that keeps the matrix positive semi-definite by construction** (tested on one tile, not in the finder; [`p05`](analysis/realistic/szifi_review/p05_mixed_fix.log)). The fix is to smooth a matrix that is positive semi-definite at every mode, rather than each pair's ratio to its isotropic cross-spectrum. On the same tile and spectra:

| covariance | modes with an eigenvalue ≤ 0 | aᵀC⁻¹a < 0 | weight relative to isotropic, median (1–99%) |
|---|---|---|---|
| SZiFi's 'mixed_gaussian' (reproduces the diagnosis above) | 45.49% | 6.13% | 0.98 (−3.6 to +5.1) |
| whitened: smooth W P W, with W the inverse matrix square root of the isotropic covariance, then transform back | 0% | 0% | 1.00 (0.94 to 1.08) |
| direct: smooth the periodogram matrix P itself, as 'anisotropic_gaussian' does | 0% | 0% | 0.99 (0.14 to 1.09) |

- Both variants are positive semi-definite by construction. On our tile neither has a single mode with an eigenvalue ≤ 0 (0 of the 309,135 modes with ℓ > 0, against 140,613 for 'mixed_gaussian').
- The whitened form keeps the idea of 'mixed_gaussian', smoothing the departure from isotropy, and stays close to the isotropic weights. It needs a positive-definite isotropic covariance.
- We did not run either variant through the cluster finder.
- Separately, a check that the covariance is positive definite and the normalisation positive would turn a silent failure into an error.

**Scope** *(our set-up)*. SZiFi's SO example runs on apodised CAR tiles cut from a larger map. Whether that avoids the breakdown was not tested. With an isotropic covariance, the rest of the SO example works on our tiles.

*Evidence:* `RESULTS.md` §7; [`z06_s3diag.log`](analysis/realistic/step3_szifi_real/z06_s3diag.log), [`z06b_mixed.log`](analysis/realistic/step3_szifi_real/z06b_mixed.log), [`z02_smoke_S3.log`](analysis/realistic/step3_szifi_real/z02_smoke_S3.log); [`p05_mixed_fix.log`](analysis/realistic/szifi_review/p05_mixed_fix.log).

### E6. More than one masking iteration never runs

- **What happens.** The noise-iteration loop stops when a convergence test is true (`szifi/mmf.py:467`).
  - Before the test, `identify_clusters` (`szifi/mmf.py:465`) matches the clusters masked after the previous pass with those to be masked now.
  - Its first output, which the test reads, is the previous list (`szifi/cat.py:301`) plus one row with S/N −1 for each new detection that matches none of them (`szifi/cat.py:373–392`).
  - So the test is meant to stop once a pass finds nothing new to mask, that is, no −1 row. But it takes `np.all` of the S/N values first, and only then compares that single boolean with −1. A boolean never equals −1, so once the loop index is above 0 the test is always true.
  - So for any `max_it` ≥ 1, SZiFi stops after the second pass.
- **Confirmed by a run** ([`p07`](analysis/realistic/szifi_review/p07_max_it.log)). With `max_it` = 2 on our tile 0 (baseline), 90 of the 313 detections to be masked after the second pass matched none of the 338 clusters masked after the first.
  - The same test applied to every entry would therefore have run a third pass, but the test as written stopped the loop.
  - The run made two passes, and its catalogues and σ equal those with `max_it` = 1 (22 of 22 arrays).
  - Separately, 12 of the 338 masked clusters were not found again. The test does not look at these.
- **On CAR tiles** the identification step before the test (`szifi/mmf.py:465`) takes the distance path that fails with current numpy (E3). So CAR runs with `max_it` ≥ 2 should stop with an error there. This is from reading the code; we did not run it.
- **Fix.** Compare each entry with −1 before combining them with `np.all`.

### E3. Merging fails with current numpy and scikit-learn

- **'fof' uses a removed scikit-learn keyword.** It passes `affinity=` to scikit-learn's `AgglomerativeClustering` (`szifi/cat.py:176–178`). scikit-learn replaced `affinity` with `metric` and removed it in version 1.4, so the call raises TypeError with 1.9.1.
- **Both modes put a length-1 array into a scalar slot.** They compute angular distances with healpy one pair at a time, and store each result, an array of length 1, into a scalar slot:
  - for 'closest', at `szifi/sphere.py:94`, called from `szifi/cat.py:140`;
  - for 'fof', in the distance function at `szifi/cat.py:291`, through scikit-learn's `pairwise_distances`.

  numpy 2.5 refuses this with "setting an array element with a sequence". The same pattern at `szifi/sphere.py:86` is reached from `identify_clusters` (`szifi/cat.py:323`, `327`), which the noise iteration uses on CAR tiles (E6).
- **Suggested fix.** Pass `metric=`, and take the scalar from healpy's result, or compute all the distances in one vectorised call. With these two adaptations, made in our driver, SZiFi's merging runs and agrees with our code (above).

*Evidence:* [`z09_merge_szifi.log`](analysis/realistic/step3_szifi_real/z09_merge_szifi.log).

### E2. A misspelled key in the default parameters

- **What happens.** The default dictionary spells the key 'snr_weigthing' (`szifi/params.py:22`), but the cluster finder reads 'snr_weighting' (`szifi/mmf.py:95`, `687`). A run with the defaults alone stops with KeyError.
- **The example scripts.** By reading the code, SZiFi's four Planck example scripts (`test_files/run_szifi_planck*.py`) stop the same way: they start from the default dictionary and never set 'snr_weighting'. We did not run them.
- **Workaround.** Set 'snr_weighting', as SZiFi's SO script does.

*Evidence:* [`z02c_defaults_key.log`](analysis/realistic/step3_szifi_real/z02c_defaults_key.log); [`p03`](analysis/realistic/szifi_review/p03_code_sites.log).

### E7. Positions are shifted by half a pixel

- **What happens.**
  - SZiFi centres its filter template on the corner of four pixels at the tile centre, index N/2 − ½ on both axes (`szifi/mmf.py:628–629`; `szifi/maps.py:58`, `65`).
  - scipy's 'same'-mode convolution (`szifi/mmf.py:1155`) instead treats a pixel as the kernel centre. So SZiFi's S/N and y maps are shifted by half a pixel along both axes: the value at pixel n belongs to position n − ½.
  - SZiFi then reports pixel n at its centre, (n + ½)·Δ (`szifi/maps.py:72`, `79`).
- **Seen in the data.**
  - Injected clusters come out exactly (+0.5, +0.5) pixels from their positions in 92–100% of cases.
  - Far from the tile edges, SZiFi's detections lie (+0.50 to +0.54) pixels from our finder's along both axes.
- **Size.** 0.25′ along each axis on our 0.5′ pixels, 0.43′ on SZiFi's Planck tiles. That is small next to the beams, but systematic.
- **Not checked:** SZiFi's sky coordinates.
- **Fix.** Centre the template on the pixel that the 'same' convolution treats as the kernel centre, or shift the reported coordinates by half a pixel.

*Evidence:* `RESULTS.md` §1; the config check's [`c07_inject.log`](verify_real/step3_szifi_real_config/c07_inject.log).

### E1. σ at the first filter scale is reported as 0

- **What happens.**
  - With the default detection method 'maxima_lomem' and more than three scales (`szifi/mmf.py:578`), as in our 15-scale runs, the loop over scales skips the first scale with `continue` (`szifi/mmf.py:711`) before σ is stored at the end of the loop (`szifi/mmf.py:748`).
  - So σ at the smallest scale is reported as 0.
  - SZiFi's default two-scale grid (`szifi/params.py:6`) and its SO example (two scales, `test_files/run_szifi_so_car.py:14`) take the other branch and are not affected.
- **What is not affected.** The catalogues.
- **Workaround.** σ at that scale is the median y0/q of its detections. At the other 14 scales this reproduces SZiFi's σ to ≤ 6×10⁻⁸; a single detection gives it to about 10⁻⁷.
- **Fix.** Store σ before the `continue`.

## Documentation and defaults

- **D1. `lrange` = None crashes.** The default dictionary says that None uses all the modes (`szifi/params.py:24`). But the cluster finder unpacks `lrange` into two numbers (`szifi/mmf.py:179`), and nothing in the package handles the parameter being None. With None the run stops with TypeError at that line ([`p04`](analysis/realistic/szifi_review/p04_lrange_none.log)).
- **D2. Covariance types.**
  - The parameter's comment lists only 'isotropic', 'anisotropic_boxcar' and 'anisotropic_gaussian' (`szifi/params.py:20`). Yet the SO example uses 'mixed_gaussian', and 'mixed_boxcar' is accepted too.
  - An unknown type matches none of the branches in `get_cov` (`szifi/spec.py:292–310`). The run then stops at `szifi/spec.py:312` with UnboundLocalError, without naming the type.
- **D3. `import szifi` switches off all warnings for the whole process** (`szifi/mmf.py:5`).
  - In p06, numpy's warning for √(−1) reaches the user before the import and not after it.
  - This hides E5's NaN σ and the `pkg_resources` deprecation (P4).
- **D4. The default dictionary points to the author's machine:** the paths, the survey file and the template folder (`szifi/params.py:49–51`, `53`).
- **D5. Template files.**
  - `save_and_load_template` defaults to True (`szifi/params.py:52`). Each scale's template is then written to the template folder, the author's path by default.
  - SZiFi refuses to overwrite an existing template file (`szifi/mmf.py:611`) and deletes the files at the end of a run (`szifi/mmf.py:484–491`). So the refusal bites after an interrupted run, or when parallel runs share the folder and field id.
  - With it off, as in our runs, the templates are rebuilt for every scale in both noise iterations (`szifi/mmf.py:613`), which costs time ([S1](#speed-and-memory)).
- **D6. With inpaint = False the noise iteration does nothing.**
  - SZiFi applies its cluster mask only through inpainting (`szifi/mmf.py:238–253`). So with inpaint = False the second pass repeats the first exactly, as it did in 6 of 6 tiles per noise level.
  - Nothing warns about this, and the parameter's comment describes inpainting as being for point sources (`szifi/params.py:31`).

## The method at SO resolution

- **A1. The masked-noise estimate is biased at SO resolution, in either direction.**
  - **How it works.** After masking its detections, SZiFi fills the holes, blends them back towards the original map, and divides the power above ℓ_sep by the unmasked fraction of the tile. The blend keeps about 63% of the power inside the holes, so the division over-corrects.
  - **Default ℓ_sep = 3000, maps without halos.** With E4 and the edge effect A3 removed (5 tiles per noise level), σ at 0.5′ is 2.5% (baseline) and 3.4% (goal) too high, and about 1% low at 15′.
  - **Default ℓ_sep = 3000, realistic tiles.** With E4 and A3 removed (12 tiles), the spectra above ℓ = 3000 end 15–19% above their value before masking. SZiFi then finds 22% (baseline) and 24% (goal) fewer q > 5 clusters than our finder.
  - **The defaults with ℓ_sep raised to 12000,** the value in SZiFi's SO example script. These were measured as configured, with E4 and the edges present. σ is 2–2.5% low at every scale (6–7 noise-only tiles per noise level), and the q > 5 counts are within 2% of our finder's (10 tiles).
  - **The SO example's own settings** compute the spectra only up to ℓ = 10000, below ℓ_sep, so they never divide. With an isotropic covariance they are unbiased on maps without halos (0.993–1.003).
  - **For comparison,** our own finder's masked-covariance estimate ([`szsim/finder.py`](szsim/finder.py)) is within 1% on the same kind of maps.
  - **It matters for cosmology.** In our step 3b, for an SO-like survey (40% of the sky), consider the S8 bias with the Y–M relation fitted (config d). At goal it is +0.68σ with our finder, and +0.90σ and +1.02σ with our re-implementation of SZiFi's masking (defaults with ℓ_sep = 12000, and defaults). At baseline the same three are +0.32σ, +0.31σ and −0.04 ± 0.15σ.
  - **Suggestion.** Calibrate the correction with simulations at the target resolution, or use an estimator that accounts for how the mask couples the modes.
- **A2. Merging radius.**
  - SZiFi's Planck script merges with 'fof' within 10′ (`test_files/run_szifi_planck.py:58–61`). We applied that to our finder's catalogue, whose real clusters we know. At SO depth it removes 7.2% (baseline) and 11.4% (goal) of the real clusters.
  - Merging within 1′ removes only 0.11–0.14% of them, and 1–2′ still removes the fragments of bright clusters.
  - Documenting a smaller radius for SO would help.
- **A3. Periodic simulated tiles** *(our set-up)*. SZiFi treats a tile as a bounded flat patch, as it should for real data. On periodic simulated tiles this has two consequences:
  - **Spurious edge peaks.** Its zero-padded convolution creates spurious peaks near the edges: up to q = 24 (baseline) and 30 (goal) within 2.5′, none beyond 8′.
  - **One-sided filling.** Holes that touch the edge are filled from one side only, while its spectra treat the tile as periodic. This raised the noise level at 0.5′ by +0.004 (baseline) and +0.006 (goal) in r on average, and by up to +0.017 on one tile.

  This is not a bug for SZiFi's intended use. Whether its apodised tiles avoid the second effect was not tested.

## Speed and memory

**Measured** ([`p01`](analysis/realistic/szifi_review/p01_profile.log): one run under cProfile, which reproduces the stored run exactly; [`p02`](analysis/realistic/szifi_review/p02_stage_times.log): the driver's own timers in all 272 stored two-iteration runs):
- **Run time.** One tile run (1200 × 1200 pixels, 6 channels, 15 scales, two noise iterations, one thread) takes:
  - 81.6 s and 82.3 s in the build's two unprofiled test runs of this tile (both alongside a 685 s run of the SO settings; one with a 10′ edge buffer);
  - 90 s in p07's unprofiled run of the stored configuration (with `max_it` = 2, which runs the same two passes, E6);
  - 90.6 s under the profiler;
  - 111.3 s on average over the build's 60 default runs on maps with halos (1–3 runs at a time).

  All 272 stored runs average 104–121 s (arbiter and build).
- **Where the time goes.** Filtering and peak finding take 73% of the profiled run. Over the stored runs they take 79% for the build (without its run with the broken covariance) and 78% for the arbiter (without its periodic-padding switch). The driver's own timers give the same stage times as cProfile within the profiled run.
- **Inside the filtering** (profiled run):
  - **Templates.** The cluster templates are rebuilt for every scale in both iterations: 14.0 s (30 calls). Band-limiting them and the map by FFT round trips takes another 15.6 s.
  - **Filter and convolution:** 33.0 s in all.
    - Transforming each 6-channel template takes 5.6 s.
    - The Fourier-space filter product C⁻¹·T takes 8.3 s, of which 6.3 s are 6 × 6 matrix-vector products at every Fourier mode.
    - Inverse-transforming each filter takes 5.1 s, and the SED weighting 1.6 s.
    - The 180 zero-padded convolutions, one per channel, scale and iteration, take 10.9 s.
- **Fourier transforms.** 1,692 2D FFTs in numpy and scipy per run take 34.4 s, 38% of the run: 1,152 with numpy on the 1200² grid and 540 inside scipy's `fftconvolve` on padded grids. NaMaster's own transforms are not counted; they are inside its 2.9 s.
- **Repeated work** (part of the times above, not added to them).
  - The tile's ℓ grid is computed 568 times (4.0 s, inside the template and band-limit times).
  - Distance maps are computed 368 times (3.4 s), 338 of them for the cluster mask.
  - The cluster mask is built as one full-tile array per detection (338 calls, 4.8 s of the 5.0 s cluster-mask stage). That is why masking 32,000 spurious detections took 342 s (E5).
- **Other costs.** The covariance inversion at every mode takes 1.9 s.
- **Memory.** 1.4–1.7 GB per run with macOS's `MallocLargeCache=0`, and 1.7–2.3 GB without. The inverse covariance alone (1200² × 6 × 6 in float64) is 415 MB.

**Suggestions** (not timed; the savings are our estimates from the profile):
- **S1. Keep templates in memory.** Templates do not depend on the data. Build each scale's template once and keep it in memory for the second iteration and for other tiles with the same pixels.
- **S2. Compute the channel weights once.** The template factorises into a channel part and a scale part, T_i(ℓ) = a_i B_i(ℓ) τ_θ(ℓ). So the channel weights C⁻¹aB can be computed once per iteration instead of a 6 × 6 product at every mode for every scale.
  - **Going further:** filter entirely in Fourier space, where each scale then costs one multiplication and one inverse FFT.
  - **A benchmark of that form.** On random 1200² × 6 inputs, with 12 scales and a Gaussian stand-in profile, it ran 36 times faster than our own re-implementation of SZiFi's per-scale filter and convolution ([`analysis/finder/f00_bench_filtering.log`](analysis/finder/f00_bench_filtering.log)). Its conditions:
    - the re-implementation leaves out building and band-limiting the templates;
    - the fast form used two FFT threads, the other one;
    - both give the same S/N in the tile interior after a one-pixel realignment, because the re-implementation centres its template on pixel N/2 rather than on SZiFi's pixel corner (the 'same' convolution's centring is the one behind E7);
    - it used periodic convolution, so zero padding would have to be added to match SZiFi near the edges;
    - SZiFi itself was not timed.
- **S3. Transform each map once.** Transform each channel's padded map once per iteration, and sum the channels in Fourier space before one inverse transform per scale. This replaces one `fftconvolve` per channel and scale.
- **S4. Invert per ℓ bin.** With the default interp_type 'nearest' (`szifi/params.py:28`), every mode's isotropic covariance is one of the 299 binned matrices. Invert those once and look them up, instead of 1.44 million inversions. This also shrinks the 415 MB inverse covariance.
- **S5. Compute shared maps once.** Compute the ℓ grid and the distance maps once per tile, and build the cluster mask in one pass.
- **S6. Fix E4 first.** With it present, a last-bit change of the scale values moves q by up to 0.1 and the final count by up to 2.7% per tile, so a faster version cannot be checked against the old one. Without it, the same change leaves the catalogues identical. A faster version should then agree to float32 round-off, though entries at near-ties along the scale axis may still move.

**Expected gain.** S1 to S5 keep SZiFi's zero-padded convolution and should give the same results up to round-off. By our estimate from the profile they would save roughly 35–50 s of the 90.6 s profiled run, the larger figure if templates are shared across tiles. Filtering entirely in Fourier space would go further.

## Packaging

- **P1. No licence.** The repository has no licence file, so by default others may not copy, modify or redistribute the code. This is why we only call SZiFi and describe fixes in words. An open-source licence would let users contribute fixes like the ones above.
- **P2. No declared dependencies.** `setup.py` has no `install_requires`. The README lists some of the packages.
- **P3. "Optional" packages are needed at import.** `import szifi` loads pixell, pymaster, scikit-learn, PyYAML and numba ([`p06`](analysis/realistic/szifi_review/p06_import_checks.log)). The README marks pixell and scikit-learn as optional and does not list PyYAML or numba.
- **P4. `pkg_resources`.** `szifi/car.py:7` imports `pkg_resources`, which setuptools has deprecated. With all warnings switched back on, importing `szifi.car`, the CAR interface that SO maps use, warns that it "is slated for removal" and asks to pin setuptools below 81 (p06). Users do not see this warning (D3). `packaging.version` does the same job.
- **P5. `szifi.utils` is pixell's `utils`.** The package's star imports (`szifi/__init__.py`) export the `utils` that `szifi/maps.py` imports from pixell, so `szifi.utils` is not SZiFi's own `utils` module (p06). Code that writes `from szifi import utils` gets the wrong module.

## Reproducing these findings

- **SZiFi runs:** [`analysis/realistic/step3_szifi_real/`](analysis/realistic/step3_szifi_real/).
  - The driver `z_szlib.py` and survey file `z_survey.py`.
  - The arbiter's two diagnostic switches in `zA_lib.py`.
  - Results in `RESULTS.md`, each script with its log.
- **This review:** [`analysis/realistic/szifi_review/`](analysis/realistic/szifi_review/).
  - `p00_map.py`: regenerates test tile 0 from its seeds, bit for bit.
  - `p01_profile.py`: one SZiFi run under cProfile. Its catalogues and σ equal the stored run's (22 of 22 arrays).
  - `p02_stage_times.py`: the time per stage over all stored runs.
  - `p03_code_sites.py`: checks the commit, that the clone is unchanged, each cited line's identifier, the absences, and the bins and modes outside the ℓ range.
  - `p04_lrange_none.py`: D1.
  - `p05_mixed_fix.py`: E5's positive-semi-definite variants.
  - `p06_import_checks.py`: D3, P3–P5.
  - `p07_max_it.py`: E6.
- **Data.** The maps are regenerated from their seeds, not stored.
- **Not done:**
  - nothing was reported to SZiFi's authors;
  - no fix was made inside SZiFi;
  - the speed-ups were not timed;
  - E4 was not tested on SZiFi's Planck configuration, nor E5 on its own SO tiles;
  - SZiFi's example scripts (E2) and CAR runs with `max_it` ≥ 2 (E6) were not run;
  - SZiFi's sky coordinates (E7) were not checked.
