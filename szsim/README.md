# szsim — full-sky Poisson halo catalogues with a tSZ matched-filter SNR

Python: `/Users/boris/GitHub/sz-poisson-blending/.venv/bin/python` (the project root is on the
venv path through `szsim_project.pth`, so `import szsim` works from any directory).

## Baseline model

| item | choice |
|---|---|
| cosmology | colossus `planck18` (h = 0.6766). The colossus default power spectrum (Eisenstein & Hu 98) is used for sigma(M) |
| HMF | Tinker08, `mdef='500c'`, 0.01 < z < 3, 1e13 < M500c < 1e16 **Msun (h-free)** |
| positions | isotropic, independent (Poisson point process) on the full sphere |
| Y–M | Planck 2013 XX: `E(z)^-2/3 D_A^2 Y500 / 1e-4 Mpc^2 = 10^-0.19 [0.8 M500 / 6e14 Msun]^1.79`, with **no** (h/0.7)^(-2+alpha) factor by default (`h70_factor=True` adds it, a +0.7% change in Y). The relation gives the **median** Y; scatter is lognormal with sigma_lnY = 0.173 |
| profile | Arnaud+10 GNFW (c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905), truncated at 5 R500 (3D) and projected |
| normalisation | **spherical Y(<R500) = Y500**, so the template integrates to Y5R500/Y500 = `RATIO_5R500` = **1.79644** (computed numerically) |
| instrument | Gaussian beam FWHM 1.4', white y-noise 4e-6 arcmin, no CMB or foregrounds |
| detection | single-cluster matched filter at the cluster's own theta500. q = Y500/sigma_Y(theta500), q_obs = q + N(0,1) |

Matched filter: with white noise and no other foregrounds, the filter is the beam-convolved
template itself. We use the flat-sky **Hankel** form
`sigma_Y^-2 = (1/2pi) ∫ dl l B_l^2 g(l theta500)^2 / N_w`. Here `g(u)` is the Hankel transform of
the unit-Y500 profile, tabulated once for u in [0, 3000] (cached in
`data/template_hankel_arnaud_5R500.npz`). sigma_Y(theta500) is tabulated on 300 log-spaced points,
theta500 in [1e-3', 3000'], and interpolated with a log-log cubic spline. Flat sky is a fair
approximation here: every halo in the catalogues has theta500 < 70'.

## API

```python
import szsim
from szsim import cosmo_hmf as ch, signal as sg, sampler

# --- cosmo_hmf.py (masses Msun, distances Mpc, all h-free)
ch.get_cosmology()                       # colossus planck18 object
ch.Ez(z); ch.chi_Mpc(z); ch.DA_Mpc(z); ch.rho_crit_Msun_Mpc3(z); ch.dV_dz_dOmega_Mpc3(z)
ch.dndlnM_Mpc3(M, z)                     # comoving dn/dlnM [Mpc^-3], scalar z
ch.d2N_dzdlnM_fullsky(M, z)              # full-sky d2N/dz/dlnM, scalar z
T = ch.HMFTable(zmin=0.01, zmax=3.0, Mmin=1e13, Mmax=1e16, nz=1201, nM=1201)   # cached in data/
T = ch.default_table()                   # the shared default table above
T.z, T.lnM, T.d2N                        # nodes and full-sky d2N/dz/dlnM (nz, nM)
T.N_total(Mmin=None, Mmax=None, zmin=None, zmax=None)  # expected full-sky counts
T.N_above(Mmin);  T.dN_dz(Mmin=None, Mmax=None)        # dN/dz on T.z

# --- signal.py
sg.RATIO_5R500                            # Y5R500/Y500 = 1.79644
sg.gnfw_P(x); sg.projected_shape(s)       # 3D shape; projected shape (unit Y500, s = theta/theta500)
sg.template_real(theta, theta500)         # unit-Y500 template tau [arcmin^-2], not beam-convolved
sg.template_hankel(u=None)                # g(u), u = l*theta500[rad]; with u=None returns the (u, g) table
m = sg.SZModel(logA=-0.19, alpha=1.79, beta_E=2/3, one_minus_b=0.8, M_pivot=6e14,
               sigma_lnY=0.173, h70_factor=False, beam_fwhm_arcmin=1.4, noise_y_arcmin=4e-6)
m.Y500_med(M, z)            # median Y500 [arcmin^2];  m.Y500_Mpc2_med(M, z) [Mpc^2]
m.R500_Mpc(M, z); m.theta500(M, z)   # [Mpc], [arcmin]
m.sigma_Y(theta500)         # matched-filter sigma_Y [arcmin^2] (spline); m.sigma_Y_exact(theta500)
m.sigma_Y_point             # point-source limit [arcmin^2]
m.q_mean(M, z, lnY_dev=0)   # mean SNR
m.p_det(M, z, q_cut)        # P(q_obs > q_cut | M, z), marginalised over Y scatter and noise
m.expected_detections(q_cut, table=None, Mmin=None, Mmax=None, zmin=None, zmax=None)  # full sky
m.cross_response(theta_A, theta_B, sep_arcmin)
    # rho_AB: Y_hat of A's matched filter (centred on A) per unit Y500 of a cluster B at sep.
    # rho(thA, thA, 0) = 1.  Blended estimate: Y_hat_A = Y_A + sum_B rho_AB Y_B + noise

# --- sampler.py
cat = sampler.draw_catalog(seed, Mmin=1e13, Mmax=1e16, zmin=0.01, zmax=3.0, model=None,
                           table=None, fields=sampler.FIELDS, chunk=2_000_000, dtype=np.float32)
    # dict of float32 arrays: ra, dec [deg], z, M500 [Msun], Y500 [arcmin^2, incl. scatter],
    # theta500 [arcmin], q, q_obs; plus cat['meta'] (seed, N, N_expected, model params)
sampler.expected_count(Mmin=1e13, Mmax=1e16, zmin=0.01, zmax=3.0)   # = sum of the cell lambdas
sampler.save_catalog(cat, path)          # uncompressed .npz (meta stored as a JSON string)
sampler.load_catalog(path, fields=None, Mmin=None, qmin=None)   # optional field subset and cuts
```

Sampler algorithm: Poisson counts in each cell of the 1200 x 1200 (z, lnM) table. Inside a cell,
(z, lnM) is drawn exactly from the bilinear interpolant of d2N/dz/dlnM (closed-form inverse CDFs).
Positions are drawn with RA ~ U[0,360) and sin(Dec) ~ U[-1,1]. A mass floor `Mmin` above 1e13
handles the partial lowest cell exactly. Draws are reproducible for a fixed seed: redrawing seed 101
gave a bit-identical catalogue. Rows are ordered by z cell, not by sky position. float32 RA/Dec
resolve about 0.1".

Cost (10-core Mac): about 7.5 s and 1.55 GB peak RSS to draw a full M>1e13 sky with all fields
(2.93e7 halos). A M>1e14 sky takes well under 1 s once the tables are warm.

## Stored catalogues (`data/catalogs/`)

* `cat_M1e13_seed{1,2,3}.npz`: full population M500 > 1e13, about 2.93e7 halos, 938 MB each.
  Use `load_catalog(path, fields=[...], Mmin=..., qmin=...)` to keep memory down.
* `cat_M1e14_seed{101..110}.npz`: mass floor M500 > 1e14, about 3.96e5 halos, 13 MB each.
* Other files: `data/hmf_table_*.npz` (HMF table), `data/template_hankel_arnaud_5R500.npz`, and
  `data/build_summary.json`.

## Validation (scripts and logs in `analysis/build/`, figures in `figures/build/`)

Every number below comes from those logs.

**HMF counts**, full sky, 0.01 < z < 3. The table trapezoid is compared with an independent
300x300 Gauss–Legendre integral of colossus.

| M500 > [Msun] | N (table) | rel. diff vs GL | dN/dz peak | median z | frac z>1 | frac z>2 |
|---|---|---|---|---|---|---|
| 1e13 | 29,295,487 | +1.5e-5 | 0.852 | 1.036 | 0.527 | 0.0719 |
| 3e13 | 4,598,090 | +6.6e-6 | 0.728 | 0.859 | 0.382 | 0.0230 |
| 1e14 | 396,387 | -1.6e-5 | 0.583 | 0.676 | 0.203 | 0.0026 |
| 3e14 | 21,840.5 | -2.3e-5 | 0.441 | 0.523 | 0.069 | 0.0001 |

**sigma_Y** (`validate_sigmaY.log`, `validate_cross_response.log`):
* Point-source limit: 4.6927e-6 arcmin^2. The table value at theta500 = 1e-3' agrees to 2e-5.
* Hankel vs an independent real-space 2D-FFT pixel calculation, theta500 = 0.3'–30': the two agree
  to better than 7e-4.
* Monte Carlo matched filter on 20,000 white-noise maps (0.25' pixels):
  * std(Y_hat) matches the pixel-grid prediction within +1.44, +0.81 and +0.32 sigma for
    theta500 = 0.5', 2' and 6'.
  * Y_hat is unbiased: +0.63, +0.30 and -0.84 sigma.
* `cross_response` vs a direct 2D pixel matched filter (30 configurations): max |difference| = 2.2e-4.

Reference values: sigma_Y(1') = 7.23e-6, sigma_Y(3') = 1.46e-5, sigma_Y(10') = 4.31e-5 arcmin^2.
Mean q (median Y, no scatter):

| M500 [Msun] | z=0.1 | z=0.3 | z=0.5 | z=1 | z=2 |
|---|---|---|---|---|---|
| 1e14 | 5.12 | 2.11 | 1.59 | 1.47 | 2.24 |
| 2e14 | 14.27 | 6.11 | 4.73 | 4.51 | 7.11 |
| 3e14 | 25.93 | 11.33 | 8.88 | 8.64 | 13.86 |
| 6e14 | 71.78 | 32.20 | 25.78 | 25.88 | 42.73 |
| 1e15 | 151.73 | 69.13 | 56.07 | 57.49 | 96.92 |

**Sampler** (`validate_sampler.log`): 24 full M>1e13 skies (seeds 1–24) and 200 skies with
M>1e14 floor (seeds 101–300). The error on each mean is sqrt(lambda/R).

| quantity | analytic | MC mean ± error | offset | var/mean (expect 1) |
|---|---|---|---|---|
| N(M>1e13), 24 skies | 29,295,487 | 29,295,190 ± 1,105 | -0.27σ | 0.96 ± 0.30 |
| N(M>3e13), 24 skies | 4,598,090 | 4,598,202 ± 438 | +0.25σ | 0.76 ± 0.30 |
| N(M>1e14), 24 skies | 396,387 | 396,514 ± 129 | +0.99σ | 0.51 ± 0.30 (p=0.05) |
| N(M>3e14), 24 skies | 21,840.5 | 21,841.6 ± 30.2 | +0.04σ | 1.53 ± 0.30 |
| N(M>1e14), 200 floor-1e14 skies | 396,387 | 396,401 ± 44.5 | +0.31σ | 0.91 ± 0.10 |
| N(M>3e14), 200 floor-1e14 skies | 21,840.5 | 21,844.1 ± 10.5 | +0.35σ | 0.84 ± 0.10 |

* Six redshift bins of the M>1e13 population (0.01–0.25–0.5–1–1.5–2–3) agree to between -1.67 and
  +0.27 sigma. Across all 18 count tests of the 24-sky set, the z-scores have rms 0.62 and
  max |z| 1.67. The rows are correlated, because counts are nested.
* KS tests against the exact marginal CDFs, about 2.9e7 halos per sky:
  * z: median D = 1.71e-4, against 1/sqrt(N) = 1.85e-4. The 24 p-values are consistent with
    uniform (KS p = 0.255).
  * lnM: median D = 1.47e-4, uniformity p = 0.761.
  * For the 200 floor-1e14 skies: z p = 0.671, lnM p = 0.077.
* Y scatter: in the 3 stored M>1e13 skies, KS of ln(Y/Y_med)/0.173 against N(0,1) gives
  p = 0.205, 0.143 and 0.276. For q_obs - q against N(0,1): p = 0.074, 0.664 and 0.788.
* Isotropy:
  * HEALPix nside=16 counts: chi2/dof = 0.9954 ± 0.0052, combined p = 0.81.
  * Angular power of the count map over the shot-noise level 4pi/N:
    l=1: 0.88 ± 0.17, l=2: 0.85 ± 0.13, l=3: 1.11 ± 0.11. The (2l+1)-weighted mean over
    l = 1..47 is 0.9924 ± 0.0060.
  * Floor-1e14 skies at nside=4: chi2/dof = 0.9994 ± 0.0072.
* Validation run peak RSS: 2.94 GB.

**Expected full-sky detections**: the selection-function integral with lognormal Y scatter
(48-point Gauss–Hermite) and unit Gaussian noise, checked against the mean of 24 skies.

| q_cut | N_det (M>1e13 floor) | MC, 24 skies | offset | N_det (M>1e14 floor) | MC, 200 skies | offset | share from M<1e14 |
|---|---|---|---|---|---|---|---|
| 4 | 132,124.8 | 132,096.2 ± 74.2 | -0.38σ | 123,959.1 | 123,975.1 ± 24.9 | +0.65σ | 6.18% |
| 5 | 85,912.3 | 85,920.4 ± 59.8 | +0.14σ | 84,076.3 | 84,095.5 ± 20.5 | +0.94σ | 2.14% |
| 6 | 61,629.5 | 61,623.4 ± 50.7 | -0.12σ | 60,782.2 | 60,793.4 ± 17.4 | +0.65σ | 1.37% |
| 7 | 46,574.5 | 46,586.5 ± 44.1 | +0.27σ | 46,094.3 | 46,107.3 ± 15.2 | +0.86σ | 1.03% |

* The detections below the mass floor, M<1e14 with q_obs>q_cut, also agree:
  * q>4: 8,165.7 analytic vs 8,175.1 ± 18.4 (+0.51σ)
  * q>5: 1,836.0 vs 1,846.9 ± 8.7 (+1.25σ)
  * q>6: 847.4 vs 852.5 ± 5.9 (+0.87σ)
  * q>7: 480.3 vs 482.8 ± 4.5 (+0.56σ)
* Detection var/mean from the 200 floor-1e14 skies is 0.93, 1.15, 1.07 and 0.99 (± 0.10) for
  q_cut = 4–7.
* The 24-sky set gives 1.5–1.9 ± 0.30 (p = 0.10, 0.02, 0.01, 0.07). These four values come from
  nested, strongly correlated counts, so they are one ~2-sigma fluctuation. The larger set does
  not reproduce it.
* The detections differ from the single-cluster selection only through the noise and the Y
  scatter. Blending is **not** included here.
