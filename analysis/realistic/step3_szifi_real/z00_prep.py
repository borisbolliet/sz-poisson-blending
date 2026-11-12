"""z00: inputs for running SZiFi itself on step 3a's tiles (project .venv; SZiFi's venv has no colossus, so szsim
cannot be imported there).

Writes data_real/step3_szifi_real/
  instrument.npz : our SO LAT channels as SZiFi's experiment object needs them: nu_eff [Hz], FWHM [arcmin],
                   tsz_sed = szsim.instrument.tsz_f_uK(freqs) [uK per unit y], white noise levels [uK arcmin] per
                   noise level (SO_LAT['white']);
  analytic_shells.npz : the analytic CMB + noise covariance of our skies (szsim.finder.Grid.analytic_cov_bands, the
                   truth the tiles are drawn from) at every distinct |k|^2 = n^2 of the 1200^2 grid, in uK^2 sr
                   (SZiFi's units: E|maps.get_fft(map)|^2 = C_l); n^2 = 0 is a dummy (identity);
  reference.npz  : sigma_Y(theta) of step 3a (s3lib.sigma_Y = step 1's MMF, arcmin^2), the finder's analytic lattice
                   sigma, step 2's noise normalisations (i), (ii), (iii) at the 15 scales, and the tile lists.
Usage: python -I -B z00_prep.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(ROOT, "data_real", "step3_szifi_real")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "out", "numba_cache_proj")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.join(ROOT, "analysis", "realistic", "step3"))
import numpy as np
import s3lib as S                                   # noqa: E402  (the build's helpers)
from szsim import finder as F, instrument as ins    # noqa: E402

t0 = time.time()
os.makedirs(OUT, exist_ok=True)
G = F.grid()
TH = F.THETAS15

# ------------------------------------------------------------------ tiles
PIDS_REAL = list(range(0, 324, 11))                  # step 3a arbiter's a01 tiles (30)
PIDS_NOISE = [p for p in PIDS_REAL if p < 100]        # 10 of them; also in t02's noise_only/ and in tiles_szifi/
PIDS_S0 = [0, 33, 66, 99, 132, 165]                   # 6 of them; also the SZiFi check's v04 tiles
PIDS_S3 = PIDS_REAL[:12]
print(f"tiles: real {PIDS_REAL}\n       noise-only {PIDS_NOISE}\n       S0 {PIDS_S0}\n       S3 (minimum) {PIDS_S3}")

# ------------------------------------------------------------------ instrument
fr = np.asarray(ins.SO_LAT["freqs"], float)
inst = dict(freqs_GHz=fr, nu_eff_Hz=fr * 1e9, fwhm_arcmin=np.asarray(ins.SO_LAT["fwhm"], float),
            tsz_sed_uK=ins.tsz_f_uK(fr), white_baseline=np.asarray(ins.SO_LAT["white"]["baseline"], float),
            white_goal=np.asarray(ins.SO_LAT["white"]["goal"], float))
np.savez(os.path.join(OUT, "instrument.npz"), **inst)
print("\ninstrument (szsim.instrument.SO_LAT, tsz_f_uK):")
for k, v in inst.items():
    print(f"  {k:15s} " + " ".join(f"{x:12.6g}" for x in v))

# ------------------------------------------------------------------ analytic covariance on the |k|^2 shells
i0 = np.rint(np.fft.fftfreq(F.N) * F.N).astype(np.int64)          # -600..599, as SZiFi's fftfreq layout
n2 = np.unique((i0[:, None] ** 2 + i0[None, :] ** 2).ravel())
print(f"\nanalytic covariance: {n2.size} distinct n^2 on the 1200^2 grid (l = 36 sqrt(n^2), max {36 * np.sqrt(n2.max()):.1f})")
fac = G.Npx / G.Om                                                  # rfft units -> uK^2 sr
out = dict(n2=n2)
for mode in S.MODES:
    C = np.zeros((n2.size, 6, 6))
    C[0] = np.eye(6)                                                # dummy for l = 0 (the template is zero there)
    for a in range(1, n2.size, 20000):
        sl = slice(a, min(a + 20000, n2.size))
        C[sl] = G.analytic_cov_bands(36.0 * np.sqrt(n2[sl].astype(float)), mode) / fac
    out[mode] = C
    print(f"  {mode}: C_00 at l = 108 / 3000 / 20000: "
          + " / ".join(f"{C[np.searchsorted(n2, int(round((l / 36.0) ** 2))), 0, 0]:.4e}" for l in (108.0, 3000.0, 20000.0))
          + " uK^2 sr;  C_33 at the same l: "
          + " / ".join(f"{C[np.searchsorted(n2, int(round((l / 36.0) ** 2))), 3, 3]:.4e}" for l in (108.0, 3000.0, 20000.0)))
np.savez(os.path.join(OUT, "analytic_shells.npz"), **out)

# ------------------------------------------------------------------ reference sigmas and step 2's normalisations
ref = dict(thetas=TH, pids_real=np.array(PIDS_REAL), pids_noise=np.array(PIDS_NOISE), pids_s0=np.array(PIDS_S0),
           pids_s3=np.array(PIDS_S3))
s2 = dict(np.load(f"{ROOT}/data_real/step2/inputs.npz"))
for mode in S.MODES:
    ref[f"{mode}_sigY"] = S.sigma_Y(mode, TH)
    ref[f"{mode}_sig_lattice"] = F.Filter(G, mode=mode, analytic=True).sigma_arcmin2
    ref[f"{mode}_r_i"] = np.interp(np.log(TH), np.log(s2[mode + "_th"]), s2[mode + "_r_i"])
    ref[f"{mode}_r_ii"] = np.interp(np.log(TH), np.log(s2[mode + "_th"]), s2[mode + "_r_ii"])
    k3 = np.load(f"{ROOT}/data_real/step2/kernel_{mode}_effiii.npz")
    ref[f"{mode}_r_iii"] = np.interp(np.log(TH), np.log(k3["th"]), k3["r"])
    print(f"\n{mode}: theta [']          " + " ".join(f"{t:8.3f}" for t in TH))
    print(f"  sigma_Y [arcmin^2]      " + " ".join(f"{x:8.2e}" for x in ref[f"{mode}_sigY"]))
    print(f"  lattice / sigma_Y - 1   " + " ".join(f"{x:+8.1e}" for x in ref[f"{mode}_sig_lattice"] / ref[f"{mode}_sigY"] - 1))
    for v in ("i", "ii", "iii"):
        print(f"  step 2 r_({v:3s})          " + " ".join(f"{x:8.4f}" for x in ref[f"{mode}_r_{v}"]))
np.savez(os.path.join(OUT, "reference.npz"), **ref)
print(f"\ndone [{time.time() - t0:.0f} s]")
