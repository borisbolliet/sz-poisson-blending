"""Step 1 / s01: lensed CMB TT power spectrum from CAMB at the project cosmology (colossus planck18).

Cosmology (szsim/cosmo_hmf.py -> colossus 'planck18'): H0 = 67.66, Om0 = 0.3111, Ob0 = 0.049, ns = 0.9665,
sigma8 = 0.8102, flat.  CAMB: one massive neutrino of 0.06 eV (Planck convention) with omch2 chosen so that
Omega_cdm + Omega_b + Omega_nu = 0.3111; tau = 0.0544; As rescaled so that sigma8(z=0) = 0.8102 exactly.

Two CAMB runs (independent accuracy settings) are compared:
   run A: lmax = 28000, lens_potential_accuracy = 12, AccuracyBoost 1.5, lAccuracyBoost 1.5 (convergence check)
   run B: lmax = 26000, lens_potential_accuracy = 8, AccuracyBoost 1.5, lAccuracyBoost 1.5
Both use non-linear (halofit) lensing: set_for_lmax(lens_potential_accuracy>0) sets NonLinear_lens (printed).
Run B is stored up to l = 22000 (CAMB lensed C_l are unreliable within ~2000 of lmax: a first attempt with
lmax = 12000/14000 showed 10-40% differences at l > 5000, see the log of that attempt quoted at the top of
s01_camb_cmb.log).  Beyond l = 22000 (never used by the l <= 20000 filters) a power law fitted on [18000, 22000].
Output: data_real/cmb_lensed_TT_camb.npz  (ell, cl_uK2 [C_l in uK^2], extrapolation slope/amplitude).
"""
import os, sys, time, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np
import camb

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = f"{ROOT}/data_real/cmb_lensed_TT_camb.npz"
from colossus.cosmology import cosmology
P18 = cosmology.cosmologies["planck18"]
h = P18["H0"] / 100
T_CMB = 2.7255


def run(lmax, lpa, boost, lboost, As=2.1e-9):
    pars = camb.CAMBparams()
    mnu = 0.06
    omnuh2 = mnu / 93.14
    ombh2 = P18["Ob0"] * h * h
    omch2 = (P18["Om0"] - P18["Ob0"]) * h * h - omnuh2
    pars.set_cosmology(H0=P18["H0"], ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=0.0544, TCMB=T_CMB)
    pars.InitPower.set_params(As=As, ns=P18["ns"])
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=lpa)
    pars.set_accuracy(AccuracyBoost=boost, lAccuracyBoost=lboost)
    print('   CAMB NonLinear setting:', pars.NonLinear)
    res = camb.get_results(pars)
    s8 = res.get_sigma8_0()
    cls = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True, lmax=lmax)
    return pars, s8, cls["lensed_scalar"][:, 0], cls["unlensed_scalar"][:, 0], (ombh2, omch2)


def main():
    t0 = time.time()
    # sigma8 normalisation: sigma8 propto sqrt(As)
    _, s8, _, _, _ = run(2000, 1, 1, 1)
    As = 2.1e-9 * (P18["sigma8"] / s8) ** 2
    print(f"As for sigma8={P18['sigma8']}: {As:.5e} (sigma8 at As=2.1e-9 was {s8:.5f})")
    pA, s8A, clA, ulA, (ombh2, omch2) = run(28000, 12, 1.5, 1.5, As)
    print(f"run A done {time.time()-t0:.0f}s  sigma8={s8A:.5f}  ombh2={ombh2:.6f} omch2={omch2:.6f}")
    pB, s8B, clB, ulB, _ = run(26000, 8, 1.5, 1.5, As)
    print(f"run B done {time.time()-t0:.0f}s  sigma8={s8B:.5f}")
    ell = np.arange(len(clB))
    # --- check 1: run A vs run B
    for L in (100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 18000, 20000, 22000):
        print(f"  l={L:6d}: D_l(B) = {ell[L]*(L+1)*clB[L]/2/np.pi:10.4f} uK^2 ; A/B - 1 = {clA[L]/clB[L]-1:+.2e};"
              f" lensed/unlensed = {clB[L]/ulB[L]:.3f}")
    # --- check 2: sanity vs well-known numbers: first peak position and height
    D = ell * (ell + 1) * clB / 2 / np.pi
    lp = 150 + np.argmax(D[150:400])
    print(f"first acoustic peak: l = {lp}, D_l = {D[lp]:.1f} uK^2 (Planck 2018 best fit: l~220, ~5700-5800 uK^2)")
    # --- extrapolation: power law over [8000, 11000], tested on held-out [11000, 13500]
    lA = slice(2, 20001)
    print(f"run A vs run B over 2 <= l <= 20000: max |A/B-1| = {np.abs(clA[lA]/clB[lA]-1).max():.2e}; "
          f"over l <= 6000: {np.abs(clA[2:6001]/clB[2:6001]-1).max():.2e}")
    lo2, hi2 = 18000, 22000
    p2 = np.polyfit(np.log(ell[lo2:hi2 + 1]), np.log(clB[lo2:hi2 + 1]), 1)
    print(f"stored extrapolation: fit [{lo2},{hi2}] slope {p2[0]:.3f}; C_20000 = {np.exp(np.polyval(p2, np.log(20000))):.3e} uK^2"
          f" -> D_20000 = {20000**2*np.exp(np.polyval(p2, np.log(20000)))/2/np.pi:.3f} uK^2")
    np.savez(OUT, ell=ell[:hi2 + 1], cl_uK2=clB[:hi2 + 1], unlensed_uK2=ulB[:hi2 + 1], ext_slope=p2[0], ext_lnamp=p2[1],
             l_trust=hi2, meta=json.dumps(dict(H0=P18["H0"], ombh2=ombh2, omch2=omch2, mnu=0.06, tau=0.0544, As=As,
                                               ns=P18["ns"], sigma8=s8B, TCMB=T_CMB, camb=camb.__version__,
                                               lmax=26000, lens_potential_accuracy=8)))
    print(f"saved {OUT}; total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
