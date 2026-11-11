"""v01: our OWN lensed CMB TT spectrum (CAMB 2.0.5), independent of the build's data_real/cmb_lensed_TT_camb.npz,
used to generate the CMB in the map simulations.  Two accuracy settings (convergence) and a comparison with
the build's stored spectrum.

Cosmology: colossus planck18 (H0 = 67.66, Om0 = 0.3111, Ob0 = 0.0490, ns = 0.9665, sigma8 = 0.8102), one
massive neutrino 0.06 eV inside Om0, Planck 2018 As = 2.105e-9, tau = 0.0561 (TT,TE,EE+lowE+lensing+BAO;
the build used 0.0544: a 0.3% amplitude difference at l > 30, irrelevant at the 3% sigma_Y level).
Non-linear (halofit) lensing.  Stored: l <= 9000 from the accurate run; power law beyond (fit 7000-9000).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402  (sets thread env vars)
import numpy as np                                  # noqa: E402
import camb                                         # noqa: E402

log = V.Tee(os.path.join(V.HERE, "v01_camb.log"))
h = 0.6766
ombh2 = 0.0490 * h * h
omnuh2 = 0.06 / 93.14
omch2 = (0.3111 - 0.0490) * h * h - omnuh2


def run(lmax, lpa, boost):
    t0 = time.time()
    p = camb.CAMBparams()
    p.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0.0, tau=0.0561, TCMB=V.TCMB)
    p.InitPower.set_params(As=2.105e-9, ns=0.9665)
    p.set_for_lmax(lmax, lens_potential_accuracy=lpa)
    p.set_accuracy(AccuracyBoost=boost)
    p.set_matter_power(redshifts=[0.0], kmax=2.0)
    p.NonLinear = camb.model.NonLinear_lens
    r = camb.get_results(p)
    cl = r.get_lensed_scalar_cls(CMB_unit="muK", raw_cl=True)[:, 0]
    log(f"   run lmax={lmax} lens_potential_accuracy={lpa} AccuracyBoost={boost}: sigma8 = {r.get_sigma8_0():.5f}, "
        f"{time.time()-t0:.0f}s")
    return cl


log(f"[v01] ombh2 = {ombh2:.5f}, omch2 = {omch2:.5f}, mnu = 0.06 eV")
A = run(10000, 4, 1.0)
B = run(14000, 8, 1.5)
ell = np.arange(9001)
a, b = A[:9001], B[:9001]
D = ell * (ell + 1) / (2 * np.pi)
log("   convergence (run A vs run B):")
for l in (220, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000):
    log(f"      l={l:5d}: D_l(B) = {D[l]*b[l]:10.4f} uK^2   A/B - 1 = {a[l]/b[l]-1:+.2e}")
ipk = 100 + np.argmax((D * b)[100:400])
log(f"   first peak l = {ipk}, D = {D[ipk]*b[ipk]:.1f} uK^2")

# comparison with the build's stored spectrum
d = np.load(f"{V.ROOT}/data_real/cmb_lensed_TT_camb.npz")
cb = np.interp(ell, d["ell"], d["cl_uK2"])
log("   ours (run B) vs build's data_real/cmb_lensed_TT_camb.npz:")
for l in (2, 10, 30, 100, 220, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000):
    log(f"      l={l:5d}: ours/build - 1 = {b[l]/cb[l]-1:+.4f}")
r = b[30:9001] / cb[30:9001] - 1
log(f"   max |ours/build - 1| over 30 <= l <= 3000: {np.abs(r[:2971]).max():.4f}; over 3000-9000: "
    f"{np.abs(r[2970:]).max():.4f}")

# store (l <= 9000 from run B) + power-law extrapolation fitted over 7000-9000
sel = (ell >= 7000) & (ell <= 9000)
slope, lnamp = np.polyfit(np.log(ell[sel]), np.log(b[sel]), 1)
log(f"   extrapolation beyond l = 9000: C_l = exp({lnamp:.3f}) l^{slope:.3f}; at l = 20000: C = "
    f"{np.exp(lnamp)*20000**slope:.3e} (build: {np.interp(20000, d['ell'], d['cl_uK2']):.3e})")
np.savez(os.path.join(V.OUT, "cmb_camb_mine.npz"), ell=ell, cl_uK2=b, lcut=9000, slope=slope, lnamp=lnamp)
log("[v01] saved out/cmb_camb_mine.npz")
