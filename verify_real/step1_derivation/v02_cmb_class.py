"""v02: lensed CMB TT from CLASS (classy 3.4.2) at the project cosmology (colossus planck18), independent of
the builder's CAMB run.  Same physical model choices as stated by the builder (one 0.06 eV neutrino inside
Omega_m, tau = 0.0544, sigma8 = 0.8102 at z = 0, non-linear lensing), but a different Boltzmann code and
non-linear model (CLASS halofit, Takahashi 2012; HMcode variant shown).  Two precision settings for
convergence.  Output: cmb_lensed_TT_class.npz (this directory) + comparison with data_real/cmb_lensed_TT_camb.npz.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vlib as V
V.set_threads(4)
import numpy as np
from classy import Class

h = 0.6766; Ob0 = 0.049; Om0 = 0.3111
mnu = 0.06; omnu = mnu / 93.14
pars = dict(h=h, omega_b=Ob0 * h * h, omega_cdm=(Om0 - Ob0) * h * h - omnu, N_ur=2.0328, N_ncdm=1, m_ncdm=mnu,
            tau_reio=0.0544, n_s=0.9665, sigma8=0.8102, T_cmb=2.7255, output="tCl,lCl,mPk", lensing="yes")


def run(tag, lmax, extra):
    t0 = time.time()
    c = Class()
    p = dict(pars); p.update(extra); p["l_max_scalars"] = lmax
    c.set(p); c.compute()
    lmx = c.lensed_cl()["ell"][-1]
    cl = c.lensed_cl(int(lmx)); ul = c.raw_cl(int(lmx))
    T2 = (2.7255e6) ** 2
    out = dict(ell=cl["ell"], cl=cl["tt"] * T2, ul=ul["tt"] * T2, s8=c.sigma8(), As=c.get_current_derived_parameters(["A_s"])["A_s"])
    print(f"[{tag}] lmax_scalars={lmax} lensed up to l={int(lmx)}; sigma8={out['s8']:.5f} A_s={out['As']:.5e}; "
          f"omega_cdm={p['omega_cdm']:.6f} omega_b={p['omega_b']:.6f}; {time.time()-t0:.0f}s", flush=True)
    c.struct_cleanup(); c.empty()
    return out


# NOTE: a first attempt with l_max_scalars = 24000-26000 and accurate_lensing needed 6-21 GB (over the 5 GB
# budget) and was aborted.  The CMB is < 1e-5 of the quietest beam-deconvolved noise above l ~ 12000, so
# CLASS is run to l_max_scalars = 15000 (2.5 GB) and the lensed tail beyond l = 14000 is a power law fitted
# on [11000, 14000]; v04 shows that sigma_Y does not depend on that tail (tail set to 0 or x3).
A = run("A halofit lmax12000", 12000, {"non_linear": "halofit", "P_k_max_1/Mpc": 30.0})
B = run("B halofit lmax15000", 15000, {"non_linear": "halofit", "P_k_max_1/Mpc": 60.0})
H = run("H hmcode lmax12000", 12000, {"non_linear": "hmcode", "P_k_max_1/Mpc": 30.0})

LT = 14000                                     # trusted CLASS range (l_max_scalars - 1000)
fit = (B["ell"] >= 11000) & (B["ell"] <= LT)
slope, lnamp = np.polyfit(np.log(B["ell"][fit]), np.log(B["cl"][fit]), 1)
ell_full = np.arange(0, 26001)
cl_full = np.zeros(len(ell_full)); cl_full[:LT + 1] = B["cl"][:LT + 1]
cl_full[LT + 1:] = np.exp(lnamp + slope * np.log(ell_full[LT + 1:]))
print(f"power-law tail fit on [11000, {LT}]: slope {slope:.3f}; held-out check none (CLASS stops at 15000); "
      f"B at 14500 vs fit: {B['cl'][14500]/np.exp(lnamp+slope*np.log(14500))-1:+.3e}")
B = dict(B); B["cl"] = cl_full; B["ell"] = ell_full
ulB = np.zeros(len(ell_full)); ulB[:LT + 1] = B["ul"][:LT + 1]; B["ul"] = np.where(ulB > 0, ulB, 1e-300)
L = np.arange(2, 20001)
D = lambda o, l: o["cl"][l] * l * (l + 1) / (2 * np.pi)
lp = np.arange(150, 300)
ip = lp[np.argmax(D(B, lp))]
print(f"first acoustic peak (B): l = {ip}, D_l = {D(B, ip):.1f} uK^2 (builder: l = 220, 5757.5 uK^2)")
print(f"sigma8 (B) = {B['s8']:.5f} (builder CAMB: 0.81025)")

cb = np.load(os.path.join(V.ROOT, "data_real", "cmb_lensed_TT_camb.npz"))
camb = cb["cl_uK2"]
print("\n  l      D_l(B)[uK^2]  A/B-1      H/B-1      CAMB(builder)/B-1  lensed/unlensed(B)")
for l in [100, 220, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 11000, 14000, 15000, 18000, 20000]:
    a = A['cl'][l]/B['cl'][l]-1 if l <= 11000 else np.nan
    hh = H['cl'][l]/B['cl'][l]-1 if l <= 11000 else np.nan
    print(f"  {l:6d} {D(B, l):12.4f}  {a:+.2e}  {hh:+.2e}  {camb[l]/B['cl'][l]-1:+.3e}"
          f"        {B['cl'][l]/B['ul'][l] if l <= LT else np.nan:.3f}")
r = camb[2:20001] / B["cl"][2:20001] - 1
for lo, hi in ((2, 3000), (2, 6000), (6000, 12000), (12000, 20000)):
    m = (L >= lo) & (L <= hi)
    print(f"  max |CAMB_builder/CLASS_B - 1| for {lo} <= l <= {hi}: {np.max(np.abs(r[m])):.3e}")
ra = A["cl"][2:11001] / B["cl"][2:11001] - 1; LL = np.arange(2, 11001)
print(f"  CLASS convergence A (lmax 12000, kmax 30) vs B (lmax 15000, kmax 60): max |A/B-1| l<=6000 "
      f"{np.max(np.abs(ra[LL<=6000])):.2e}; l<=11000 {np.max(np.abs(ra)):.2e}")
rh = H["cl"][2:11001] / B["cl"][2:11001] - 1
print(f"  non-linear model: HMcode vs halofit: max |H/B-1| l<=3000 {np.max(np.abs(rh[LL<=3000])):.2e}; "
      f"l<=6000 {np.max(np.abs(rh[LL<=6000])):.2e}; l<=11000 {np.max(np.abs(rh)):.2e}")
np.savez(os.path.join(HERE, "cmb_lensed_TT_class.npz"), ell=B["ell"], cl_uK2=B["cl"], unlensed_uK2=B["ul"],
         tail_slope=slope, tail_lnamp=lnamp, l_trust=LT)
print("saved cmb_lensed_TT_class.npz")
