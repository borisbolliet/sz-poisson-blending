"""[REPAIR v2 of c06: converged model cosmolib2, v2 catalogue truth; output mcmc_v2_*.npz]
emcee posterior on an Asimov data vector: compare widths with Fisher and the location with the ML (c05).
usage: c06_mcmc.py CONFIG DATA SURVEY NWALK NSTEP     (CONFIG from c05.CONFIGS, DATA in self|cat|map_clean, q_cut 5)
Flat priors inside the HMF grid (0.20<Om<0.45, 0.62<s8<1.02), (1-b)>0, slnY>0, alpha unbounded; Gaussian priors as in c05.
Output: data_cosmo/mcmc_<CONFIG>_<DATA>_<SURVEY>.npz
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, emcee

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL, cosmolib2 as C2
from c05_fits import CONFIGS, SURVEYS

cn, dn, sv = sys.argv[1], sys.argv[2], sys.argv[3]
nwalk, nstep = int(sys.argv[4]), int(sys.argv[5])
t0 = time.time()
free, pri = CONFIGS[cn]; fsky = SURVEYS[sv]
model = C2.SmoothModel(5.0)
N0 = model.counts(CL.FID)
if dn == "self":
    d = N0
elif dn == "cat":
    d = N0 + np.load(f"{CL.DC}/truth_catalogue_v2.npz")["D_q5"]
else:
    tm = np.load(f"{CL.DC}/truth_maps.npz"); sc = float(tm["fullsky_deg2"] / tm["area_deg2"])
    d = N0 + (tm["clean5"] - tm["iso5"]).sum(0) * sc
F = C2.Fitter2(model, d * fsky, free, fsky=fsky, priors=pri)
fi = [CL.PNAMES.index(p) for p in free]
xml, Lml, _ = F.fit(polish=True)
Fm = F.fisher(xml); C = np.linalg.inv(Fm)
Ft = F.fisher(CL.FID[fi])
sig_t, _ = CL.derived_sigma(Ft, fi)
print(f"config {cn} free={free} priors={pri} data={dn} survey={sv}: ML {np.round(xml, 5)}  [{time.time()-t0:.0f}s]")
rng = np.random.default_rng(7)
p0 = rng.multivariate_normal(xml, 0.25 * np.linalg.inv(Ft), nwalk)   # init ball from the Fisher at the truth
if 'slnY' in free:
    p0[:, free.index('slnY')] = np.abs(p0[:, free.index('slnY')]) + 1e-3
assert np.all(np.isfinite([F.lnL(p) for p in p0]))
S = emcee.EnsembleSampler(nwalk, len(fi), F.lnL)
S.run_mcmc(p0, nstep, progress=False)
ch = S.get_chain(discard=nstep // 4, flat=True)
try:
    tau = S.get_autocorr_time(discard=nstep // 4, quiet=True)
except Exception as e:
    tau = np.full(len(fi), np.nan)
print(f"done {time.time()-t0:.0f}s; acceptance {S.acceptance_fraction.mean():.2f}; tau {np.round(tau, 1)}; "
      f"chain {nstep - nstep // 4} steps x {nwalk} walkers -> N/tau = {np.round((nstep - nstep//4) * nwalk / tau, 0)}")
S8c = CL.S8_of(ch[:, 0], ch[:, 1])
cols = {p: ch[:, k] for k, p in enumerate(free)}; cols["S8"] = S8c
sF_ml = np.sqrt(np.diag(C))
for k, p in enumerate(free + ["S8"]):
    v = cols[p]; tr = CL.FID[CL.PNAMES.index(p)] if p != "S8" else CL.S8_of(*CL.FID[:2])
    ml = xml[k] if p != "S8" else CL.S8_of(xml[0], xml[1])
    lo, me, hi = np.percentile(v, [15.87, 50, 84.13])
    print(f"  {p:5s}: truth {tr:.4f} | ML {ml:.4f} | MCMC mean {v.mean():.4f} median {me:.4f} std {v.std():.4f} "
          f"(68% half-width {(hi-lo)/2:.4f}) | Fisher sigma at truth {sig_t[p]:.4f} | MCMC std / Fisher = {v.std()/sig_t[p]:.3f} | "
          f"(mean-truth)/sigma_F = {(v.mean()-tr)/sig_t[p]:+.2f}, (ML-truth)/sigma_F = {(ml-tr)/sig_t[p]:+.2f}")
np.savez(f"{CL.DC}/mcmc_v2_{cn}_{dn}_{sv}.npz", chain=ch, free=np.array(free), xml=xml, Cml=C, Ft=Ft, tau=tau)
