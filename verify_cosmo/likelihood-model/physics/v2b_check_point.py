"""V2b: evaluate my -2lnL (Poisson + priors) for the S2 naive fit at c05's reported best fit vs my best fit."""
import json, numpy as np, v2_fits as F, vlib as v
popf = v.population(nd=F.ND); NB = v.counts_blended(popf).ravel()
names = F.SETUPS["S2"]; pc, ps = F.prior_vecs(names)
mine = json.load(open('/private/tmp/claude-501/-Users-boris-GitHub/0935154e-1cde-4eab-ab64-4515ca671a83/scratchpad/v2_fits_all.json'))["1.0|S2|naive"]["best"]
c05 = json.load(open(v.ROOT + "/data_cosmo/cosmo_bias_fsky1.0.json"))
key = [k for k in c05 if k.startswith("fsky1.0|S2") and k.endswith("|naive")][0]
theirs = c05[key]["best"]
for lab, p in (("mine", mine), ("c05", theirs), ("truth", list(pc))):
    p = np.array(p); mu = F.model(p, names, "naive")
    print(f"{lab:6s} p={np.round(p,4)}  -2lnL = {v.cash(NB, mu) + np.sum(((p-pc)/ps)**2):.3f} (Poisson part {v.cash(NB, mu):.3f})")
# continue my fit from c05's point
p, Fm, c2, nfev, st = F.fit(NB, names, "naive", 1.0, p0=theirs)
sig = np.sqrt(np.diag(np.linalg.inv(Fm)))
print(f"refit from c05 point: p={np.round(p,4)} -2lnL={c2:.3f} status {st} nfev {nfev}; bias/sigma {np.round((p-pc)/sig,2)}; sigma {np.round(sig,4)}")
