"""[REPAIR v2] Full-ML patch bootstrap of the map-based (cleaned maps) bias, SO40, q>5, configs b-e
(the linearised bootstrap of c05/c12 is inaccurate where the fit is nonlinear, e.g. slnY at its bound).
NB resamples of the 324 patches; each refit by Fisher scoring + bounded L-BFGS-B, started at the central ML.
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL, cosmolib2 as C2
from c05_fits import CONFIGS

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CF = sys.argv[2].split(",") if len(sys.argv) > 2 else ["b", "c", "d", "e"]
DN = sys.argv[3] if len(sys.argv) > 3 else "clean"
t0 = time.time(); fsky = 0.4
M = C2.SmoothModel(5.0); N0 = M.counts(CL.FID)
tm = np.load(f"{CL.DC}/truth_maps.npz"); sc = float(tm["fullsky_deg2"] / tm["area_deg2"])
H, Hi = tm[DN + "5"], tm["iso5"]
rng = np.random.default_rng(99); B = rng.integers(0, H.shape[0], (NB, H.shape[0]))
S8t = CL.S8_of(*CL.FID[:2]); out = {}
for cn in CF:
    free, pri = CONFIGS[cn]; fi = [CL.PNAMES.index(p) for p in free]
    sig, _ = CL.derived_sigma(C2.Fitter2(M, N0 * fsky, free, fsky=fsky, priors=pri).fisher(CL.FID[fi]), fi)
    d0 = N0 + (H - Hi).sum(0) * sc
    xc, _, _ = C2.Fitter2(M, np.clip(d0, 0, None) * fsky, free, fsky=fsky, priors=pri).fit(polish=True)
    res = []
    for b in B:
        d = np.clip(N0 + (H[b] - Hi[b]).sum(0) * sc, 0, None)
        x, _, _ = C2.Fitter2(M, d * fsky, free, fsky=fsky, priors=pri).fit(x0=xc, polish=True)
        res.append(np.concatenate([x, [CL.S8_of(x[0], x[1])]]))
    res = np.array(res); out[cn] = res.tolist()
    tr = np.concatenate([CL.FID[fi], [S8t]]); cen = np.concatenate([xc, [CL.S8_of(xc[0], xc[1])]])
    sg = np.array([sig[p] for p in free + ["S8"]])
    print(f"config {cn} ({DN}, {NB} ML bootstrap resamples, {time.time()-t0:.0f}s): " + "; ".join(
        f"{p}: central {(cen[k]-tr[k])/sg[k]:+.2f}, boot mean {(res[:, k].mean()-tr[k])/sg[k]:+.2f}, boot std {res[:, k].std(ddof=1)/sg[k]:.2f}"
        for k, p in enumerate(free + ["S8"])), flush=True)
json.dump(out, open(f"{CL.DC}/boot_ml_v2_{DN}.json", "w"))
print(f"total {time.time()-t0:.0f}s")
