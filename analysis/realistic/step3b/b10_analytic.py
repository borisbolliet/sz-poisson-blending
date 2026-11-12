"""Step 3b / b10: the analytic analyst.  Instead of the measured kernel, the analyst models the finder's selection with
Phi(sqrt(q'^2 + k) - e), q' = Y500 / sigma_eff (sigma_eff = the variant's rbar sigma_Y, as b09):
  k = 0  (steps 1-2: unit Gaussian noise on the true S/N)
  k = 2, 3  (the optimisation-bias corrections used for ACT / SPT / SZiFi-type selection: q_obs ~ N(sqrt(q^2 + k), 1)).
Two data vectors per variant and noise level (q_cut 5, full sky, scaled by f_sky):
  (i)  blended: mu_K(truth) x R_s (the finder's catalogue: b05 kernel times b07's pooled blending ratio);
  (ii) no blending: mu_K(truth) (isolated clusters only: the finder's own selection without neighbours).
(ii) measures the finder-systematics part (how wrong the analytic kernel is for isolated clusters); (i) - (ii) the part
due to blending.  Variants prod and szifi3000; configs a, d, e; SO40 and full sky.  ML fits (s2lib.run_fit), plus the
tile-bootstrap error of (i) (linearised at the truth with the analytic model, b07's R_s resamples).
With the argument 'cl' the analytic kernel is set to zero below the credit limit q' = 2 (no credited detection is
possible there; without it Phi(sqrt(k) - 5) ~ 2e-4 - 5e-4 is given to every faint halo of the M500 > 1e13 population).
Output: data_real/step3b/fits_analytic.json (fits_analytic_cl.json).  Usage: python b10_analytic.py NPROC [cl]
"""
import os, sys, time, json
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CL_ = len(sys.argv) > 2 and sys.argv[2] == "cl"     # 'cl': the analytic kernel set to 0 below the credit limit q' = 2
KERN = "analytic_cl" if CL_ else "analytic"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
import s3bcosmo as SC                              # noqa: E402
S2 = SC.S2
KS = (0.0, 2.0, 3.0)
VARS = ("prod", "szifi3000")
CF = ("a", "d", "e")
_W = {}


def get(variant, mode, k):
    key = (variant, mode, k)
    if key not in _W:
        for kk in [kk for kk in _W if kk[0] != "v"][:-1]:              # keep at most 2 analytic models per process
            del _W[kk]
        Mk = SC.model(variant, mode, 5.0, kernel=KERN, k=k)
        if ("v", variant, mode) not in _W:
            Mv = SC.model(variant, mode, 5.0)
            _W[("v", variant, mode)] = Mv.counts(SC.FID)
        _W[key] = Mk
    return _W[key], _W[("v", variant, mode)]


def _job(task):
    variant, mode, k, route, sv, cn = task
    t = time.time()
    Mk, Nv = get(variant, mode, k)
    dv = np.load(os.path.join(B.OUT, f"datavec_{variant}_none_{mode}.npz"))
    data = Nv * dv["Rs_q5"] if route == "blended" else Nv.copy()
    r = S2.run_fit(Mk, np.clip(data, 0, None), cn, SC.FID, SC.SURVEYS[sv], polish=True)
    # bootstrap error of the blended route, linearised with the analytic model at the truth
    if route == "blended":
        free, pri = S2.config(cn, SC.FID)
        fi = [SC.CL.PNAMES.index(p) for p in free]
        Nk = Mk.counts(SC.FID)
        F_ = S2.Fitter3(Mk, Nk * SC.SURVEYS[sv], free, fsky=SC.SURVEYS[sv], priors=pri, truth=SC.FID)
        Ft = F_.fisher(SC.FID[fi]); J = F_.jac(SC.FID[fi])
        sig, Cov, gS = S2.derived_sigma3(Ft, fi, SC.FID)
        RB = dv["Rsboot_q5"].astype(float)
        X = ((RB - dv["Rs_q5"]) * Nv / Nk).reshape(RB.shape[0], -1)
        x = np.linalg.solve(Ft, J.T @ X.T).T
        r["boot_S8"] = float((x @ gS / sig["S8"]).std())
    r.update(variant=variant, mode=mode, k=k, route=route, survey=sv, sec=time.time() - t)
    return r


if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    tasks = [(v, m, k, route, sv, cn) for v in VARS for m in B.MODES for k in KS for route in ("blended", "isolated")
             for sv in SC.SURVEYS for cn in CF]
    print(f"b10: {len(tasks)} ML fits of the analytic analyst ({KERN}) on {NPROC} processes", flush=True)
    rows = []
    with get_context("spawn").Pool(NPROC) as p:
        for r in p.imap_unordered(_job, tasks, chunksize=1):
            rows.append(r)
    json.dump(rows, open(os.path.join(B.OUT, "fits_analytic_cl.json" if CL_ else "fits_analytic.json"), "w"), indent=1)
    print("S8 bias [sigma] of the analytic analyst, q_cut 5: total (i) = finder systematics (ii) + blending (i) - (ii); "
          "(i) +- tile bootstrap; Om, s8 of (i)")
    for v in VARS:
        for m in B.MODES:
            for sv in SC.SURVEYS:
                for cn in CF:
                    parts = []
                    for k in KS:
                        a = [r for r in rows if r["variant"] == v and r["mode"] == m and r["survey"] == sv and r["config"] == cn
                             and r["k"] == k]
                        bl = [r for r in a if r["route"] == "blended"][0]; iso = [r for r in a if r["route"] == "isolated"][0]
                        parts.append(f"k={k:.0f}: {bl['b_S8']:+.2f}+-{bl['boot_S8']:.2f} = {iso['b_S8']:+.2f} + {bl['b_S8'] - iso['b_S8']:+.2f} "
                                     f"(Om {bl['b_Om']:+.2f}, s8 {bl['b_s8']:+.2f}; power {bl['power']:.2f}/{iso['power']:.2f})")
                    print(f"  {v:9s} {m:8s} {sv:4s} {cn}: " + ";  ".join(parts), flush=True)
    print(f"fits with a parameter at a bound: {sum(bool(r['at_bound']) for r in rows)}; max polish shift {max(r['polish_shift'] for r in rows):.3f} sigma")
    print(f"total {time.time() - t0:.0f} s; cpu in fits {sum(r['sec'] for r in rows):.0f} s")
