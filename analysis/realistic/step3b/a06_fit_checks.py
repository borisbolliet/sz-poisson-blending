"""Step 3b arbitration / a06: are the build's ML fits at the maximum of its own likelihood?  (claims K15, P3, B2, B5/B6)

(1) f_sky invariance.  In configs without a prior (a, b, c, e) the Asimov likelihood at f_sky is f_sky x the full-sky one,
    so the ML point must be the same for SO40 and the full sky; only the Fisher sigma changes (ratio sqrt(1/0.4)).  Checked
    on every row of fits.json (theta of SO40 vs full sky; S8 bias ratio).
(2) Careful re-optimisation of selected fits with the build's model and data (b07 R_s): s2lib.run_fit as b09, then
    bounded Nelder-Mead in Fisher-scaled coordinates (tight tolerances) from the build's solution and from the truth,
    each followed by L-BFGS-B with a small finite-difference step.  Reported: the best log-likelihood found against the
    build's, and the S8 bias there.  Cases: the headline (prod_none, SO40) configs d and e at both noise levels (the fits
    verifier's ML for config e is 0.02-0.03 sigma lower); fof 10' goal, full sky, configs d and e (at the sigma_lnY bound;
    the fits verifier's full-sky config e is +6.32 against the build's +7.49).
Usage: python a06_fit_checks.py NPROC
"""
import os, sys, json, time
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
import s3bcosmo as SC                              # noqa: E402
S2, CL = SC.S2, SC.CL
_M = {}
CASES = [("prod_none", "baseline", "SO40", "d"), ("prod_none", "baseline", "SO40", "e"), ("prod_none", "goal", "SO40", "d"),
         ("prod_none", "goal", "SO40", "e"), ("prod_fof10", "goal", "full", "d"), ("prod_fof10", "goal", "full", "e")]


def get(setup, mode):
    key = (setup, mode)
    if key not in _M:
        _M.clear()
        M = SC.model(setup.split("_")[0], mode, 5.0)
        R = np.load(os.path.join(B.OUT, f"datavec_{setup}_{mode}.npz"))["Rs_q5"]
        _M[key] = (M, M.counts(SC.FID), R)
    return _M[key]


def job(case):
    from scipy.optimize import minimize
    setup, mode, sv, cn = case
    t0 = time.time()
    M, N0, R = get(setup, mode)
    data = np.clip(N0 * R, 0, None)
    fs = SC.SURVEYS[sv]
    r = S2.run_fit(M, data, cn, SC.FID, fs, polish=True)
    free, pri = S2.config(cn, SC.FID)
    fi = [CL.PNAMES.index(p) for p in free]
    Fz = S2.Fitter3(M, data * fs, free, fsky=fs, priors=pri, truth=SC.FID)
    lo, hi = Fz._lohi()
    sc = np.sqrt(np.diag(np.linalg.inv(Fz.fisher(SC.FID[fi]))))
    xb = np.array(r["theta"])[fi]
    Lb = Fz.lnL(xb)
    S8t = CL.S8_of(SC.FID[0], SC.FID[1])
    res = dict(case=case, b_S8_build=r["b_S8"], lnL_build=Lb, theta_build=list(xb), at_bound=r["at_bound"], starts=[])
    best = (Lb, xb)
    for lab, x0 in (("build solution", xb), ("truth", SC.FID[fi].copy())):
        ref = x0.copy()
        f = lambda u: -Fz.lnL(np.clip(ref + u * sc, lo, hi))
        bnds = list(zip((lo - ref) / sc, (hi - ref) / sc))
        nm = minimize(f, np.zeros(len(x0)), method="Nelder-Mead", bounds=bnds,
                      options=dict(xatol=1e-5, fatol=1e-11, maxfev=2500, initial_simplex=np.vstack([np.zeros(len(x0)), 0.3 * np.eye(len(x0))])))
        x1 = np.clip(ref + nm.x * sc, lo, hi)
        ref2 = x1.copy()
        lb = minimize(lambda u: -Fz.lnL(np.clip(ref2 + u * sc, lo, hi)), np.zeros(len(x0)), method="L-BFGS-B",
                      bounds=list(zip((lo - ref2) / sc, (hi - ref2) / sc)), options=dict(eps=1e-6, ftol=1e-15, gtol=1e-9, maxiter=800))
        x2 = np.clip(ref2 + lb.x * sc, lo, hi)
        L2 = Fz.lnL(x2)
        th = Fz.full(x2)
        bS8 = (CL.S8_of(th[0], th[1]) - S8t) / r["s_S8"]
        res["starts"].append(dict(start=lab, lnL=L2, b_S8=bS8, theta=list(x2), nfev=int(nm.nfev + lb.nfev)))
        if L2 > best[0]:
            best = (L2, x2)
    th = Fz.full(best[1])
    res.update(lnL_best=best[0], b_S8_best=(CL.S8_of(th[0], th[1]) - S8t) / r["s_S8"], theta_best=list(best[1]), sec=time.time() - t0)
    return res


if __name__ == "__main__":
    t00 = time.time()
    rows = json.load(open(os.path.join(B.OUT, "fits.json")))
    print("(1) f_sky invariance of the ML point in configs without a prior (a, b, c, e): SO40 vs full-sky rows of fits.json", flush=True)
    idx = {}
    for r in rows:
        if r["route"] == "cat":
            idx[(r["setup"], r["mode"], r["estimator"], r["qcut"], r["config"], r["survey"])] = r
    worst, n = (0.0, None), 0
    rat = []
    for k, r in idx.items():
        if k[-1] != "SO40" or k[4] not in ("a", "b", "c", "e"):
            continue
        f = idx.get(k[:-1] + ("full",))
        if f is None:
            continue
        n += 1
        d = float(np.max(np.abs(np.array(r["theta"]) - np.array(f["theta"]))))
        if d > worst[0]:
            worst = (d, k)
        if abs(r["b_S8"]) > 0.05:
            rat.append(f["b_S8"] / r["b_S8"])
    print(f"  {n} pairs: largest |theta_SO40 - theta_full| {worst[0]:.2e} ({worst[1]}); full / SO40 S8-bias ratio {np.min(rat):.4f}-{np.max(rat):.4f} "
          f"(sqrt(1/0.4) = {np.sqrt(2.5):.4f})", flush=True)
    for cn in ("e",):
        for sv in ("SO40", "full"):
            k = ("prod_fof10", "goal", "Rs", 5.0, cn, sv)
            print(f"  fof 10', goal, config {cn}, {sv}: S8 bias {idx[k]['b_S8']:+.3f}, theta {np.round(idx[k]['theta'], 4).tolist()}, "
                  f"at bound {idx[k]['at_bound']}", flush=True)
    print(f"(2) careful re-optimisation ({NPROC} processes)", flush=True)
    from multiprocessing import get_context
    with get_context("spawn").Pool(NPROC) as p:
        out = list(p.imap(job, CASES))
    for res in out:
        s = res["case"]
        print(f"  {s[0]:10s} {s[1]:8s} {s[2]:4s} {s[3]}: build ML S8 {res['b_S8_build']:+.4f} (lnL {res['lnL_build']:.6f}, at bound "
              f"{res['at_bound']}); re-optimised: " + "; ".join(f"from {a['start']}: lnL {a['lnL']:.6f} (+{a['lnL'] - res['lnL_build']:.2e}), "
                                                                   f"S8 {a['b_S8']:+.4f}" for a in res["starts"])
              + f" -> best S8 {res['b_S8_best']:+.4f} [{res['sec']:.0f} s]", flush=True)
    json.dump(out, open(os.path.join(B.OUT, "arbiter", "a06_fit_checks.json"), "w"), indent=1, default=float)
    print(f"done [{time.time() - t00:.0f} s]", flush=True)
