"""w06: summary of w05 (part ii): Poisson realisations of the build's blended SO40 data vector (baseline, fdom 1).

Per config (a, d) and data set (blended, null = no blending; paired draws):
  mean (ML - truth) / sigma_Fisher with its error, median, scatter / sigma_Fisher (error 1/sqrt(2(N-1)));
  the paired difference blended - null (the blending shift in the presence of Poisson noise);
  vs the build's Asimov bias (data_real/step2/fits.json, baseline_f1, SO40, q_cut 5) and the Fisher sigma.
"""
import os, sys, glob, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w05_poisson as P5                              # noqa: E402

rows = []
for fn in sorted(glob.glob(os.path.join(HERE, "out", "w05_poisson_*.json"))):
    rows += json.load(open(fn))
w = P5.setup()
S, CL, truth = w["S"], w["CL"], w["truth"]
S8t = CL.S8_of(truth[0], truth[1])
fits = json.load(open(f"{S.D2}/fits.json"))


def asimov(cn):
    for r in fits:
        if r["setup"] == "baseline_f1" and r["config"] == cn and r["survey"] == "SO40" and r["qcut"] == 5.0 \
                and r["route"] == "cat" and r.get("start", "truth") == "truth":
            return r


out = open(os.path.join(HERE, "w06_poisson_summary.log"), "w")


def log(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()


log(f"[w06] {len(rows)} fits from {len(set(r['r'] for r in rows))} realisations (w05); SO40, q_cut 5, SO baseline, fdom 1; "
    f"truth S8 = {S8t:.5f}")
for cn in P5.CFGS:
    free, fi, sig = P5.fisher_sigmas(cn)
    A = asimov(cn)
    log(f"\n=== config {cn}: free {free}; Fisher sigma at the truth: " + ", ".join(f"{k} {v:.5f}" for k, v in sig.items())
        + f"; build Asimov (blended): S8 {A['b_S8']:+.3f}, Om {A['b_Om']:+.3f}, s8 {A['b_s8']:+.3f} sigma")
    rr = {}
    for wh in ("blend", "null"):
        R = sorted([r for r in rows if r["cfg"] == cn and r["which"] == wh], key=lambda r: r["r"])
        X = np.array([r["x"] for r in R])
        S8 = np.array([r["S8"] for r in R])
        nb = sum(bool(r["at_bound"]) for r in R)
        rr[wh] = (np.array([r["r"] for r in R]), X, S8)
        n = len(R)
        log(f"  [{wh:5s}] {n} fits; at a bound: {nb}; mean deviance {np.mean([r['dev'] for r in R]):.1f} (ndof {142 - len(free)})")
        for k, p in enumerate(free + ["S8"]):
            v = (X[:, k] - truth[CL.PNAMES.index(p)]) / sig[p] if p != "S8" else (S8 - S8t) / sig["S8"]
            asv = f"{A['b_' + p]:+.3f}" if wh == "blend" else "+0.000"
            log(f"      {p:5s}: mean bias {v.mean():+.3f} +- {v.std(ddof=1)/np.sqrt(n):.3f} sigma_F (Asimov {asv}); median "
                f"{np.median(v):+.3f}; scatter / sigma_F = {v.std(ddof=1):.3f} +- {1/np.sqrt(2*(n-1)):.3f}")
    # paired difference
    ib, Xb, S8b = rr["blend"]; in_, Xn, S8n = rr["null"]
    common = np.intersect1d(ib, in_)
    kb = np.searchsorted(ib, common); kn = np.searchsorted(in_, common)
    log(f"  [blend - null, paired, {common.size} realisations]")
    for k, p in enumerate(free + ["S8"]):
        if p == "S8":
            dv = (S8b[kb] - S8n[kn]) / sig["S8"]
        else:
            dv = (Xb[kb, k] - Xn[kn, k]) / sig[p]
        log(f"      {p:5s}: mean {dv.mean():+.3f} +- {dv.std(ddof=1)/np.sqrt(common.size):.3f} sigma_F (Asimov {A['b_' + p]:+.3f}); "
            f"scatter of the difference {dv.std(ddof=1):.3f} sigma_F")
log(f"\n[w06] done")
