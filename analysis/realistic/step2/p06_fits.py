"""Step 2 / p06: cosmology bias from ignoring blending, realistic SO MMF.  Same experiment as c12_fits_v2:
Asimov maximum-likelihood fits of the NAIVE (no-blending) model to the BLENDED expectation
   data = mu_naive(truth) + Delta        (Delta from p05; 'cat' route)
and of the naive model to its own expectation ('self' route: must return the truth).
Configs (free parameters; the rest fixed at the truth):
   a: Om, s8   b: + (1-b)   c: + (1-b), slnY   d / d3 / d5: + (1-b), alpha, slnY with a 4% / 3% / 5% Gaussian prior on (1-b)
   centred on the truth (0.8 +- 0.032; low-pressure population 0.6 +- 0.024)   e: same, no prior.
Surveys: SO40 (f_sky = 0.4) and full sky; q_cut = 5 and 6 (bins of cosmolib.default_bins).
Bias = (ML - truth) / sigma_Fisher(truth) (marginal, incl. prior); 'lin' = linearised estimator F^-1 J^T (d - mu)/mu.
Goodness of fit: Asimov deviance chi2_min; power = P(noncentral chi2(ndof, chi2_min) > 95% quantile of chi2(ndof)).
Fitter: cosmolib2.Fitter2 (Fisher scoring + bounded L-BFGS-B polish from the scoring solution and from the truth),
with the truth configurable (s2lib.Fitter3).
Extra setups: 'Q3_c11' = Q3's own data vector (truth_catalogue_v2.npz) with cosmolib2.SmoothModel (must reproduce
c12_fits_v2_q5/q6.log 'cat'); 'white_prefix' (step-2 code, b03 pre-fix rates).
Non-trivial self check: config a and d, SO40, q_cut 5, started 2 sigma off the truth in every parameter.
Output: data_real/step2/fits.json
"""
import os, sys, time, json
NPROC = int(os.environ.get("P06_PROCS", "6"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                  # noqa: E402
from s2lib import CL, C2                            # noqa: E402

CAT = ["Q3_c11", "white", "white_prefix", "baseline_f1", "baseline_f0.5", "baseline_f2", "goal_f1", "goal_f0.5", "goal_f2",
       "baseline_effi", "baseline_effii", "goal_effi", "goal_effii", "lowP_f1", "lowP_effi"]
# distinct naive models for the self route: (label, a setup that defines the model)
SELF = [("white", "white"), ("baseline", "baseline_f1"), ("goal", "goal_f1"), ("baseline_effi", "baseline_effi"),
        ("baseline_effii", "baseline_effii"), ("goal_effi", "goal_effi"), ("goal_effii", "goal_effii"),
        ("lowP", "lowP_f1"), ("lowP_effi", "lowP_effi")]
CFGS = ["a", "b", "c", "d", "d3", "d5", "e"]
SURVEYS = {"SO40": 0.4, "full": 1.0}
QCUTS = [5.0, 6.0]
_W = {}


def model_and_data(setup, qc):
    """count model, truth and blended data vector (full sky) for a setup; cached per worker."""
    key = (setup, qc)
    if key in _W:
        return _W[key]
    if "G" not in _W:
        _W["G"] = CL.HMFGrid()
    G = _W["G"]
    s = f"{qc:.0f}"
    if setup == "Q3_c11":
        M = C2.SmoothModel(qcut=qc, grid=G); truth = S.FID.copy()
        D = np.load(f"{S.ROOT}/data_cosmo/truth_catalogue_v2.npz")[f"D_q{s}"]
    else:
        t = np.load(f"{S.D2}/truth_{setup}.npz")
        k = np.load(f"{S.D2}/kernel_{setup}.npz")
        truth = t["truth"]; D = t[f"D_q{s}"]
        if setup.startswith("white"):
            lnsig = S.LnSigma(S.sz_model("white"))
        else:
            run = "goal" if setup.startswith("goal") else "baseline"
            lnsig = S.LnSigma(S.sz_model(run), k["th"], k["r"]) if bool(k["eff"]) else S.LnSigma(S.sz_model(run))
        M = S.SmoothModelS(qc, lnsig, grid=G)
    N0 = M.counts(truth)
    _W[key] = (M, truth, N0, D)
    return _W[key]


def _job(task):
    setup, route, qc, sv, cn, start = task
    t = time.time()
    M, truth, N0, D = model_and_data(setup, qc)
    data = N0 + D if route == "cat" else N0.copy()
    nneg = int(np.sum(data < 0))
    data = np.clip(data, 0, None)
    x0 = None
    if start == "offset":
        free, pri = S.config(cn, truth)
        fi = [CL.PNAMES.index(p) for p in free]
        F_ = S.Fitter3(M, N0 * SURVEYS[sv], free, fsky=SURVEYS[sv], priors=pri, truth=truth)
        sig = np.sqrt(np.diag(np.linalg.inv(F_.fisher(truth[fi]))))
        lo, hi = F_._lohi()
        x0 = np.clip(truth[fi] + 2 * sig * np.array([1, -1, 1, -1, 1][:len(fi)]), lo, hi)
    r = S.run_fit(M, data, cn, truth, SURVEYS[sv], polish=(route != "self" or start == "offset"), x0=x0)
    r.update(setup=setup, route=route, qcut=qc, survey=sv, start=start, nneg=nneg, sec=time.time() - t)
    return r


def line(r):
    nui = "  ".join(f"{p}:{r['d_'+p]:+.4f}({r['b_'+p]:+.2f}s)" for p in r["free"][2:])
    return (f"q{r['qcut']:.0f} {r['survey']:4s} {r['config']:2s} {r['setup']:14s} {r['route']:4s}{'*' if r['start']=='offset' else ' '}: "
            f"dOm={r['d_Om']:+.4f} ({r['b_Om']:+.2f}s, lin {r['lin_Om']:+.2f})  ds8={r['d_s8']:+.4f} ({r['b_s8']:+.2f}s, lin {r['lin_s8']:+.2f})"
            f"  dS8={r['d_S8']:+.4f} ({r['b_S8']:+.2f}s, lin {r['lin_S8']:+.2f}) | sig(Om,s8,S8)=({r['s_Om']:.4f},{r['s_s8']:.4f},{r['s_S8']:.4f})"
            f" | {nui} | chi2min={r['dev']:.1f} ndof={r['ndof']} power={r['power']:.2f} [it {r['nit']}; LBFGSB dlnL "
            f"{r['polish_gain']:+.1e}, shift {r['polish_shift']:.2f}s, at bound {r['at_bound']}; neg bins {r['nneg']}]")


def main():
    from multiprocessing import get_context
    t0 = time.time()
    only = os.environ.get("SETUPS")
    cat = [s for s in CAT if (not only or s in only.split(",")) and (s == "Q3_c11" or os.path.exists(f"{S.D2}/truth_{s}.npz"))]
    selfm = [(lab, s) for lab, s in SELF if s in cat]
    tasks = [(s, "cat", qc, sv, cn, "truth") for s in cat for qc in QCUTS for sv in SURVEYS for cn in CFGS]
    tasks += [(s, "self", qc, sv, cn, "truth") for _, s in selfm for qc in QCUTS for sv in SURVEYS for cn in CFGS]
    tasks += [(s, "self", 5.0, "SO40", cn, "offset") for _, s in selfm for cn in ("a", "d")]
    print(f"{len(tasks)} fits ({len(cat)} blended data vectors, {len(selfm)} self models) on {NPROC} processes", flush=True)
    rows = []
    with get_context("spawn").Pool(NPROC) as p:
        for k, r in enumerate(p.imap_unordered(_job, tasks, chunksize=1)):
            rows.append(r)
            if k % 50 == 49:
                print(f"   {k+1}/{len(tasks)} fits [{time.time()-t0:.0f}s]", flush=True)
    for r in rows:
        if r["route"] == "self":
            r["setup_label"] = dict((s, lab) for lab, s in SELF)[r["setup"]]
    json.dump(rows, open(f"{S.D2}/fits.json", "w"), indent=1)
    key = lambda r: (r["qcut"], list(SURVEYS).index(r["survey"]), CFGS.index(r["config"]), r["route"] != "cat",
                     (CAT.index(r["setup"]) if r["setup"] in CAT else 99), r["start"])
    rows.sort(key=key)
    cur = None
    for r in rows:
        if (r["qcut"], r["survey"]) != cur:
            cur = (r["qcut"], r["survey"]); print(f"\n######## q_cut = {r['qcut']:.0f}, {r['survey']}")
        print(line(r))
    # sanity summaries
    sr = [r for r in rows if r["route"] == "self"]
    mx = max(abs(r[f"b_{p}"]) for r in sr for p in r["free"])
    print(f"\nSANITY: naive model fitted to its own expectation: max |bias| over {len(sr)} fits (incl. {sum(r['start']=='offset' for r in sr)} "
          f"started 2 sigma off) and all free params = {mx:.2e} sigma; max chi2min = {max(r['dev'] for r in sr):.2e}")
    cr = [r for r in rows if r["route"] == "cat"]
    print(f"L-BFGS-B polish after Fisher scoring: max lnL gain {max(r['polish_gain'] for r in cr):.2e}, max shift "
          f"{max(r['polish_shift'] for r in cr):.3f} sigma over {len(cr)} fits; fits with a parameter at a bound: "
          f"{sum(bool(r['at_bound']) for r in cr)}; data bins < 0 clipped: {max(r['nneg'] for r in rows)}")
    # Q3 reproduction
    c12 = {}
    for fn in ("c12_fits_v2_q5.log", "c12_fits_v2_q6.log"):
        for ln in open(f"{S.ROOT}/analysis/cosmo/bias/{fn}"):
            if " cat      :" in ln:
                tok = ln.split()
                c12[(float(tok[0][1:]), tok[1], tok[2])] = ln
    worst = 0.0
    for r in rows:
        if r["setup"] == "Q3_c11" and r["route"] == "cat":
            ln = c12.get((r["qcut"], r["survey"], r["config"]))
            if ln is None:
                continue
            b = float(ln.split("dS8=")[1].split("(")[1].split("s")[0])
            worst = max(worst, abs(round(r["b_S8"], 2) - b))
    print(f"Q3 reproduction: max |b_S8 (this code, Q3 data) - c12_fits_v2 log| = {worst:.3f} sigma over all configs/surveys/q_cuts")
    print(f"total {time.time()-t0:.0f}s; cpu in fits {sum(r['sec'] for r in rows):.0f}s")


if __name__ == "__main__":
    main()
