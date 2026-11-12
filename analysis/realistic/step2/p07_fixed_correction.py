"""Step 2 / p07: the fixed-correction model (r02's 'recompute once' model) with the truth OFF the fiducial.

   M_fixed(theta) = C_fid(z, q) x N_naive(theta),   C_fid = N_b(fid) / N_n(fid) = (mu_naive(fid) + Delta_fid) / mu_naive(fid)
(SO baseline MMF, dominance fdom = 1; Delta_fid from p05 truth_baseline_f1.npz).  At the fiducial it is exact by
construction.  Truths: S8 = 0.75 and 0.87 at Om = 0.3111 (sigma8 = 0.7365, 0.8543); their blended data
   d_t = mu_naive(theta_t) + Delta_t,   Delta_t from the P(D) recomputed with the neighbours of that truth (p02, p03, p05;
   same rho tables) and the primaries of that truth.
Fitted with M_fixed (C computed at the fiducial) and with the naive model (ignoring blending), configs a-e, SO40 and
full sky, q_cut 5 and 6, priors centred on the truth.  Bias in Fisher sigma of the fitted model at the truth.
Output: data_real/step2/fixed_correction.json
"""
import os, sys, time, json
NPROC = int(os.environ.get("P07_PROCS", "6"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                  # noqa: E402
from s2lib import CL                                # noqa: E402

TRUTHS = {"fid": "baseline_f1", "S8=0.75": "S8_0.75_f1", "S8=0.87": "S8_0.87_f1"}
CFGS = ["a", "b", "c", "d", "d3", "d5", "e"]
SURVEYS = {"SO40": 0.4, "full": 1.0}
_W = {}


def models(qc):
    if qc not in _W:
        if "G" not in _W:
            _W["G"] = CL.HMFGrid()
        base = S.SmoothModelS(qc, S.LnSigma(S.sz_model("baseline")), grid=_W["G"])
        tf = np.load(f"{S.D2}/truth_baseline_f1.npz")
        s = f"{qc:.0f}"
        N0f = base.counts(S.FID)
        assert np.allclose(N0f, tf[f"N0_q{s}"], rtol=1e-12, atol=1e-9)
        Cfid = np.where(N0f > 1e-12, (N0f + tf[f"D_q{s}"]) / np.where(N0f > 1e-12, N0f, 1.0), 1.0)
        _W[qc] = (base, S.FixedCorrModel(base, Cfid), Cfid)
    return _W[qc]


def _job(task):
    tname, model_kind, qc, sv, cn = task
    t = time.time()
    base, fixed, Cfid = models(qc)
    tr = np.load(f"{S.D2}/truth_{TRUTHS[tname]}.npz")
    truth = tr["truth"]
    data = np.clip(base.counts(truth) + tr[f"D_q{qc:.0f}"], 0, None)
    r = S.run_fit(base if model_kind == "ignore" else fixed, data, cn, truth, SURVEYS[sv], polish=True)
    r.update(truth_name=tname, model=model_kind, qcut=qc, survey=sv, sec=time.time() - t)
    return r


def main():
    from multiprocessing import get_context
    t0 = time.time()
    for qc in (5.0, 6.0):
        base, fixed, Cfid = models(qc)
        s = f"{qc:.0f}"
        print(f"q_cut {qc:.0f}: C_fid = N_b/N_n (fiducial) ranges {Cfid.min():.4f}-{Cfid.max():.4f}; total {np.sum(Cfid*base.counts(S.FID))/base.counts(S.FID).sum():.5f}")
        for tname, setup in TRUTHS.items():
            tr = np.load(f"{S.D2}/truth_{setup}.npz")
            N0 = base.counts(tr["truth"]); D = tr[f"D_q{s}"]
            Ct = (N0 + D) / np.where(N0 > 1e-12, N0, 1.0)
            w = N0 > 10
            print(f"   truth {tname:8s} (sigma8 = {tr['truth'][1]:.5f}, S8 = {CL.S8_of(*tr['truth'][:2]):.4f}): N_naive {N0.sum():,.0f}, "
                  f"Delta {D.sum():+,.1f} ({D.sum()/N0.sum():+.3%}); C_truth / C_fid - 1 over bins with N > 10: max |.| "
                  f"{np.abs(Ct/Cfid-1)[w].max():.2e}, in Poisson sigma at full sky (N |C_t - C_fid| / sqrt N): max "
                  f"{np.max(np.abs(Ct-Cfid)[w]*np.sqrt(N0[w])):.2f}, rms {np.sqrt(np.mean((np.abs(Ct-Cfid)*np.sqrt(np.maximum(N0, 0)))**2)):.3f}", flush=True)
    tasks = [(t, m, qc, sv, cn) for t in TRUTHS for m in ("ignore", "fixed") for qc in (5.0, 6.0) for sv in SURVEYS for cn in CFGS]
    print(f"{len(tasks)} fits on {NPROC} processes", flush=True)
    rows = []
    with get_context("spawn").Pool(NPROC) as p:
        for r in p.imap_unordered(_job, tasks, chunksize=1):
            rows.append(r)
    json.dump(rows, open(f"{S.D2}/fixed_correction.json", "w"), indent=1)
    rows.sort(key=lambda r: (r["qcut"], list(SURVEYS).index(r["survey"]), CFGS.index(r["config"]), list(TRUTHS).index(r["truth_name"]),
                             r["model"] != "ignore"))
    cur = None
    for r in rows:
        if (r["qcut"], r["survey"]) != cur:
            cur = (r["qcut"], r["survey"]); print(f"\n######## q_cut = {r['qcut']:.0f}, {r['survey']}")
        nui = "  ".join(f"{p}:{r['b_'+p]:+.2f}s" for p in r["free"][2:])
        print(f"q{r['qcut']:.0f} {r['survey']:4s} {r['config']:2s} truth {r['truth_name']:8s} {r['model']:6s}: "
              f"Om {r['b_Om']:+.2f}s  s8 {r['b_s8']:+.2f}s  S8 {r['b_S8']:+.2f}s (dS8 {r['d_S8']:+.4f}, sig {r['s_S8']:.4f}) | {nui} | "
              f"chi2min {r['dev']:.2f} power {r['power']:.2f} [it {r['nit']}; polish dlnL {r['polish_gain']:+.1e}; at bound {r['at_bound']}]")
    fr = [r for r in rows if r["model"] == "fixed" and r["truth_name"] == "fid"]
    print(f"\nSANITY: fixed correction at the fiducial truth: max |bias| {max(abs(r['b_'+p]) for r in fr for p in r['free'] + ['S8']):.2e} sigma "
          f"over {len(fr)} fits; max chi2min {max(r['dev'] for r in fr):.1e}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
