"""w05 (part ii): Poisson realisations of the BUILD's blended SO40 data vector (SO baseline, fdom = 1, q_cut = 5),
fitted with the BUILD's fitter (analysis/realistic/step2/s2lib.Fitter3: Fisher scoring + bounded L-BFGS-B polish
from the scoring solution and from the truth), configs a and d.

Realisation r (seed [2026, 11, 12, 5, r]):
   d_null  ~ Poisson(0.4 N0)                       (naive expectation, no blending)
   d_blend = d_null + Poisson(0.4 Delta)  where Delta >= 0,  Binomial(d_null, (N0 + Delta) / N0) where Delta < 0
             -> d_blend ~ Poisson(0.4 (N0 + Delta)) exactly, and strongly correlated with d_null (paired design)
   config d: the (1-b) prior centre is redrawn per realisation, N(0.8, 0.032), the same draw for both data sets
             (as Q3's c13: an external mass calibration scatters too; then the Fisher sigma incl. the prior is the
             expected scatter).
Tests: mean bias (blended) vs the Asimov bias of the build (p06: a +1.33, d -0.01 sigma in S8); scatter / Fisher sigma;
the paired difference blend - null (blending shift in the presence of Poisson noise, free of the non-linear fit bias).
Env: NR (default 200), R0 (0), NPROC (1).  Output: out/w05_poisson_<R0>.npz
"""
import os, sys, time, json
NPROC = int(os.environ.get("NPROC", "1"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{ROOT}/analysis/realistic/step2")
NR = int(os.environ.get("NR", "200"))
R0 = int(os.environ.get("R0", "0"))
FSKY = 0.4
CFGS = ("a", "d")
_W = {}


def setup():
    if "M" not in _W:
        import s2lib as S
        from s2lib import CL
        G = CL.HMFGrid()
        M = S.SmoothModelS(5.0, S.LnSigma(S.sz_model("baseline")), grid=G)
        truth = S.FID.copy()
        N0 = M.counts(truth)
        D = np.load(f"{S.D2}/truth_baseline_f1.npz")["D_q5"]
        _W.update(S=S, CL=CL, M=M, truth=truth, N0=N0, D=D)
    return _W


def fisher_sigmas(cn):
    w = setup(); S, CL, M, truth, N0 = w["S"], w["CL"], w["M"], w["truth"], w["N0"]
    free, pri = S.config(cn, truth)
    fi = [CL.PNAMES.index(p) for p in free]
    F_ = S.Fitter3(M, N0 * FSKY, free, fsky=FSKY, priors=pri, truth=truth)
    sig, Cov, gS = S.derived_sigma3(F_.fisher(truth[fi]), fi, truth)
    return free, fi, sig


def fit_one(task):
    r, cn, which = task
    w = setup(); S, CL, M, truth, N0, D = w["S"], w["CL"], w["M"], w["truth"], w["N0"], w["D"]
    rng = np.random.default_rng([2026, 11, 12, 5, r])
    dn = rng.poisson(FSKY * N0).astype(float)
    pos = D >= 0
    db = dn.copy()
    db[pos] += rng.poisson(FSKY * D[pos])
    db[~pos] = rng.binomial(dn[~pos].astype(np.int64), np.clip((N0[~pos] + D[~pos]) / N0[~pos], 0, 1))
    omb_c = rng.normal(0.8, 0.032)                       # prior centre draw (used by config d only)
    data = db if which == "blend" else dn
    free, pri = S.config(cn, truth)
    if pri:
        pri = {"omb": (float(omb_c), pri["omb"][1])}
    fi = [CL.PNAMES.index(p) for p in free]
    t = time.time()
    F_ = S.Fitter3(M, data, free, fsky=FSKY, priors=pri, truth=truth)
    x, L, nit = F_.fit(polish=True)
    th = F_.full(x)
    return dict(r=r, cfg=cn, which=which, x=[float(v) for v in x], S8=float(CL.S8_of(th[0], th[1])), dev=float(F_.deviance(x)),
                nit=int(nit), at_bound=list(F_.at_bound), polish_gain=float(F_.polish_gain), omb_c=float(omb_c),
                ndata=float(data.sum()), sec=time.time() - t)


def main():
    from multiprocessing import get_context
    t0 = time.time()
    tasks = [(r, cn, wh) for r in range(R0, R0 + NR) for cn in CFGS for wh in ("blend", "null")]
    print(f"[w05] {NR} realisations from {R0}, configs {CFGS}, blended + null (paired) -> {len(tasks)} fits on {NPROC} "
          f"process(es)", flush=True)
    rows = []
    if NPROC == 1:
        for k, tk in enumerate(tasks):
            rows.append(fit_one(tk))
            if k % 40 == 39:
                print(f"   {k+1}/{len(tasks)} fits [{time.time()-t0:.0f}s]", flush=True)
    else:
        with get_context("spawn").Pool(NPROC) as p:
            for k, rr in enumerate(p.imap_unordered(fit_one, tasks, chunksize=2)):
                rows.append(rr)
                if k % 40 == 39:
                    print(f"   {k+1}/{len(tasks)} fits [{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open(os.path.join(HERE, "out", f"w05_poisson_{R0}.json"), "w"))
    print(f"[w05] saved out/w05_poisson_{R0}.json; {time.time()-t0:.0f}s wall; cpu in fits {sum(r['sec'] for r in rows):.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
