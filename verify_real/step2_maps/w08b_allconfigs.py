"""w08b: implied S8 bias of (map - build kernel) on the SAME primaries (w08's arrays) for every config a-e,
SO40 and full sky, q_cut 5: the part of any map-vs-build bias difference that is due to the P(D) kernel itself.
Bootstrap over patches (2000).  Also the correlation of these differences across configs (they share the skies).
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{ROOT}/analysis/realistic/step2")
import s2lib as S                                       # noqa: E402  (naive model and fitter only)
from s2lib import CL, C2                                # noqa: E402

d = np.load(os.path.join(HERE, "out", "w08_sameprimaries.npz"))
names = list(d["names"]); A_map, A_bld, SCALE = d["A_map"], d["A_bld"], float(d["scale"])
EDGES = S.EDGES
inp = np.load(f"{ROOT}/data_real/step2/inputs.npz")
TH = {m: np.load(f"{ROOT}/data_real/pd_tables_{m}.npz")["th"] for m in ("baseline", "goal")}
f = open(os.path.join(HERE, "w08b_allconfigs.log"), "w")


def log(s=""):
    print(s, flush=True); f.write(s + "\n"); f.flush()


def qbins(C, qcut=5.0):
    qe = np.geomspace(qcut, 100.0, 9)
    ix = [int(np.argmin(np.abs(EDGES - e))) for e in qe]
    out = np.empty(C.shape[:-1] + (9,))
    for k in range(8):
        out[..., k] = C[..., ix[k]] - C[..., ix[k + 1]]
    out[..., 8] = C[..., ix[8]]
    return out


NP = A_map.shape[0]
rng = np.random.default_rng(11)
cnt = np.stack([np.bincount(rng.integers(0, NP, NP), minlength=NP) for _ in range(2000)]).astype(float)
G = CL.HMFGrid()
MOD = {}
log("[w08b] S8 bias implied by (map - build kernel) on the same primaries, linearised, q_cut 5 (bootstrap 1 sigma)")
for nm in ("baseline_f1", "goal_f1", "baseline_f0.5", "goal_f0.5", "baseline_f2", "goal_f2", "white_f1",
           "baseline_effi", "baseline_effii", "goal_effi", "goal_effii", "lowP_f1"):
    iv = names.index(nm)
    m = nm.split("_")[0]
    truth = S.FID_LOW.copy() if m == "lowP" else S.FID.copy()
    run = "baseline" if m == "lowP" else m
    key = nm if "eff" in nm else (run + ("_low" if m == "lowP" else ""))
    if key not in MOD:
        if run == "white":
            M = C2.SmoothModel(qcut=5.0, grid=G)
        elif "eff" in nm:
            r = inp[f"{run}_r_{nm.split('eff')[1]}"]
            M = S.SmoothModelS(5.0, S.LnSigma(S.sz_model(run), TH[run], r), grid=G)
        else:
            M = S.SmoothModelS(5.0, S.LnSigma(S.sz_model(run)), grid=G)
        MOD[key] = (M, M.counts(truth))
    M, N0 = MOD[key]
    dD = qbins(A_map[:, iv] - A_bld[:, iv]) * SCALE
    tot = dD.sum(0).ravel(); bt = np.tensordot(cnt, dD, axes=(1, 0)).reshape(len(cnt), -1)
    for sv, fs in (("SO40", 0.4), ("full", 1.0)):
        outs = []
        vals = []
        for cn in ("a", "b", "c", "d", "e"):
            free, pri = S.config(cn, truth)
            fi = [CL.PNAMES.index(p) for p in free]
            F_ = S.Fitter3(M, N0 * fs, free, fsky=fs, priors=pri, truth=truth)
            Ft = F_.fisher(truth[fi]); J = F_.jac(truth[fi])
            sig, Cov, gS = S.derived_sigma3(Ft, fi, truth)
            v = gS @ (np.linalg.solve(Ft, J.T) / N0.ravel()[None, :]) / sig["S8"]
            outs.append(f"{cn} {tot @ v:+.3f}+-{(bt @ v).std():.3f} ({(tot @ v)/(bt @ v).std():+.1f}s)")
            vals.append(bt @ v)
        cc = np.corrcoef(np.array(vals))
        log(f"  [{nm:14s}] {sv:4s}: " + "; ".join(outs) + f" | corr(c, e) {cc[2, 4]:+.2f}, corr(a, c) {cc[0, 2]:+.2f}")
log("[w08b] done")
