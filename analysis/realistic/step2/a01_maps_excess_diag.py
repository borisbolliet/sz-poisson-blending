"""Arbiter / a01: where does the maps verifier's config-c excess come from?  (read-only use of its stored arrays)

The maps verifier (verify_real/step2_maps, w08/w08b) evaluated the build's kernels on the primaries of its 500 painted
patches and found map - kernel = +0.25 +- 0.11 sigma in the S8 bias of goal config c (SO40; 2.2 sigma), the basis of its
'partially' verdict on C5.  Here, from its per-patch arrays (out/w08_sameprimaries.npz: per patch, per setup, the map and
kernel detection sums per z bin and q edge on the SAME primaries):
  (1) per-patch contributions t_p = v . dD_p to the linearised S8 bias difference (v = the build's linear response,
      configs a-e, SO40), their distribution (skewness, largest patches) and leave-out tests;
  (2) which z bins carry the config-c difference;
  (3) the error from patch bootstrap vs a 10-group jackknife;
  (4) how many comparisons the verifier made (look-elsewhere): 12 setups x 5 configs, and the largest |pull|.
Output: a01_maps_excess_diag.log
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                   # noqa: E402
from s2lib import CL                                 # noqa: E402

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "a01_maps_excess_diag.log"), "w")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


W = np.load(f"{S.ROOT}/verify_real/step2_maps/out/w08_sameprimaries.npz")
names = list(W["names"]); A_map, A_bld, SCALE = W["A_map"], W["A_bld"], float(W["scale"])
NP = A_map.shape[0]
EDGES = S.EDGES
ZED = CL.default_bins(5.0)[0]


def qbins(C, qcut=5.0):
    qe = np.geomspace(qcut, 100.0, 9)
    ix = [int(np.argmin(np.abs(EDGES - e))) for e in qe]
    out = np.empty(C.shape[:-1] + (9,))
    for k in range(8):
        out[..., k] = C[..., ix[k]] - C[..., ix[k + 1]]
    out[..., 8] = C[..., ix[8]]
    return out


G = CL.HMFGrid()
inp = np.load(f"{S.D2}/inputs.npz")
TH = {m: np.load(f"{S.DR}/pd_tables_{m}.npz")["th"] for m in ("baseline", "goal")}
MOD = {}


def response(nm, sv):
    """linear S8-bias response vectors v_c (per bin, per count) for configs a-e at the truth of setup nm."""
    m = nm.split("_")[0]
    truth = S.FID_LOW.copy() if m == "lowP" else S.FID.copy()
    run = "baseline" if m == "lowP" else m
    key = (nm if "eff" in nm else run + ("_low" if m == "lowP" else ""))
    if key not in MOD:
        if run == "white":
            from s2lib import C2
            M = C2.SmoothModel(qcut=5.0, grid=G)
        elif "eff" in nm:
            M = S.SmoothModelS(5.0, S.LnSigma(S.sz_model(run), TH[run], inp[f"{run}_r_{nm.split('eff')[1]}"]), grid=G)
        else:
            M = S.SmoothModelS(5.0, S.LnSigma(S.sz_model(run)), grid=G)
        MOD[key] = (M, M.counts(truth), truth)
    M, N0, truth = MOD[key]
    fs = {"SO40": 0.4, "full": 1.0}[sv]
    out = {}
    for cn in ("a", "b", "c", "d", "e"):
        free, pri = S.config(cn, truth)
        fi = [CL.PNAMES.index(p) for p in free]
        F_ = S.Fitter3(M, N0 * fs, free, fsky=fs, priors=pri, truth=truth)
        Ft = F_.fisher(truth[fi]); J = F_.jac(truth[fi])
        sig, _, gS = S.derived_sigma3(Ft, fi, truth)
        out[cn] = (gS @ (np.linalg.solve(Ft, J.T) / N0.ravel()[None, :]) / sig["S8"]).reshape(N0.shape)
    return out


rng = np.random.default_rng(11)
NB = 4000
cnt = np.stack([np.bincount(rng.integers(0, NP, NP), minlength=NP) for _ in range(NB)]).astype(float)
grp = np.arange(NP) % 10

log(f"[a01] maps verifier arrays: {NP} patches x {len(names)} setups; full-sky scale {SCALE:.4f}")
log("\n=== (1) per-patch contributions to the linearised S8-bias difference (map - kernel, same primaries), SO40 ===")
log("    columns: total | bootstrap sigma | jackknife (10 groups) sigma | skewness of t_p | share of the 5 largest |t_p| |"
    " total without the 1 / 5 most positive patches")
pulls = []
for nm in ("goal_f1", "baseline_f1", "goal_f0.5", "goal_f2", "baseline_f0.5", "baseline_f2", "white_f1", "baseline_effi",
           "baseline_effii", "goal_effi", "goal_effii", "lowP_f1"):
    iv = names.index(nm)
    dD = qbins(A_map[:, iv] - A_bld[:, iv]) * SCALE                          # (NP, 16, 9) full sky counts
    V = response(nm, "SO40")
    for cn in ("a", "b", "c", "d", "e"):
        t = (dD * V[cn][None]).sum((1, 2))                                     # per-patch contribution
        tot = t.sum(); bs = (cnt @ t).std()
        jk = np.array([t[grp != g].sum() * NP / (grp != g).sum() for g in range(10)])
        sjk = np.sqrt(9.0 / 10.0 * ((jk - jk.mean()) ** 2).sum())
        sk = float(((t - t.mean()) ** 3).mean() / t.std() ** 3)
        o = np.argsort(-t)
        top5 = np.abs(t)[np.argsort(-np.abs(t))[:5]].sum() / np.abs(t).sum()
        pulls.append((nm, cn, tot / bs))
        if nm in ("goal_f1", "baseline_f1") or cn in ("c", "e"):
            log(f"  [{nm:14s}] {cn}: {tot:+.3f} | {bs:.3f} | {sjk:.3f} | skew {sk:+.2f} | top-5 share {top5:.3f} | "
                f"w/o top 1: {t[o[1:]].sum():+.3f}, w/o top 5: {t[o[5:]].sum():+.3f}")

log("\n=== (2) goal_f1, config c (SO40): z bins carrying the difference (sum over q bins, full-sky counts and bias share) ===")
iv = names.index("goal_f1")
dD = qbins(A_map[:, iv] - A_bld[:, iv]) * SCALE
V = response("goal_f1", "SO40")
Dz = dD.sum(0)                                     # (16, 9)
bz = (Dz * V["c"]).sum(1)
bzb = np.einsum("bp,pzq,zq->bz", cnt, dD, V["c"])
for k in range(16):
    log(f"   z {ZED[k]:.2f}-{ZED[k+1]:.2f}: map - kernel {Dz[k].sum():+7.1f} counts; config-c S8 share {bz[k]:+.3f} +- "
        f"{bzb[:, k].std():.3f} sigma")
hz = ZED[:-1] >= 1.5
log(f"   z >= 1.5: share {bz[hz].sum():+.3f} +- {bzb[:, hz].sum(1).std():.3f} sigma; z < 1.5: {bz[~hz].sum():+.3f} +- "
    f"{bzb[:, ~hz].sum(1).std():.3f} sigma")

log("\n=== (3) look-elsewhere: pulls of the 60 (setup, config) comparisons at SO40 (12 setups x configs a-e) ===")
P = np.array([p for _, _, p in pulls])
i = int(np.argmax(np.abs(P)))
log(f"   largest |pull| {abs(P[i]):.2f} ({pulls[i][0]}, config {pulls[i][1]}); number with |pull| > 2: {(np.abs(P) > 2).sum()}; "
    f"> 1.5: {(np.abs(P) > 1.5).sum()}; mean pull {P.mean():+.2f}")
for cn in ("a", "b", "c", "d", "e"):
    pc = np.array([p for _, c, p in pulls if c == cn])
    log(f"   config {cn}: mean pull over the 12 setups {pc.mean():+.2f} (they share skies, so they are strongly correlated)")
log(f"\n[a01] done {time.time()-t0:.0f}s")
