"""Arbiter / a05: independent checks of the claims neither verifier tested (C23, C24, C25 of the build's list).

(1) C23 on the STORED production kernels (not recomputed): kernel_<setup>.npz interpolated in q_A to step 1's s09 q
    nodes vs s09's stored per-node tables T{fdom}_det (q_cut 5, 6, 7), inner nodes 0.25-80' (s09's stored extra variance
    outside that range was the spurious term the step-1 arbiter removed).
(2) C24 by a different route: theta interpolation.  The build integrates the kernel linearly in ln theta between its 48
    nodes.  Here the kernel is refined to 95 nodes by CUBIC interpolation in ln theta (midpoints) and the data vector is
    recomputed (still linear between the 95 nodes).  The change estimates the interpolation error of production.
(3) C25: step 1's s09 rows (P(D) expectation, primaries q_A > 0.5) vs the build's truth vectors restricted to q_A >= 0.5.
Output: a05_quick_checks.log
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                   # noqa: E402
from s2lib import CL, pl                             # noqa: E402

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "a05_quick_checks.log"), "w")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


log("=== (1) stored kernels vs s09 node tables T{fdom}_det (q_A interpolated linearly onto s09's q nodes) ===")
for setup, run, f in (("baseline_f1", "baseline", 1.0), ("baseline_f0.5", "baseline", 0.5), ("baseline_f2", "baseline", 2.0),
                      ("goal_f1", "goal", 1.0), ("goal_f0.5", "goal", 0.5), ("goal_f2", "goal", 2.0), ("white", "white", 1.0)):
    k = np.load(f"{S.D2}/kernel_{setup}.npz")
    d = np.load(f"{S.DR}/pd_tables_{run}.npz")
    th, QA, Kb, E = k["th"], k["QA"], k["Kb"], k["EDGES"]
    qn = d["qnodes"]
    inner = np.nonzero((th >= 0.25 - 1e-9) & (th <= 80 + 1e-9) & (d["extra"] == 0))[0]
    worst = 0.0; where = None
    for qc in (5.0, 6.0, 7.0):
        jc = int(np.argmin(np.abs(E - qc))); ic = int(np.argmin(np.abs(d["qcuts"] - qc)))
        for i in inner:
            ks = np.nonzero((d["H"][i] != 0) & (qn >= QA[0]) & (qn <= 30.0))[0]
            Ki = np.interp(qn[ks], QA, Kb[i, :, jc])
            ref = d[f"T{f}_det"][ic, i, ks]
            dd = np.abs(Ki - ref).max() if ks.size else 0.0
            if dd > worst:
                worst, where = dd, (round(float(th[i]), 3), qc)
    log(f"   [{setup:13s}] {len(inner)} inner nodes, q_cut 5/6/7, s09 q nodes in 0.05-30: max |K_stored - T_det| = {worst:.1e} "
        f"(at theta {where[0]}', q_cut {where[1]:.0f})")

log("\n=== (2) theta interpolation: 48 nodes (production, linear in ln theta) vs 95 nodes (cubic midpoints) ===")
G = CL.HMFGrid()
for setup in ("baseline_f1", "goal_f1", "white", "baseline_effi"):
    k = np.load(f"{S.D2}/kernel_{setup}.npz")
    t = np.load(f"{S.D2}/truth_{setup}.npz")
    run = "white" if setup == "white" else setup.split("_")[0]
    th, Kb = k["th"], k["Kb"]
    lt = np.log(th)
    lt2 = np.sort(np.concatenate([lt, 0.5 * (lt[1:] + lt[:-1])]))
    K2 = CubicSpline(lt, Kb, axis=0)(lt2)
    K2 = np.clip(K2, 0.0, 1.0)
    for qc in (5.0,):
        s = f"{qc:.0f}"
        if bool(k["eff"]):
            r2 = np.exp(CubicSpline(lt, np.log(k["r"]))(lt2))
            M1 = S.SmoothModelS(qc, S.LnSigma(S.sz_model(run), th, k["r"]), grid=G)
            M2 = S.SmoothModelS(qc, S.LnSigma(S.sz_model(run), np.exp(lt2), r2), grid=G)
        else:
            lns = S.LnSigma(S.sz_model(run))
            M1 = S.SmoothModelS(qc, lns, grid=G); M2 = M1
        Nb1, Nn1, N0 = S.delta_counts(M1, Kb, th, k["QA"], k["EDGES"], t["truth"])
        Nb2, Nn2, _ = S.delta_counts(M2, K2, np.exp(lt2), k["QA"], k["EDGES"], t["truth"])
        D1, D2 = Nb1 - Nn1, Nb2 - Nn2
        x = np.abs(D2 - D1) / np.sqrt(N0 + 1e-9)
        log(f"   [{setup:13s}] q>{s}: Delta {D1.sum():+.1f} (48 nodes) vs {D2.sum():+.1f} (95 nodes, cubic midpoints): total diff "
            f"{D2.sum()-D1.sum():+.2f} counts ({(D2.sum()-D1.sum())/N0.sum():+.1e} of N); per bin max {x.max():.4f} Poisson sigma "
            f"(full sky), at SO40 {x.max()*np.sqrt(0.4):.4f}; production data vector check: |D1 - stored| max "
            f"{np.abs(D1 - t['D_q5']).max():.1e}")

log("\n=== (3) step 1 (s09 rows: N_det,b / N_single - 1, primaries q_A > 0.5) vs the build (truth D05 / Nn05), q > 5 ===")
for setup, run, f in (("baseline_f1", "baseline", 1.0), ("baseline_f0.5", "baseline", 0.5), ("baseline_f2", "baseline", 2.0),
                      ("goal_f1", "goal", 1.0), ("goal_f0.5", "goal", 0.5), ("goal_f2", "goal", 2.0), ("white", "white", 1.0)):
    d = np.load(f"{S.DR}/pd_tables_{run}.npz")
    row = d[f"rows_fdom{f}"][1]
    t = np.load(f"{S.D2}/truth_{setup}.npz")
    a = row[2] / row[1] - 1; b = t["Nb05_q5"].sum() / t["Nn05_q5"].sum() - 1
    log(f"   [{setup:13s}] step 1 {100*a:+.4f}% | step 2 {100*b:+.4f}% | difference {100*(b-a):+.4f} pp")

log("\n=== (4) sensitivity of the S8 bias to the open per-redshift offsets (linear response, q_cut 5) ===")
log("    perturbation: Delta/N changed by a fixed amount (pp) in every q bin of a redshift range (dN = shift x N_naive)")
zed = CL.default_bins(5.0)[0]
PERT = (("z 2-3, -0.3 pp (step 1 direct counts, goal)", 2.0, 3.0, -0.003),
        ("z 2-3, -0.5 pp (largest step-1 offset)", 2.0, 3.0, -0.005),
        ("z 1-1.5, +0.1 pp (catalogue verifier)", 1.0, 1.5, +0.001),
        ("z 1.5-3, +0.7 pp (maps verifier, map - kernel, goal)", 1.5, 3.0, +0.007))
for setup in ("baseline_f1", "goal_f1"):
    run = setup.split("_")[0]
    M = S.SmoothModelS(5.0, S.LnSigma(S.sz_model(run)), grid=G)
    tr = S.FID.copy()
    N0 = M.counts(tr)
    for sv, fs in (("SO40", 0.4), ("full", 1.0)):
        V = {}
        for cn in ("a", "b", "c", "d", "e"):
            free, pri = S.config(cn, tr)
            fi = [CL.PNAMES.index(p) for p in free]
            F_ = S.Fitter3(M, N0 * fs, free, fsky=fs, priors=pri, truth=tr)
            Ft = F_.fisher(tr[fi]); J = F_.jac(tr[fi])
            sig, _, gS = S.derived_sigma3(Ft, fi, tr)
            V[cn] = (gS @ (np.linalg.solve(Ft, J.T) / N0.ravel()[None, :]) / sig["S8"]).reshape(N0.shape)
        for lab, a, b, sh in PERT:
            s = (zed[:-1] >= a - 1e-9) & (zed[1:] <= b + 1e-9)
            dN = np.zeros_like(N0); dN[s] = sh * N0[s]
            log(f"   [{setup:11s}] {sv:4s} {lab}: S8 bias shift " + "; ".join(f"{cn} {(V[cn] * dN).sum():+.3f}" for cn in V))
log(f"\n[a05] done {time.time()-t0:.0f}s")
