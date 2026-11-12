"""Arbiter / a02b: compare the kernel measured on painted maps at random points (a02_map_kernel_sim.py) with the build's
P(D) kernel, and propagate the difference to the data vector and the S8 bias.

For each fresh patch p, mode (baseline, goal) and node j = 7..25 (theta 0.29-4.69'), fdom = 1:
   K_map,p(theta_j, q; e) = (1/NPTS) sum_points 1[x_max < q] Phi(q + delta - e)       (noise marginalised analytically)
on q = 0.5-12 (step 0.25) and 13-30 (step 1), all 19 bin edges e.  Points with x_max < 0.5 enter through their delta
histogram (bin 0.005; Phi convolved by FFT, then linear interpolation in q - e: error < 1e-6), the others one by one.
dK_p = K_map,p - K_build (the stored data_real/step2/kernel_<mode>_f1.npz at the same q nodes).
Propagation: dK_p (linear in q between the q nodes, 0 outside q = 0.5-30 and at the other theta nodes) goes through the
build's own count machinery (s2lib.bin_tables + SmoothModelS.counts_table, linear in K) -> dN_p(z, q_obs) -> linearised
S8 bias shift with the build's response F^-1 J^T / N0 (configs a-e, SO40 and full sky, q_cut 5 and 6).  Estimate = mean
over patches; error = scatter / sqrt(n_patches) (patches are independent skies).
Also: the delta variance at random points vs the build's rate-histogram variance (second moment of the P(D)).
Output: a02b_map_kernel_compare.log, data_real/step2/a02_mapkernel_summary.npz
"""
import os, sys, glob, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np
from scipy.special import ndtr
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                   # noqa: E402
from s2lib import CL                                 # noqa: E402

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "a02b_map_kernel_compare.log"), "w")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


NODES = list(range(7, 26))
MODES = ("baseline", "goal")
DLO, DD, NBIN = -4.0, 0.005, 4000
DC = DLO + (np.arange(NBIN) + 0.5) * DD
QC = np.round(np.concatenate([np.arange(0.5, 12.0001, 0.25), np.arange(13.0, 30.0001, 1.0)]), 6)
E = S.EDGES
TT = QC[:, None] - E[None, :]                                          # q - e  (nq, ne)
TG = np.arange(-25.0, 30.0 + DD / 2, DD)                               # t grid for the bulk convolution
UG = TG[0] + DC[0] + DD * np.arange(TG.size + NBIN - 1)                # Phi samples: TG[m] + DC[i] = UG[m + i]
PHI_U = ndtr(UG)
files = sorted(glob.glob(f"{S.D2}/a02_mapkernel/patch_*.npz"))
NP = len(files)
KB = {m: np.load(f"{S.D2}/kernel_{m}_f1.npz") for m in MODES}
QA = KB["goal"]["QA"]
IQ = np.array([int(np.argmin(np.abs(QA - q))) for q in QC])
assert np.allclose(QA[IQ], QC)
TH = KB["goal"]["th"]
inp = np.load(f"{S.D2}/inputs.npz")
log(f"[a02b] {NP} fresh patches ({NP * 100:.0f} deg^2); nodes {NODES[0]}..{NODES[-1]} ({TH[NODES[0]]:.3f}-{TH[NODES[-1]]:.2f}'); "
    f"q nodes {len(QC)} ({QC[0]}-{QC[-1]}); fdom 1")


def kmap(h, td, tx, npts):
    F = fftconvolve(PHI_U, h[::-1].astype(float), mode="valid")       # F[m] = sum_i h_i Phi(TG[m] + DC_i)
    Kb = np.interp(TT, TG, F)
    s = tx < QC[-1]
    td, tx = td[s].astype(float), tx[s].astype(float)
    Kt = np.zeros_like(TT)
    for a in range(0, td.size, 2000):
        d_, x_ = td[a:a + 2000], tx[a:a + 2000]
        Kt += np.einsum("pqe,pq->qe", ndtr(TT[None] + d_[:, None, None]), (x_[:, None] < QC[None, :]).astype(float))
    return (Kb + Kt) / npts


dK = {m: np.zeros((NP, len(NODES), len(QC), len(E))) for m in MODES}
VAR = {m: np.zeros((NP, len(NODES))) for m in MODES}
MEAN = {m: np.zeros((NP, len(NODES))) for m in MODES}
PX = {m: np.zeros((NP, len(NODES))) for m in MODES}
for ip, fn in enumerate(files):
    d = np.load(fn)
    n = int(d["npts"])
    for m in MODES:
        Kbld = KB[m]["Kb"]
        mom = d[f"{m}_mom"]
        MEAN[m][ip] = mom[:, 0] / n; VAR[m][ip] = mom[:, 1] / n - (mom[:, 0] / n) ** 2; PX[m][ip] = mom[:, 3] / n
        for k, jn in enumerate(NODES):
            dK[m][ip, k] = kmap(d[f"{m}_{jn}_h"], d[f"{m}_{jn}_td"], d[f"{m}_{jn}_tx"], n) - Kbld[jn][IQ]
log(f"   kernels measured [{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ (1) node-level comparison
log("\n=== (1) K_map - K_build at selected nodes (mean over patches +- error of the mean), edge e = 5 ===")
log("    also: delta variance at random points / build rate-histogram variance - 1, and P(x_max >= 0.5)")
iqs = [int(np.argmin(np.abs(QC - q))) for q in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)]
for m in MODES:
    vb = inp[f"{m}_var"]
    for k, jn in enumerate(NODES):
        a = dK[m][:, k, :, 1]
        mu = a.mean(0); er = a.std(0, ddof=1) / np.sqrt(NP)
        vr = VAR[m][:, k].mean() / vb[jn] - 1; vre = VAR[m][:, k].std(ddof=1) / np.sqrt(NP) / vb[jn]
        Kref = KB[m]["Kb"][jn][IQ][:, 1]
        log(f"  [{m:8s}] theta {TH[jn]:6.3f}': " + " ".join(f"q={QC[i]:.0f}: {1e4*mu[i]:+6.1f}+-{1e4*er[i]:4.1f}" for i in iqs)
            + f"  (1e-4; K_build(q=5) = {Kref[iqs[3]]:.4f}) | var ratio - 1 = {vr:+.4f} +- {vre:.4f} | P(x_max>=0.5) "
            f"{PX[m][:, k].mean():.4f}")
log("    chi2 of dK over the q nodes 2-8 and edges 5, 6 (diagonal, per node):")
for m in MODES:
    out = []
    for k, jn in enumerate(NODES):
        sel = (QC >= 2) & (QC <= 8)
        a = dK[m][:, k][:, sel][:, :, 1:3].reshape(NP, -1)
        z = a.mean(0) / (a.std(0, ddof=1) / np.sqrt(NP))
        out.append(f"{TH[jn]:.2f}':{np.mean(z):+.2f}")
    log(f"  [{m}] mean pull of dK (q = 2-8, e = 5, 6) per node: " + " ".join(out))

# ------------------------------------------------------------------ (2) propagation to the data vector and the biases
G = CL.HMFGrid()
lth = np.log(TH)


def dN_from_dK(M, dk_nodes, truth):
    """counts shift (16, 9) from a kernel shift at NODES (on QC), linear in q between QC nodes."""
    full = np.zeros((len(NODES), len(QA), len(E)))
    sel = (QA >= QC[0]) & (QA <= QC[-1])
    for k in range(len(NODES)):
        for j in range(len(E)):
            full[k, sel, j] = np.interp(QA[sel], QC, dk_nodes[k, :, j])
    Pn = S.bin_tables(M, full, QA, E, truth[4])
    P = np.zeros((len(TH),) + Pn.shape[1:])
    P[NODES] = Pn
    return M.counts_table(truth, P, lth)


def responses(M, truth, fs):
    N0 = M.counts(truth)
    out = {}
    for cn in ("a", "b", "c", "d", "e"):
        free, pri = S.config(cn, truth)
        fi = [CL.PNAMES.index(p) for p in free]
        F_ = S.Fitter3(M, N0 * fs, free, fsky=fs, priors=pri, truth=truth)
        Ft = F_.fisher(truth[fi]); J = F_.jac(truth[fi])
        sig, _, gS = S.derived_sigma3(Ft, fi, truth)
        out[cn] = (gS @ (np.linalg.solve(Ft, J.T) / N0.ravel()[None, :]) / sig["S8"]).reshape(N0.shape)
    return out, N0


truth = S.FID.copy()
ZED = CL.default_bins(5.0)[0]
ZG = [(0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
summ = {}
log("\n=== (2) the kernel shift propagated: dN = N_b(map-calibrated kernel) - N_b(build), full sky, q > 5 ===")
for m in MODES:
    for qc in (5.0, 6.0):
        M = S.SmoothModelS(qc, S.LnSigma(S.sz_model(m)), grid=G)
        dN = np.stack([dN_from_dK(M, dK[m][ip], truth) for ip in range(NP)])          # (NP, 16, 9)
        Nn = np.load(f"{S.D2}/truth_{m}_f1.npz")[f"Nn_q{qc:.0f}"]
        Dref = np.load(f"{S.D2}/truth_{m}_f1.npz")[f"D_q{qc:.0f}"]
        mu = dN.mean(0); er = dN.sum((1, 2)).std(ddof=1) / np.sqrt(NP)
        if qc == 5.0:
            zs = []
            for a, b in ZG:
                s = (ZED[:-1] >= a - 1e-9) & (ZED[1:] <= b + 1e-9)
                v = dN[:, s].sum((1, 2)) / Nn[s].sum()
                zs.append(f"z {a}-{b}: {100*v.mean():+.3f}+-{100*v.std(ddof=1)/np.sqrt(NP):.3f}% "
                          f"(build Delta/N {100*Dref[s].sum()/Nn[s].sum():+.2f}%)")
            log(f"  [{m}] total dN {mu.sum():+.1f} +- {er:.1f} counts = {100*mu.sum()/Nn.sum():+.4f} +- {100*er/Nn.sum():.4f}% of N "
                f"(build Delta {100*Dref.sum()/Nn.sum():+.3f}%)")
            log("      " + "; ".join(zs))
        for sv, fs in (("SO40", 0.4), ("full", 1.0)):
            V, N0 = responses(M, truth, fs)
            for cn in ("a", "b", "c", "d", "e"):
                b = (dN * V[cn][None]).sum((1, 2))
                summ[(m, qc, sv, cn)] = (b.mean(), b.std(ddof=1) / np.sqrt(NP))
    log(f"   [{time.time()-t0:.0f}s]")

log("\n=== (3) S8 bias shift (map-calibrated kernel - build kernel), linearised, in Fisher sigma (mean +- error of the mean) ===")
log("    compare: maps verifier, map - build kernel on its own primaries (w08b): goal c SO40 +0.249 +- 0.111; full +0.393 +- 0.177")
for m in MODES:
    for qc in (5.0, 6.0):
        for sv in ("SO40", "full"):
            log(f"  [{m:8s}] q>{qc:.0f} {sv:4s}: " + "; ".join(
                f"{cn} {summ[(m, qc, sv, cn)][0]:+.3f}+-{summ[(m, qc, sv, cn)][1]:.3f}" for cn in ("a", "b", "c", "d", "e")))
np.savez(f"{S.D2}/a02_mapkernel_summary.npz", QC=QC, NODES=np.array(NODES), dK_goal=dK["goal"], dK_baseline=dK["baseline"],
         keys=np.array([f"{k[0]}|{k[1]}|{k[2]}|{k[3]}" for k in summ]), vals=np.array([summ[k] for k in summ]))
log(f"\n[a02b] done {time.time()-t0:.0f}s")
