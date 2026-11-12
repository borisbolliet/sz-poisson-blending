"""Arbiter / a02c: the same random-point map test as a02b for the LOW-PRESSURE population ((1-b) = 0.6: every Y x c,
c = 0.75^1.79 = 0.5975), SO baseline, fdom 1.  The map check found its largest total map - kernel offset here
(+0.14 +- 0.07% of N, 2.0 sigma; w08).  The maps are linear in Y, so the low-pressure neighbour sum at a random point is
c x delta and the brightest neighbour c x x_max (exactly what the map check did with its primaries):
   K_low,map(theta_j, q; e) = (1/NPTS) sum_points 1[c x_max < q] Phi(q + c delta - e),
compared with the build's data_real/step2/kernel_lowP_f1.npz, and propagated through the build's count model with the
low-pressure truth ((1-b) = 0.6).  Points with x_max < 0.5 enter through their delta histogram (bin centres scaled by c).
Output: a02c_map_kernel_lowP.log
"""
import os, sys, glob, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2lib as S                                   # noqa: E402
from s2lib import CL                                 # noqa: E402

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "a02c_map_kernel_lowP.log"), "w")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


NODES = list(range(7, 26))
DLO, DD, NBIN = -4.0, 0.005, 4000
DC = DLO + (np.arange(NBIN) + 0.5) * DD
QC = np.round(np.concatenate([np.arange(0.5, 12.0001, 0.25), np.arange(13.0, 30.0001, 1.0)]), 6)
E = S.EDGES
TT = QC[:, None] - E[None, :]
c = float(S.C_LOW)
files = sorted(glob.glob(f"{S.D2}/a02_mapkernel/patch_*.npz"))
NP = len(files)
KB = np.load(f"{S.D2}/kernel_lowP_f1.npz")
QA, TH = KB["QA"], KB["th"]
IQ = np.array([int(np.argmin(np.abs(QA - q))) for q in QC])
assert np.allclose(QA[IQ], QC)
log(f"[a02c] {NP} patches; low pressure c = {c:.5f}; baseline MMF; nodes {TH[NODES[0]]:.3f}-{TH[NODES[-1]]:.2f}'; fdom 1")


TGC = np.arange(-25.0, 30.0 + c * DD / 2, c * DD)                     # t grid with the scaled bin spacing c*DD
UGC = TGC[0] + c * DC[0] + c * DD * np.arange(TGC.size + NBIN - 1)
PHI_UC = ndtr(UGC)


def kmap(h, td, tx, npts):
    from scipy.signal import fftconvolve
    F = fftconvolve(PHI_UC, h[::-1].astype(float), mode="valid")      # F[m] = sum_i h_i Phi(TGC[m] + c DC_i)
    Kb = np.interp(TT, TGC, F)
    s = c * tx < QC[-1]
    td, tx = c * td[s].astype(float), c * tx[s].astype(float)
    Kt = np.zeros_like(TT)
    for a in range(0, td.size, 2000):
        d_, x_ = td[a:a + 2000], tx[a:a + 2000]
        Kt += np.einsum("pqe,pq->qe", ndtr(TT[None] + d_[:, None, None]), (x_[:, None] < QC[None, :]).astype(float))
    return (Kb + Kt) / npts


dK = np.zeros((NP, len(NODES), len(QC), len(E)))
for ip, fn in enumerate(files):
    d = np.load(fn)
    n = int(d["npts"])
    for k, jn in enumerate(NODES):
        dK[ip, k] = kmap(d[f"baseline_{jn}_h"], d[f"baseline_{jn}_td"], d[f"baseline_{jn}_tx"], n) - KB["Kb"][jn][IQ]
log(f"   kernels measured [{time.time()-t0:.0f}s]")
iqs = [int(np.argmin(np.abs(QC - q))) for q in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)]
log("\n=== (1) K_low,map - K_build (1e-4), e = 5, mean +- error of the mean over patches ===")
for k, jn in enumerate(NODES):
    a = dK[:, k, :, 1]; mu = a.mean(0); er = a.std(0, ddof=1) / np.sqrt(NP)
    log(f"   theta {TH[jn]:6.3f}': " + " ".join(f"q={QC[i]:.0f}: {1e4*mu[i]:+6.1f}+-{1e4*er[i]:4.1f}" for i in iqs))

G = CL.HMFGrid()
lth = np.log(TH)
truth = S.FID_LOW.copy()
M = S.SmoothModelS(5.0, S.LnSigma(S.sz_model("baseline")), grid=G)


def dN_from_dK(dk_nodes):
    full = np.zeros((len(NODES), len(QA), len(E)))
    sel = (QA >= QC[0]) & (QA <= QC[-1])
    for k in range(len(NODES)):
        for j in range(len(E)):
            full[k, sel, j] = np.interp(QA[sel], QC, dk_nodes[k, :, j])
    Pn = S.bin_tables(M, full, QA, E, truth[4])
    P = np.zeros((len(TH),) + Pn.shape[1:]); P[NODES] = Pn
    return M.counts_table(truth, P, lth)


dN = np.stack([dN_from_dK(dK[ip]) for ip in range(NP)])
t = np.load(f"{S.D2}/truth_lowP_f1.npz")
Nn = t["Nn_q5"]
tot = dN.sum((1, 2))
log(f"\n=== (2) propagated: dN = {tot.mean():+.2f} +- {tot.std(ddof=1)/np.sqrt(NP):.2f} counts = {100*tot.mean()/Nn.sum():+.4f} +- "
    f"{100*tot.std(ddof=1)/np.sqrt(NP)/Nn.sum():.4f}% of N (build Delta {100*t['D_q5'].sum()/Nn.sum():+.3f}%; map check: map - kernel on its "
    f"primaries +0.138 +- 0.069%)")
N0 = M.counts(truth)
for sv, fs in (("SO40", 0.4), ("full", 1.0)):
    out = []
    for cn in ("a", "b", "c", "d", "e"):
        free, pri = S.config(cn, truth)
        fi = [CL.PNAMES.index(p) for p in free]
        F_ = S.Fitter3(M, N0 * fs, free, fsky=fs, priors=pri, truth=truth)
        Ft = F_.fisher(truth[fi]); J = F_.jac(truth[fi])
        sig, _, gS = S.derived_sigma3(Ft, fi, truth)
        V = (gS @ (np.linalg.solve(Ft, J.T) / N0.ravel()[None, :]) / sig["S8"]).reshape(N0.shape)
        b = (dN * V[None]).sum((1, 2))
        out.append(f"{cn} {b.mean():+.3f}+-{b.std(ddof=1)/np.sqrt(NP):.3f}")
    log(f"   S8 bias shift, {sv}: " + "; ".join(out))
log(f"\n[a02c] done {time.time()-t0:.0f}s")
