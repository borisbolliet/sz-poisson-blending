"""C7 map side: variance of the matched-filtered map painted with only q_true<2 (and q_true<5) halos, own patches,
vs the analytic Poisson one-halo values from v01 (0.0532/0.0914/0.1696 and 0.1194/0.2192/0.4664)."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import fft as sfft
import v03_mappipe as P
import own_model as om

psis, sig = P.filters()
KS = [0, 5, 11]
LNT0, DLT = np.log(0.03), 0.08
res = {2: [], 5: []}
npch = int(sys.argv[1])
for pid in range(npch):
    h = P.draw(5000 + pid)
    for qthr in (2, 5):
        s = h["q"] < qthr
        t = (np.log(h["th"][s]) - LNT0) / DLT; i0 = np.floor(t).astype(int); w = t - i0
        bins = np.concatenate([i0, i0 + 1]); ww = np.concatenate([h["Y"][s] * (1 - w), h["Y"][s] * w])
        xx = np.concatenate([h["x"][s]] * 2); yy = np.concatenate([h["y"][s]] * 2)
        G = np.zeros((P.N, P.N // 2 + 1), np.complex128); m = np.zeros((P.N, P.N))
        for b in np.unique(bins):
            q = bins == b; m[:] = 0
            P.tsc_deposit(m, xx[q], yy[q], ww[q], P.PIX)
            G += sfft.rfft2(m) * om.gfun(P.LL * np.exp(LNT0 + DLT * b))
        D = (G / P.WTSC / P.PIX ** 2 * P.BEAM).astype(np.complex64)
        res[qthr].append([sfft.irfft2(D * psis[k], s=(P.N, P.N)).var(dtype=np.float64) / sig[k] ** 2 for k in KS])
an = {2: [0.0532, 0.0914, 0.1696], 5: [0.1194, 0.2192, 0.4664]}
for qthr in (2, 5):
    a = np.array(res[qthr]); mu = a.mean(0); er = a.std(0, ddof=1) / np.sqrt(len(a))
    print(f"q_true<{qthr} ({npch} patches): " + "  ".join(f"{P.TK[k]:.2f}': {m_:.4f}+-{e_:.4f} vs analytic {x:.4f} ({(m_-x)/e_:+.2f}sig)"
                                              for k, m_, e_, x in zip(KS, mu, er, an[qthr])))
