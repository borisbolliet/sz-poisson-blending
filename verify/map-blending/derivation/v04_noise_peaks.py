"""C5: single-scale peak counts of matched-filtered white noise vs our own Kac-Rice numbers (v01_analytic.log),
with nearest-pixel heights and with 3x3 least-squares quadratic sub-pixel refinement."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["NUMBA_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import fft as sfft
import v03_mappipe as P

psis, sig = P.filters()
KR = {0: {3.0: 808.928, 3.5: 184.293, 4.0: 32.201, 4.5: 4.322}, 6: {3.0: 272.239, 3.5: 61.249, 4.0: 10.627, 4.5: 1.421}}
# design matrix for 3x3 quadratic fit
dx, dy = np.meshgrid([-1, 0, 1], [-1, 0, 1]); dx = dx.ravel(); dy = dy.ravel()
A = np.stack([np.ones(9), dx, dy, dx * dx, dy * dy, dx * dy], 1); Ap = np.linalg.pinv(A)
npatch = int(sys.argv[1]) if len(sys.argv) > 1 else 60
res = {k: {"raw": [], "ref": []} for k in KR}
for p in range(npatch):
    rng = np.random.default_rng(777000 + p)
    F = sfft.rfft2((rng.standard_normal((P.N, P.N)) * (np.sqrt(P.NW) / P.PIX)).astype(np.float32))
    for k in KR:
        m = sfft.irfft2(F * psis[k], s=(P.N, P.N)) / sig[k]
        iy, ix = P.local_max(m.astype(np.float32), np.float32(2.7))
        raw = m[iy, ix]
        V = np.stack([m[(iy + a) % P.N, (ix + b) % P.N] for a, b in zip(dy, dx)], 1)   # (n,9)
        c = V @ Ap.T
        H = np.stack([[2 * c[:, 3], c[:, 5]], [c[:, 5], 2 * c[:, 4]]], -1).transpose(1, 0, 2)  # (n,2,2)
        g = np.stack([c[:, 1], c[:, 2]], 1)
        det = H[:, 0, 0] * H[:, 1, 1] - H[:, 0, 1] ** 2
        ok = (det > 0) & (H[:, 0, 0] < 0)
        off = np.zeros_like(g)
        off[ok] = -np.linalg.solve(H[ok], g[ok][:, :, None])[:, :, 0]
        ok &= (np.abs(off) <= 1).all(1)
        ref = raw.copy()
        ref[ok] = c[ok, 0] + 0.5 * (g[ok] * off[ok]).sum(1)
        ref = np.maximum(ref, raw)
        res[k]["raw"].append([(raw > nu).sum() for nu in KR[k]]); res[k]["ref"].append([(ref > nu).sum() for nu in KR[k]])
for k in KR:
    for mode in ("raw", "ref"):
        a = np.array(res[k][mode], float)
        mu = a.mean(0); er = a.std(0, ddof=1) / np.sqrt(len(a))
        print(f"scale {P.TK[k]:.3f}' {mode}: " + "  ".join(
            f"nu>{nu}: {m_:.2f}+-{e_:.2f} vs KR {kr:.2f} ({(m_-kr)/e_:+.2f}sig)" for nu, m_, e_, kr in zip(KR[k], mu, er, KR[k].values())))
