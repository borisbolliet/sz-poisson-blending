"""[repair after verification] Kernel-interpolation systematic, step A: rate histograms R(x|thA) with the
finer cross-response table (NT=79; its nodes nest the NT=40 production nodes, so thA = TH40[i] = TH79[2i]).
Same population discretisation as b03 (zsub=1, msub=6, ngh=16).  Output: data/blending/rate_nt79.npz"""
import os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl
from szsim import signal as sg

t0 = time.time()
m = sg.SZModel()
A = np.load(os.path.join(bl.DATA, "analytic.npz"))
TH40 = A["th"]
tab = bl.RhoTable(np.load(os.path.join(bl.DATA, "rho_table_nt79.npz")))
assert np.allclose(tab.th[::2], TH40), "NT=79 nodes must nest the NT=40 nodes"
pop = bl.neighbour_population(zsub=1, msub=6, ngh=16)
NT = len(TH40); nb_ = len(bl.XLOG_EDGES) - 1
W = np.zeros((NT, 3, nb_)); WX = np.zeros_like(W); WX2 = np.zeros_like(W); CAP = np.zeros((NT, 3)); CAPX = np.zeros((NT, 3))
sigA = m.sigma_Y(TH40)
for i in range(NT):
    W[i], WX[i], WX2[i], CAP[i], CAPX[i] = bl.rate_histogram(TH40[i], sigA[i], pop, tab)
    if i % 5 == 0:
        print(f"node {i} thA={TH40[i]:.3f}' mu NT79 {WX[i].sum() + CAPX[i].sum():.5f} vs NT40 {A['mu_exact'][i].sum():.5f}  ({time.time()-t0:.0f}s)", flush=True)
np.savez(os.path.join(bl.DATA, "rate_nt79.npz"), th=TH40, W=W, WX=WX, WX2=WX2, CAP=CAP, CAPX=CAPX, mu=WX.sum(2) + CAPX)
print(f"done {time.time()-t0:.1f}s")
