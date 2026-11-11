"""Check the cosmology-dependent population of cclib against szsim (fiducial): full-sky counts and
single-cluster expected detections; timing of one cosmology evaluation."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
import time
import numpy as np
import cclib as cc
from szsim import cosmo_hmf as ch, signal as sg

t0 = time.time()
hmf = cc.hmf_grid(cc.FID)
t1 = time.time()
pop = cc.population(cc.FID, hmf=hmf, qmin=0.0)
t2 = time.time()
print(f"hmf_grid {t1-t0:.2f}s, population {t2-t1:.2f}s, points {pop['q'].size:,}")
d2N = hmf[0]
T = ch.default_table()
for Mmin in (1e13, 1e14, 3e14):
    s = cc.GRID.M >= Mmin
    # partial cells: grid edges are at 1e13 * 10^(3k/180); compare with the cell-aligned table integral
    Mlo = np.exp(cc.GRID.lnME[np.searchsorted(cc.GRID.lnME, np.log(Mmin))])
    n = (d2N[:, cc.GRID.M >= Mlo] * cc.GRID.dz * cc.GRID.dlnM).sum()
    print(f"N(M>{Mlo:.3e}, 0.01<z<3) grid {n:,.1f}  szsim table {T.N_total(Mlo):,.1f}  rel {n/T.N_total(Mlo)-1:+.2e}")
m = sg.SZModel()
for qc in (4.0, 5.0, 6.0, 7.0):
    Ng = cc.counts_gauss(pop, [qc, np.inf], 1)[0, 0]
    Ns = m.expected_detections(qc)
    print(f"single-cluster N_det(q_obs>{qc}) cclib {Ng:,.1f}  szsim {Ns:,.1f}  rel {Ng/Ns-1:+.2e}")
pop01 = cc.population(cc.FID, hmf=hmf, qmin=0.1)
for qc in (4.0, 5.0):
    a = cc.counts_gauss(pop, [qc, np.inf], 1)[0, 0]; b = cc.counts_gauss(pop01, [qc, np.inf], 1)[0, 0]
    print(f"share of q_obs>{qc} single-cluster detections from primaries with q<0.1: {1-b/a:.2e}")
t3 = time.time(); hmf2 = cc.hmf_grid(dict(cc.FID, sigma8=0.83)); print(f"new-cosmology hmf_grid {time.time()-t3:.2f}s")
