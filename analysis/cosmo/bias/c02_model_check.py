"""Check the naive count model at the fiducial point against szsim (expected detections) and time it."""
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np, cosmolib as CL
from szsim import signal as sg
m = sg.SZModel()
for qc in (5.0, 6.0):
    M = CL.NaiveModel(qcut=qc)
    t = time.time(); N = M.counts(CL.FID); dt = time.time() - t
    t = time.time()
    for _ in range(20): M.counts(CL.FID)
    dt2 = (time.time() - t) / 20
    ref = m.expected_detections(qc)
    print(f"q_cut={qc}: naive model total {N.sum():,.1f} vs szsim expected_detections {ref:,.1f} (rel {N.sum()/ref-1:+.2e});"
          f" bins {N.shape}, bins with mu>1: {(N>1).sum()}, mu>10: {(N>10).sum()}; eval time {dt2*1e3:.1f} ms (first {dt*1e3:.0f} ms)")
    print("   counts per z bin:", np.round(N.sum(1), 1))
    print("   counts per q bin:", np.round(N.sum(0), 1), " q edges", np.round(M.qed, 2))
    # q0 check vs szsim
    lnN, th, lq = M.nodes(CL.FID)
    iz = 49; zz = M.Z[iz]
    r = np.exp(lq[iz]) / m.q_mean(M.M, zz) - 1; rt = th[iz] / m.theta500(M.M, zz) - 1
    print(f"   z={zz:.2f}: q0 vs SZModel.q_mean max|rel| {np.abs(r).max():.1e}; theta500 {np.abs(rt).max():.1e}")
