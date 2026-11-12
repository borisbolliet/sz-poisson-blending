"""v07: extra checks.
(1) Claim A1 with our own MMF algebra: sigma(theta) of the build's template with the analytic covariance on the grid
    (vsz_lib.SZFilter, independent code) against step 1's MMFModel.sigma_Y_exact (Hankel integral).
(2) Merging: our own implementation of SZiFi's 'closest' rule as read in cat.merge_detections (anchor = first remaining
    entry in catalogue order, keep the highest q with distance < radius, drop that group; flat, NON-periodic
    distances) against the build's szsim.finder.merge_closest (periodic distances) on the build's stored final q > 5
    catalogues of 10 tiles per noise level.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "2"
sys.path.insert(0, HERE)
import numpy as np
import vsz_lib as V
from szsim import instrument as ins

F = V.F
g = V.szgrid()
G = F.grid()
print("(1) sigma(theta), the build's template + analytic covariance, our MMF vs MMFModel.sigma_Y_exact (l = 100-20000)")
for mode in V.MODES:
    ex = ins.MMFModel(ins.Instrument.so_lat(mode, lmin=100.0, lmax=20000.0)).sigma_Y_exact(V.THETAS)
    filt = V.SZFilter(analytic_mode=mode, g=g)
    s = np.array([V.sigma_from(V.template_modes(t, "build", g), filt) for t in V.THETAS]) / V.AR ** 2
    print(f"  {mode}: max |ours / exact - 1| = {np.abs(s / ex - 1).max():.1e} (theta = {V.THETAS[np.abs(s / ex - 1).argmax()]:.2f}')")
    del filt


def szifi_closest(y, x, q, radius):
    rem = list(range(len(q)))
    keep = []
    while rem:
        a = rem[0]
        d = np.hypot(y[rem] - y[a], x[rem] - x[a])
        grp = [rem[i] for i in np.flatnonzero(d < radius)]
        keep.append(grp[int(np.argmax(q[grp]))])
        gs = set(grp)
        rem = [i for i in rem if i not in gs]
    return np.array(keep, dtype=np.int64)


print("(2) merging of the build's final q > 5 catalogues: SZiFi 'closest' (non-periodic) vs the build's (periodic)")
TILES = [0, 33, 66, 99, 132, 165, 198, 231, 264, 297]
for mode in V.MODES:
    for rad in (1.0, 2.0, 10.0):
        n_in = n_b = n_s = n_same = 0
        n_diff_interior = 0
        for p in TILES:
            st = V.load_stored_tile(p)
            q = st[f"{mode}_it1_q"]
            det = dict(q=q, y=st[f"{mode}_it1_iy"] * V.PIXA, x=st[f"{mode}_it1_ix"] * V.PIXA)
            kb = F.apply_merge(det, {1.0: "closest1", 2.0: "closest2", 10.0: "closest10"}[rad], 5.0)
            idx = np.flatnonzero(q > 5.0)
            ks = idx[szifi_closest(det["y"][idx], det["x"][idx], q[idx], rad)]
            n_in += idx.size; n_b += kb.size; n_s += ks.size
            sb, ss = set(kb.tolist()), set(ks.tolist())
            n_same += len(sb & ss)
            far = lambda k: min(det["y"][k], det["x"][k], 600 - det["y"][k], 600 - det["x"][k]) > 2 * rad
            n_diff_interior += sum(1 for k in (sb ^ ss) if far(k))
        print(f"  {mode:8s} radius {rad:4.1f}': q > 5 entries {n_in}; kept: build {n_b}, SZiFi rule {n_s}; in both {n_same}; "
              f"differing entries farther than 2 radii from an edge: {n_diff_interior}")
print("done")
