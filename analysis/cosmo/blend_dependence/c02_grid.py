"""Run the fast blending pipeline on a set of parameter points (3 worker processes, 1 thread each).
usage: c02_grid.py grid|extra
  grid : full 3^4 grid  sigma8 {0.75,0.81,0.87} x Om {0.27,0.31,0.35} x (1-b) {0.7,0.8,0.9} x sigma_lnY {0.1,0.17,0.3}
  extra: fiducial finite-difference steps (for Fisher), Planck18 +-1/3 sigma points, random interior test points
Output: data_cosmo/c02_<name>.npz"""
import os, sys, time
os.environ["NUMBA_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import numpy as np
from multiprocessing import Pool

S8 = [0.75, 0.81, 0.87]; OM = [0.27, 0.31, 0.35]; OMB = [0.7, 0.8, 0.9]; SL = [0.1, 0.17, 0.3]
FIDP = (0.81, 0.31, 0.8, 0.17)
STEP = (0.01, 0.01, 0.02, 0.03)          # finite-difference half-steps
SIG_P18 = (0.0060, 0.0073)               # Planck18 TT,TE,EE+lowE+lensing 1-sigma on sigma8, Om


def points(name):
    if name == "grid":
        return [tuple(p) for p in itertools.product(S8, OM, OMB, SL)]
    pts = []
    for a in range(4):
        for sgn in (-1, 1):
            p = list(FIDP); p[a] += sgn * STEP[a]; pts.append(tuple(p))
    for a in range(2):
        for k in (-3, -1, 1, 3):
            p = list(FIDP); p[a] += k * SIG_P18[a]; pts.append(tuple(p))
    for k in (-3, 3):                              # joint, along the Planck sigma8-Om degeneracy sign (+,-)
        p = list(FIDP); p[0] += k * SIG_P18[0]; p[1] -= k * SIG_P18[1]; pts.append(tuple(p))
    rng = np.random.default_rng(7)
    lo = np.array([0.75, 0.27, 0.7, 0.1]); hi = np.array([0.87, 0.35, 0.9, 0.3])
    for _ in range(10):
        pts.append(tuple(np.round(lo + (hi - lo) * rng.random(4), 4)))
    return pts


def work(p):
    import cosmolib as cl
    t0 = time.time()
    r = cl.run_point(*p)
    keep = {k: r[k] for k in ["N_single", "N_det_b", "ratio", "B2_10", "B2_20", "B3", "lost", "ybias", "mu_c",
                              "var500_c", "var5_c", "lam1", "ybar_c", "ytot", "cnt_c", "mu_det", "var5_det",
                              "var500_det", "Nzq_b", "Nzq_n", "wdet", "params"]}
    keep["Hs"] = r["H"].sum(0)
    keep["Pdet"] = r["Pdet"]
    print(f"{p}: ratio {r['ratio']:.5f} B2 {r['B2_10']:.4f} B3 {r['B3']:.4f} lost {r['lost']:.4f} "
          f"N_single {r['N_single']:,.0f} ({time.time()-t0:.0f}s)", flush=True)
    return keep


def main():
    name = sys.argv[1]
    pts = points(name)
    print(f"{len(pts)} points", flush=True)
    t0 = time.time()
    with Pool(3) as pool:
        res = pool.map(work, pts, chunksize=1)
    out = {k: np.array([r[k] for r in res]) for k in res[0]}
    np.savez(f"/Users/boris/GitHub/sz-poisson-blending/data_cosmo/c02_{name}.npz", **out)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
