"""Step 1 / s06a: MMF cross-response tables rho(theta_A, theta_B, s) for SO baseline and goal (lmin = 100).

Format = analysis/blending rho table (blendlib.RhoTable): common log grid of theta (96 nodes,
0.1'-150'; a first build with 48 nodes had theta-interpolation errors up to 3.3% of rho(0) at large theta, see the
history at the top of the log), s = S_AB t^2 with t uniform (NU nodes), S_AB = 5 thA + 5 thB + R_BEAM and R_BEAM = 150' (the MMF
kernel is compensated and has wings well beyond the 5 theta500 truncation).  Symmetry: rho_AB / sigma_A^2
symmetric in (A, B).
Checks:
 (1) trilinear interpolation (blendlib.rho_at, as used by the rate kernel) vs exact
     MMFModel.cross_response at 60 random (thA, thB) x 8 random s;
 (2) DC response: table integral int 2 pi s rho ds (exact value 0 for the compensated MMF), in units of
     the white-noise-filter-like scale int 2 pi s |rho| ds;
 (3) int 2 pi s rho^2 ds from the table vs Parseval rho2_area (support truncation + table resolution).
Output: data_real/rho_mmf_{baseline,goal}.npz
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["NUMBA_NUM_THREADS"] = "4"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdlib1 as pl                      # noqa: E402
import blendlib as bl                    # noqa: E402

ROOT = pl.ROOT
TH = np.geomspace(0.1, 150.0, int(os.environ.get("RHO_NT", 96)))
NU = int(os.environ.get("RHO_NU", 401))
R_EXTRA = 150.0


def main(mode):
    t0 = time.time()
    spec = ("so", mode, 100.0)
    out = f"{ROOT}/data_real/rho_mmf_{mode}.npz"
    d = pl.build_rho_table(spec, TH, NU=NU, R_extra=R_EXTRA, nproc=4)
    np.savez(out, **d)
    print(f"[{mode}] table {len(TH)}x{len(TH)}x{NU} built in {time.time()-t0:.0f}s -> {out}", flush=True)
    tab = bl.RhoTable(np.load(out))
    m = pl.make_model(spec)
    rng = np.random.default_rng(7)
    e0, eI = [], []
    for k in range(60):
        a = float(np.exp(rng.uniform(np.log(0.11), np.log(120))))
        b = float(np.exp(rng.uniform(np.log(0.11), np.log(120))))
        S = 5 * a + 5 * b + R_EXTRA
        s = np.sort(rng.uniform(0, 1, 8) ** 2 * S)
        ex = m.cross_response(a, b, s)
        ip = np.array([bl.rho_interp_py(tab, a, b, x) for x in s])
        r0 = m.cross_response(a, b, [0.0])[0]
        e0.append(np.max(np.abs(ip - ex)) / abs(r0))
    e0 = np.array(e0)
    print(f"[{mode}] (1) max|rho_interp - rho_exact| / |rho(0)| over 60 x 8 random points: median {np.median(e0):.2e}, "
          f"95% {np.percentile(e0, 95):.2e}, max {e0.max():.2e}", flush=True)
    t = tab.t
    dc_rel, r2 = [], []
    for i in range(0, len(TH), 4):
        for j in range(0, len(TH), 4):
            S = 5 * TH[i] + 5 * TH[j] + R_EXTRA
            s = S * t ** 2
            I1 = np.trapezoid(2 * np.pi * s * tab.rho[i, j], s)
            Iabs = np.trapezoid(2 * np.pi * s * np.abs(tab.rho[i, j]), s)
            I2 = np.trapezoid(2 * np.pi * s * tab.rho[i, j] ** 2, s)
            dc_rel.append(I1 / Iabs)
            r2.append(I2 / m.rho2_area(TH[i], TH[j]) - 1)
    dc_rel = np.array(dc_rel); r2 = np.array(r2)
    print(f"[{mode}] (2) DC: int rho / int |rho| over 144 pairs: median {np.median(np.abs(dc_rel)):.2e}, max {np.abs(dc_rel).max():.2e} "
          f"(exact 0)", flush=True)
    print(f"[{mode}] (3) int rho^2 (table) / Parseval - 1 over 144 pairs: median {np.median(r2):+.2e}, "
          f"min {r2.min():+.2e}, max {r2.max():+.2e}", flush=True)
    print(f"[{mode}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    for mode in sys.argv[1:] or ["baseline", "goal"]:
        main(mode)
