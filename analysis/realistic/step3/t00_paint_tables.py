"""Step 3 / t00: paint tables for the tiles, and checks of the painting.

The y map of a tile is painted at 0.25' with a 0.8' Gaussian painting beam, Fourier-cropped to the 0.5' grid and the
painting beam divided out (szsim.finder.paint_y), as in analysis/visual/v01_patches.py Part B.  A halo of size
theta500 is painted as the mixture (1-w) P_i + w P_{i+1} of the two neighbouring tabulated profiles (nodes linear in
ln theta, step 0.04, 0.24'-62'); the response tables of the decomposition use the same mixture, so painting and
decomposition are consistent by construction.
Checks:
 (1) every tabulated profile integrates to g(0) (int 2 pi r P dr = g(0); the cached Hankel table has
     g(0) = RATIO_5R500 (1 + 1.8e-5));
 (2) painted single halos (several sizes, sub-pixel positions) against their exact Fourier synthesis
     Y [(1-w) g(l th_i) + w g(l th_i+1)] B_paint(l) exp(-i k.x) / Omega on all in-band modes (|k| = 100-20000);
     and the node mixture against the exact template of the halo's own size;
 (3) time to paint a full stored patch.
Output: data_real/step3/paint_tables.npz
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/analysis/q2_maps")
from szsim import finder as F, signal as sg      # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    path = os.path.join(F.STEP3_DATA, "paint_tables.npz")
    th, rP = F.paint_table_nodes()
    print(f"theta nodes: {len(th)} from {th[0]:.3f}' to {th[-1]:.2f}' (d ln theta = {F.PT_DLN}); r nodes per profile {F.PT_NR}; "
          f"painting beam {F.FWHM_PAINT}' FWHM; table radius 5 theta + 7 sigma_paint")
    tabs = F.build_paint_tables(path, nproc=6)
    print(f"tables built in {time.time() - t0:.0f} s -> {path}")
    prev = os.environ.get("PREV_TABLES")
    if prev and os.path.exists(prev):
        old = np.load(prev)
        print(f"    rebuilt tables vs the copy used by the production runs ({os.path.basename(prev)}): "
              f"max |difference| = {np.abs(old['tab'] - tabs['tab']).max():.1e} (max value {np.abs(tabs['tab']).max():.2e})")

    # (1) normalisation
    from scipy.integrate import simpson
    err = []
    for i in range(len(th)):
        r = np.linspace(0, rP[i], F.PT_NR)
        err.append(simpson(2 * np.pi * r * tabs["tab"][i], x=r) / F.gt(np.array([0.0]))[0] - 1)
    err = np.array(err)
    print(f"(1) int 2 pi r P dr / g(0) - 1 (Simpson on the table nodes): max |.| = {np.abs(err).max():.2e} "
          f"(median {np.median(np.abs(err)):.1e}); g(0) / RATIO_5R500 - 1 = {F.gt(np.array([0.0]))[0] / sg.RATIO_5R500 - 1:.2e}")

    # (2) painted single halos vs exact Fourier synthesis
    G = F.grid()
    rng = np.random.default_rng(7)
    print("(2) single painted halo vs exact Fourier synthesis on the in-band modes "
          "(max |diff| / max |exact|; rms over modes / rms exact)")
    worst = 0.0
    for thh in [0.26, 0.5, 0.731, 1.0, 2.37, 5.0, 12.3, 30.0, 55.0]:
        y0, x0 = rng.uniform(0, F.LBOX, 2)
        Ys = F.paint_y([y0], [x0], [1.0], [thh], tabs, G).ravel()[G.m_idx]
        i0, w = F.theta_node(np.log(thh), float(tabs["lnth0"]), float(tabs["dln"]), len(th))
        gmix = (1 - w) * F.gt(G.m_L * th[i0] * F.AR) + w * F.gt(G.m_L * th[i0 + 1] * F.AR)
        gown = F.gt(G.m_L * thh * F.AR)
        Bp = np.exp(-0.5 * (G.m_L * F.SIG_PAINT * F.AR) ** 2)
        ph = np.exp(-1j * (G.m_k0 * y0 + G.m_k1 * x0) * F.AR)
        ex = gmix * Bp * ph / (F.PIX ** 2)
        d = np.abs(Ys - ex).max() / np.abs(ex).max()
        rms = np.sqrt(np.mean(np.abs(Ys - ex) ** 2) / np.mean(np.abs(ex) ** 2))
        dm = np.abs(gmix - gown).max() / gown.max()
        worst = max(worst, d)
        print(f"    theta500 = {thh:6.3f}' at ({y0:6.2f}, {x0:6.2f})': painted vs exact max {d:.2e}, rms {rms:.2e}; "
              f"node mixture vs own-size template: max {dm:.1e} of g(0)")
    print(f"    worst max |diff| / max|exact| = {worst:.2e}")

    # (3) full patch
    import q2lib as L
    d = L.load_patch(0)
    t1 = time.time()
    Ysig = F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], tabs, G)
    t_paint = time.time() - t1
    tot = Ysig[0, 0].real * F.PIX ** 2 / (d["Y500"].sum() * sg.RATIO_5R500)
    print(f"(3) patch 0: {d['x'].size} halos painted (incl. FFT and crop) in {t_paint:.1f} s on one core; "
          f"k = 0 mode / (sum Y500 RATIO) = {tot:.6f}")
    print(f"done [{time.time() - t0:.0f} s]")
