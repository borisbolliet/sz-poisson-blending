"""v09: the builder's stored MMF cross-response tables (data_real/rho_mmf_{baseline,goal}.npz, read as data) against
MY exact rho (1D radial quadrature on my N_y / CLASS CMB / 2D-FFT template, validated against my 2D grid in v07).
(1) node values: rho[i, j, k] at s = S_ij t_k^2, S_ij = 5 th_i + 5 th_j + R_BEAM  (no interpolation)
(2) the builder's lookup scheme (trilinear in ln thA, ln thB and t = sqrt(s/S) (NU-1), S from the actual thetas;
    re-implemented here from its documented definition) at random (thA, thB, s) vs my exact rho.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vlib as V
V.set_threads(4)
import numpy as np
from scipy.special import j0

t0 = time.time()
AM = V.AM
rng = np.random.default_rng(11)
for mode in ("baseline", "goal"):
    d = np.load(os.path.join(V.ROOT, "data_real", f"rho_mmf_{mode}.npz"))
    th, t, rho, sig, RB = d["th"], d["t"], d["rho"], d["sigY"], float(d["R_BEAM"])
    NT, NU = len(th), len(t)
    ell = np.arange(100.0, 20000.5, 1.0); inv = V.inv_Ny(ell, mode)
    w = V.trap_w(ell) * ell / (2 * np.pi) * inv

    def exact(tA, tB, s_am):
        gA = V.g_u(ell * tA * AM); gB = V.g_u(ell * tB * AM)
        return (1.0 / np.sum(w * gA * gA)) * (j0(np.outer(np.atleast_1d(s_am) * AM, ell)) @ (w * gA * gB))

    mysig = np.array([1 / np.sqrt(np.sum(w * V.g_u(ell * x * AM) ** 2)) / AM ** 2 for x in th])
    print(f"[{mode}] table: {NT} theta {th[0]:.2f}-{th[-1]:.1f}', NU = {NU}, R_BEAM = {RB}; stored sigY vs mine: max |rel| "
          f"{np.max(np.abs(sig/mysig-1)):.1e} (theta <= 50': {np.max(np.abs((sig/mysig-1)[th<=50])):.1e})")
    # (1) node values
    errs = []; errs_rel = []
    for _ in range(250):
        i, j = rng.integers(0, NT, 2)
        if th[i] > 70 or th[j] > 70:
            continue
        S = 5 * th[i] + 5 * th[j] + RB
        ex = exact(th[i], th[j], S * t ** 2)
        r0 = max(1.0, abs(ex[0]))
        errs.append(np.max(np.abs(rho[i, j] - ex)) / r0)
    errs = np.array(errs)
    print(f"   (1) node values vs my exact rho ({len(errs)} pairs x {NU} s): |diff| / max(1, |rho_AB(0)|): median {np.median(errs):.1e}, "
          f"95% {np.percentile(errs, 95):.1e}, max {errs.max():.1e}")

    # (2) builder's lookup scheme at random points
    lth0, dlth = np.log(th[0]), np.log(th[1] / th[0])

    def idx(f, n):
        f = np.clip(f, 0.0, n - 1.000001); i = f.astype(int); return i, f - i

    def lookup(tA, tB, s):
        S = 5 * tA + 5 * tB + RB
        it, wt = idx(np.sqrt(s / S) * (NU - 1), NU)
        ia, wa = idx((np.log(tA) - lth0) / dlth, NT); ib, wb = idx((np.log(tB) - lth0) / dlth, NT)
        r = 0.0
        for da, fa in ((0, 1 - wa), (1, wa)):
            for db, fb in ((0, 1 - wb), (1, wb)):
                r = r + fa * fb * ((1 - wt) * rho[ia + da, ib + db, it] + wt * rho[ia + da, ib + db, it + 1])
        return np.where(s < S, r, 0.0)

    E1 = []; E2 = []; E3 = []
    for _ in range(300):
        tA = np.exp(rng.uniform(np.log(0.3), np.log(40))); tB = np.exp(rng.uniform(np.log(0.3), np.log(40)))
        ss = np.concatenate([rng.uniform(0, 3 * (tA + tB) + 3, 6), rng.uniform(0, 5 * tA + 5 * tB + 60, 2)])
        ex = exact(tA, tB, ss); lk = lookup(np.full(8, tA), np.full(8, tB), ss)
        r0 = abs(exact(tA, tB, [0.0])[0])
        E1.append(np.max(np.abs(lk - ex)) / max(1.0, r0)); E2.append(np.max(np.abs(lk - ex)) / r0)
        E3.append(np.max(np.abs(lk - ex)))
    E1, E2, E3 = map(np.array, (E1, E2, E3))
    print(f"   (2) builder lookup vs my exact rho (300 random pairs, thetas 0.3-40', x 8 s): |diff| / max(1,|rho_AB(0)|): "
          f"median {np.median(E1):.1e}, 95% {np.percentile(E1, 95):.1e}, max {E1.max():.1e}; / |rho_AB(0)|: median "
          f"{np.median(E2):.1e}, 95% {np.percentile(E2, 95):.1e}, max {E2.max():.1e}")
    print(f"       builder claim (s06a): 95% 5.2e-3 rho(0), max 6.4e-3 (vs their own exact rho)")
print(f"total {time.time()-t0:.0f}s")
