"""Step 1 / s02: verify the SO LAT noise model and the tSZ SED of szsim/instrument.py.

(1) Noise: szsim.instrument.so_lat_noise (paper Tables 1-2) vs an independent re-implementation, written
    here, of the algorithm of the public SO code (simonsobs/so_noise_models, SO_Noise_Calculator_Public_v3_1_2.py,
    class SOLatV3 = atmospheric model version 0 = the [SO19] model, reference elevation).  The downloaded file
    was READ, not executed (execution of downloaded code was not permitted); its constants are transcribed:
      per-tube NETs [uK sqrt(s)] baseline LF (48, 24), MF (5.4, 6.7)*sqrt(4), UHF (15, 36)*sqrt(2);
                                 goal     LF (35, 18), MF (3.9, 4.2)*sqrt(4), UHF (10, 25)*sqrt(2);
      tubes LF 1, MF 4, UHF 2; band NET = (sum_tubes NET_tube^-2)^-1/2;
      white [uK^2 arcmin^2] = NET^2 * A_arcmin2 / t_eff,  A = 4 pi f_sky, t_eff = 5 yr * 0.2 * 0.85;
      atmosphere: C_v0 = (200, 77, 1800, 12000, 68000, 124000) uK^2 s, times Tatmos_FOV_mod = 0.5, divided by
      the number of tubes carrying the band; N_red(l) = C' (l/1000)^-3.5 * A_sr / t_eff; correlation 0.9
      within a tube's band pair (cov = 0.9 sqrt(C'_i C'_j) ...).
    Expected differences: Table 1 rounds the white levels; Table 2 rounds N_red at 39 (38.5) and 93 GHz (225).
(1b) Lensed CMB C_l relative to the beam-deconvolved noise of the quietest channel (where does the CMB matter).
(2) tSZ SED: closed form g(x) = x coth(x/2) - 4 vs an independent numerical route: Kompaneets intensity change
    Delta I = y I0 x^4 e^x/(e^x-1)^2 [x coth(x/2) - 4] converted to Delta T with a finite-difference dB/dT of
    the Planck law.  Plus the null frequency (literature: 217.5 GHz).
(3) Effective-noise algebra: N_y(l) = 1/(a^T C^-1 a) computed by np.linalg.solve (library) vs an
    eigen-decomposition inverse here, and vs the Sherman-Morrison closed form for the CMB rank-1 term.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np
from szsim import instrument as ins


NET = {"baseline": dict(LF=[48., 24.], MF=[5.4 * 2, 6.7 * 2], UHF=[15. * 2 ** .5, 36. * 2 ** .5]),
       "goal": dict(LF=[35., 18.], MF=[3.9 * 2, 4.2 * 2], UHF=[10. * 2 ** .5, 25. * 2 ** .5])}
NTUBE = dict(LF=1, MF=4, UHF=2)
C_V0 = np.array([200., 77., 1800., 12000., 68000., 124000.])


def so_reimpl(mode, ell, fsky=0.4):
    """Re-implementation of SOLatV3(mode).get_white_noise / get_noise_curves(full_covar, deconv_beam=False)."""
    t = 5 * 365.25 * 86400. * 0.2 * 0.85
    A_sr = 4 * np.pi * fsky
    A_am = A_sr * (60 * 180 / np.pi) ** 2
    band_net, ntube = np.zeros(6), np.zeros(6)
    for k, (tube, idx) in enumerate([("LF", (0, 1)), ("MF", (2, 3)), ("UHF", (4, 5))]):
        for b, i in enumerate(idx):
            band_net[i] = (NTUBE[tube] * NET[mode][tube][b] ** -2) ** -0.5
            ntube[i] = NTUBE[tube]
    white_am = band_net ** 2 * A_am / t                     # uK^2 arcmin^2
    Cp = C_V0 * 0.5
    cov = np.zeros((6, 6))
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        for a in (i, j):
            for b in (i, j):
                corr = 1.0 if a == b else 0.9
                cov[a, b] = corr * np.sqrt(Cp[a] * Cp[b]) / ntube[a]
    shape = (ell / 1000.) ** -3.5
    N = cov[:, :, None] * shape[None, None, :] * A_sr / t
    for i in range(6):
        N[i, i] += band_net[i] ** 2 * A_sr / t
    return np.sqrt(white_am), N


def main():
    ell_check = np.array([100, 300, 1000, 2000, 3000, 5000, 8000, 12000, 20000])
    print("=== (1) noise model (paper tables) vs re-implementation of the SO code algorithm (NET based) ===")
    for mode in ("baseline", "goal"):
        # [arbiter fix] full filter range l = 2..20000 (the first version stopped at 10000 and printed l <= 8000)
        ell = np.arange(2, 20001, 1).astype(float)
        wn, T = so_reimpl(mode, ell)
        print(f"[{mode}] white noise uK-arcmin  code: {np.round(wn, 2)}")
        print(f"[{mode}]                        paper: {ins.SO_LAT['white'][mode]}  ratio paper/code: "
              f"{np.round(ins.SO_LAT['white'][mode] / wn, 4)}")
        Nm = ins.so_lat_noise(ell, mode)
        k = np.searchsorted(ell, ell_check)
        for i in range(6):
            r = Nm[i, i, k] / T[i, i, k]
            print(f"   ch {ins.SO_LAT['freqs'][i]:5.0f} GHz: N_ii(mine)/N_ii(code) at l={list(ell_check)}: {np.round(r, 4)}")
        for (i, j) in ins.SO_LAT["tubes"]:
            r = Nm[i, j, k] / T[i, j, k]
            print(f"   pair {ins.SO_LAT['freqs'][i]:.0f}x{ins.SO_LAT['freqs'][j]:.0f}: N_ij(mine)/N_ij(code): {np.round(r, 4)}")
        # [arbiter fix] maximum over the full grid l = 2..20000 and every nonzero entry; analytic check: at high l the
        # white term dominates, so N_ii(mine)/N_ii(code) -> (white_paper / white_code)^2
        worst = (0.0, None, None)
        for i in range(6):
            for j in range(6):
                if np.all(T[i, j] == 0):
                    continue
                d = np.abs(Nm[i, j] / T[i, j] - 1)
                k = int(np.argmax(d))
                if d[k] > worst[0]:
                    worst = (d[k], (i, j), ell[k])
        asym = (ins.SO_LAT["white"][mode] / wn) ** 2 - 1
        print(f"   full grid l = 2..20000, all nonzero N_ij: max |mine/code - 1| = {worst[0]:.4f} at "
              f"{ins.SO_LAT['freqs'][worst[1][0]]:.0f}x{ins.SO_LAT['freqs'][worst[1][1]]:.0f} GHz, l = {worst[2]:.0f}; "
              f"white-limit (paper/code)^2 - 1 per channel: {np.round(asym, 4)} (max |.| {np.abs(asym).max():.4f})")
        off = [(i, j) for i in range(6) for j in range(6) if i != j and (min(i, j), max(i, j)) not in ins.SO_LAT["tubes"]]
        print(f"   cross-tube entries: max |mine| = {max(abs(Nm[i, j]).max() for i, j in off):.1e}, "
              f"max |code| = {max(abs(T[i, j]).max() for i, j in off):.1e}")
        # knee where atmosphere = white, per channel
        N0 = ins.so_lat_noise(np.array([1000.0]), mode, atm=True)[:, :, 0].diagonal()
        Nw = ins.so_lat_noise(np.array([1000.0]), mode, atm=False)[:, :, 0].diagonal()
        lk = 1000 * ((N0 - Nw) / Nw) ** (1 / 3.5)
        print(f"   effective knee l (red = white): {np.round(lk, 0)}")

    print("\n=== (1b) lensed CMB power relative to the beam-deconvolved noise of the quietest channel at each l ===")
    for mode in ("baseline", "goal"):
        ell = np.array([1000., 2000., 3000., 4000., 6000., 8000., 12000., 20000.])
        N = ins.so_lat_noise(ell, mode)
        Bl = ins.beam_l(ell, ins.SO_LAT["fwhm"])
        nmin = np.min(np.array([N[i, i] / Bl[i] ** 2 for i in range(6)]), axis=0)
        print(f"[{mode}] C_l^CMB / min_i(N_ii/B_i^2) at l={list(ell.astype(int))}: {np.array2string(ins.cmb_cl_TT(ell) / nmin, precision=2)}")

    print("\n=== (2) tSZ SED ===")
    nus = np.array([27.0, 39.0, 93.0, 145.0, 150.0, 217.0, 225.0, 280.0, 353.0])
    g1 = ins.tsz_g(nus)
    hk = ins.H_PLANCK / ins.K_BOLTZ
    T = ins.T_CMB

    def B(nu, Tk):
        x = hk * nu / Tk
        return nu ** 3 / np.expm1(x)

    g2 = []
    for nu in nus * 1e9:
        x = hk * nu / T
        # the Kompaneets form: Delta I = y I0 x^4 e^x/(e^x-1)^2 (x coth(x/2) - 4), I0 propto nu^3/x^3 -> dI/y below
        I0 = (nu / x) ** 3
        dI = I0 * x ** 4 * np.exp(x) / np.expm1(x) ** 2 * (x / np.tanh(x / 2) - 4)
        h = 1e-4 * T
        dBdT = (B(nu, T + h) - B(nu, T - h)) / (2 * h)
        g2.append(dI / dBdT / T)
    g2 = np.array(g2)
    for nu, a, b in zip(nus, g1, g2):
        print(f"   {nu:6.1f} GHz: g closed form = {a:+.5f}; via Kompaneets + numerical dB/dT = {b:+.5f}; "
              f"diff {a-b:+.1e};  f = {ins.tsz_f_uK(nu):+9.1f} uK per unit y")
    from scipy.optimize import brentq
    nu0 = brentq(lambda n: ins.tsz_g(n), 150, 300)
    print(f"   null frequency: {nu0:.2f} GHz (literature 217.5 GHz for T_CMB = 2.7255 K)")

    print("\n=== (3) N_y algebra: library solve vs eigendecomposition vs Sherman-Morrison ===")
    for mode in ("baseline", "goal"):
        I = ins.Instrument.so_lat(mode, lmin=100, lmax=20000)
        ell = np.array([150.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 8000.0, 12000.0, 20000.0])
        Bl = ins.beam_l(ell, ins.SO_LAT["fwhm"])
        f = ins.tsz_f_uK(ins.SO_LAT["freqs"])
        N = ins.so_lat_noise(ell, mode)
        Ccmb = ins.cmb_cl_TT(ell)
        r_eig, r_sm = [], []
        for k in range(len(ell)):
            a = Bl[:, k] * f
            C = N[:, :, k] + np.outer(Bl[:, k], Bl[:, k]) * Ccmb[k]
            ev, V = np.linalg.eigh(C)
            inv_eig = a @ V @ np.diag(1 / ev) @ V.T @ a
            Ni = np.linalg.inv(N[:, :, k]); b = Bl[:, k]
            # Sherman-Morrison: (N + c b b^T)^-1 = N^-1 - c N^-1 b b^T N^-1 / (1 + c b^T N^-1 b)
            inv_sm = a @ Ni @ a - Ccmb[k] * (a @ Ni @ b) ** 2 / (1 + Ccmb[k] * b @ Ni @ b)
            lib = np.interp(ell[k], I.ell, I.inv_Ny)
            r_eig.append(lib / inv_eig - 1); r_sm.append(lib / inv_sm - 1)
        print(f"[{mode}] max |lib/eig - 1| = {np.abs(r_eig).max():.1e}; max |lib/SM - 1| = {np.abs(r_sm).max():.1e}")
        Ny = I.Ny(ell)
        print(f"[{mode}] N_y(l) [sr] at l={list(ell.astype(int))}: {np.array2string(Ny, precision=3)}")
        print(f"[{mode}] equivalent white y-noise of a 1.4' map, sqrt(N_y B_1.4^2) [arcmin]: "
              f"{np.array2string(I.equivalent_white_y(ell), precision=3)}  (old setup: 4e-6)")
        wl = np.array([np.interp(L, I.ell, I.weights[:, i]) for L in (500, 2000, 5000) for i in range(6)]).reshape(3, 6)
        # w_i a_i (beam-deconvolved: weights act on beam-convolved maps, the y response is sum_i w_i B_i f_i = 1)
        for L, row in zip((500, 2000, 5000), wl):
            print(f"      l={L}: w_i B_i f_i = {np.round(row * ins.beam_l([L], ins.SO_LAT['fwhm'])[:, 0] * f, 3)}  "
                  f"(sum {np.sum(row * ins.beam_l([L], ins.SO_LAT['fwhm'])[:, 0] * f):.6f})")


if __name__ == "__main__":
    main()
