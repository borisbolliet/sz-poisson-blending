"""Step 3 / t01: check (a) and the exactness of the decomposition.

(a) Given the true analytic noise covariance (lensed CMB + white noise + atmosphere, no tSZ), the finder's S/N equals
    step 1's MMF:
    (a1) sigma(theta) of the finder (szsim.finder.Filter, analytic per-mode covariance, step 1's template) against
         szsim.instrument.MMFModel.sigma_Y_exact (Hankel integral) at the 15 finder scales; also the smooth template
         (cubic in ln u) the painting uses;
    (a2) S/N maps of one tile (pid 0, baseline, all halos) from the finder against an independent re-implementation of
         analysis/visual/v01_patches.py Part B (explicit per-mode 6x6 solve on the full grid, Melin Psi form): max |dq|.
(d) Decomposition.  The response of the tile's filter to a halo is (lattice sum over Fourier modes)
        R(D) = sum_k w_k tau_d(k) aCa(k) G_B(k) cos(k.D) / norm_d .
    (d1) radial tables (angle-averaged lattice response, szsim.finder.response_tables) against the exact 2D lattice sum
         at several offsets and directions, for the analytic covariance and for the binned covariance estimated from
         a tile;
    (d2) closure on tile 0 with the ANALYTIC filter: the exact filtered tSZ value at every detection against the sum
         over all 71,291 halos of their table contributions;
    (d3) the same with the finder's estimated filter (as in production).
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
from scipy import fft as sfft
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                  # noqa: E402
from szsim import instrument as ins, signal as sg  # noqa: E402
F = S.F
t0 = time.time()
G = F.grid()
TH = F.THETAS15

# ------------------------------------------------------------------ (a1)
print("(a1) finder sigma(theta) with the analytic covariance vs step 1's MMFModel.sigma_Y_exact (l = 100-20000)")
for mode in S.MODES:
    lib = ins.MMFModel(ins.Instrument.so_lat(mode, lmin=100.0, lmax=20000.0)).sigma_Y_exact(TH)
    fl = F.Filter(G, mode=mode, analytic=True, template="lin")
    fc = F.Filter(G, mode=mode, analytic=True, template="cub")
    rl, rc = fl.sigma_arcmin2 / lib - 1, fc.sigma_arcmin2 / lib - 1
    print(f"  {mode}: step-1 template: max |sigma/lib - 1| = {np.abs(rl).max():.1e} "
          f"(theta = {TH[np.abs(rl).argmax()]:.2f}'); smooth template: {rc.min():+.1e} .. {rc.max():+.1e}")
    print(f"     sigma [arcmin^2] at 0.5, 1.04, 2.15, 4.45, 9.23, 15': "
          + ", ".join(f"{v:.4e}" for v in fl.sigma_arcmin2[[0, 3, 6, 9, 12, 14]]))
    print(f"     sigma_Y spline (steps 1-2, signal.SZModel.sigma_Y) / exact - 1: max |.| = "
          f"{np.abs(S.sigma_Y(mode, TH) / lib - 1).max():.1e}")
    del fl, fc

# ------------------------------------------------------------------ (a2) S/N maps vs an independent v01 Part B route
print("(a2) S/N maps of tile 0 (baseline, all halos): finder with the analytic covariance vs a re-implementation of "
      "v01 Part B")
tables = F.build_paint_tables()
d = S.truth(0)
Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], tables, G), G)
X = Xt + F.sim_cmb(S.SEED_CMB, G) + F.sim_noise(S.SEED_NOISE, "baseline", G)
its = F.run_finder(X, G, analytic_mode="baseline", keep_cube=True)
qf = its[0]["q"]
# v01 Part B route, written out again: full grid, explicit 6x6 covariance per mode, solve, Psi = tau C^-1 a / norm
AR, Om, Npx = ins.ARCMIN, G.Om, G.Npx
Lm = G.L
fw, fr = ins.SO_LAT["fwhm"], ins.SO_LAT["freqs"]
Bl = np.exp(-0.5 * (Lm[None] * (fw[:, None, None] * AR / np.sqrt(8 * np.log(2)))) ** 2)
f_uK = ins.tsz_f_uK(fr)
sel = (Lm >= 100.0) & (Lm <= 20000.0)
Cia = np.zeros(Lm.shape + (6,))
a_ = np.moveaxis(Bl * f_uK[:, None, None], 0, -1)
Ls = Lm[sel]
Cs = np.moveaxis(ins.so_lat_noise(Ls, "baseline"), 2, 0) + (Bl[:, sel].T[:, :, None] * Bl[:, sel].T[:, None, :]) * ins.cmb_cl_TT(Ls)[:, None, None]
Cia[sel] = np.linalg.solve(Cs * (Npx / Om), a_[sel][..., None])[..., 0]
del Cs
aCa = np.einsum("...i,...i->...", a_, Cia)
wcol = np.full(Lm.shape[1], 2.0); wcol[0] = 1.0; wcol[-1] = 1.0
CX = np.einsum("xyi,ixy->xy", Cia, X)
dq = []
for k, tk in enumerate(TH):
    tau = sg.template_hankel(Lm * tk * AR) / Om
    norm = np.sum(wcol[None, :] * tau ** 2 * aCa)
    q = np.fft.irfft2(tau * CX / norm, s=(F.N, F.N)) * Npx * np.sqrt(norm)
    dq.append(np.abs(q - qf[k]).max())
print(f"  max |q_finder - q_v01route| over the 15 scales and all 1.44e6 pixels: {max(dq):.1e} "
      f"(per scale: {', '.join(f'{v:.0e}' for v in dq[::3])} ...)")
del CX, Cia, aCa, Bl, a_

# ------------------------------------------------------------------ (d1) radial tables vs exact lattice sums
print("(d1) radial (angle-averaged lattice) response tables vs exact 2D lattice sums R(D); theta_d = 1.04', unit-Y500 halo")
basis = F.ResponseBasis(G, tables)
Xcn = F.sim_cmb(S.SEED_CMB + 7, G) + F.sim_noise(S.SEED_NOISE + 7, "baseline", G)
filt_est = F.Filter(G, Cbin=F.binned_cov(Xcn, G))
filt_ana = F.Filter(G, mode="baseline", analytic=True)
del Xcn
for lab, fl in (("analytic covariance", filt_ana), ("estimated binned covariance (nearest)", filt_est)):
    Rn, Rf = F.response_tables(fl, basis)
    for thB in (1.0, 0.31, 4.7):
        i0, w = F.theta_node(np.log(thB), basis.lnth0, basis.dln, len(basis.th_nodes))
        rows = []
        for r in (0.0, 0.3, 1.0, 2.5, 5.0, 10.0, 30.0, 100.0):
            tab = F._resp_eval(Rn, Rf, 3, i0, w, r, F.R_SWITCH, basis.rn[1] - basis.rn[0], F.R_FAR_MIN, basis.rf[1] - basis.rf[0])
            ex = [F.exact_contrib(fl, basis, 3, 1.0, thB, r * np.cos(p), r * np.sin(p)) for p in (0.0, 0.37, np.pi / 4)]
            rows.append((r, tab, np.max(np.abs(np.array(ex) - tab))))
        R0 = F.exact_contrib(fl, basis, 3, 1.0, thB, 0.0, 0.0)
        print(f"  {lab}, theta_B = {thB}': R(0) = {R0:.6f}; max over 3 directions |lattice - table| / R(0) at r = "
              + ", ".join(f"{r:g}': {e / abs(R0):.1e}" for r, t_, e in rows))
    del Rn, Rf

# ------------------------------------------------------------------ (d2), (d3) closure on tile 0
cxs = dict(tsz=None, cmb=None, noise=None)
Xc = F.sim_cmb(S.SEED_CMB, G); Xn = F.sim_noise(S.SEED_NOISE, "baseline", G)
for lab, it in (("analytic filter", its[0]), ("estimated filter (finder, iteration 1)", F.run_finder(X, G)[-1])):
    fl = it["filt"]
    if it["q"] is not None:
        it = dict(it, q=None)
    cx = dict(tsz=fl.CX(Xt), cmb=fl.CX(Xc), noise=fl.CX(Xn))
    t1 = time.time()
    dec = S.decompose(it, cx, d, basis, "q_baseline", forced=False)
    det = it["det"]
    sig = fl.sigma_arcmin2[det["j"]]
    tot = dec["d_S_tot"]
    comp = dec["d_S_tsz"] + dec["d_S_cmb"] + dec["d_S_noise"]
    rel_c = np.abs(tot - comp) / np.abs(tot)
    res = dec["d_S_tsz"] - dec["d_tab_sum"]
    m = np.isfinite(dec["d_top_ex"][:, 0])
    top_rel = np.abs(dec["d_top_ex"][m, 0] / dec["d_top_tab"][m, 0] - 1)
    print(f"(d{2 if 'analytic' in lab else 3}) tile 0, baseline, {lab}: {det['q'].size} detections q > 4 [{time.time() - t1:.1f} s]")
    print(f"     map value = tSZ + CMB + noise: max relative difference {rel_c.max():.1e}")
    print(f"     tSZ value - sum over all {d['x'].size} halos of the table contributions: median |.| = "
          f"{np.median(np.abs(res) / np.abs(tot)):.1e} of the map value ({np.median(np.abs(res) / sig):.1e} sigma), "
          f"max {np.max(np.abs(res) / np.abs(tot)):.1e} ({np.max(np.abs(res) / sig):.1e} sigma)")
    print(f"     largest contributor (q >= 4.5): exact lattice / table - 1: median {np.median(top_rel):.1e}, max {top_rel.max():.1e}")
print(f"done [{time.time() - t0:.0f} s]")
