"""w01: set-up and checks of the step-2 map machinery (lens: maps).

(1) NUFFT read-out: filtered map evaluated by finufft at pixel centres = the irfft2 map (same coefficients), and at a
    painted cluster's own continuous position = Y (response 1) for the three filters.
(2) Self response R_j(theta) (discrete mode sum on the block) at theta = theta_j: = 1 to the grid accuracy; table of
    R_j over [theta_{j-1}, theta_{j+1}] saved.
(3) Own Hankel tables rho_j(theta_B, s) (dominance only) for baseline, goal, white; checks:
    (a) vs the build's MMFModel.cross_response / SZModel-based white response (build template on both sides);
    (b) vs a painted single halo filtered on the map and read by NUFFT at separations s (exact map response);
    (c) vs step 1's stored rho tables (blendlib.rho_at), for information.
(4) Timing of one patch's steps.
Outputs: out/rho_<mode>.npz, out/self_<mode>.npz
"""
import os, sys, time, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib2 as W                                     # noqa: E402
from vlib2 import V, np                               # noqa: E402

log = W.Tee(os.path.join(W.HERE, "w01_setup.log"))
t0 = time.time()
G = V.Grid()
P = V.Painter()
blk = W.Block(G)
gm = W.gmine_from_paint_tables()
MF = {m: W.ModeFilter(G, P, m, blk) for m in W.MODES}
log(f"[w01] grid {G.n}^2 x {G.pix}' (L = {G.L}'), NUFFT block {blk.N}^2 (|k| <= {W.KB}, l <= {blk.ell[W.KB, -1]:.0f} on the axes); "
    f"threads {W.NTHREADS}; set-up {time.time()-t0:.1f}s")
for m in W.MODES:
    mf = MF[m]
    log(f"   [{m}] {len(mf.th)} nodes {mf.th[0]:.3f}-{mf.th[-1]:.2f}' (dln theta {mf.dlth:.5f}); sigma_Y(nodes) "
        f"{mf.sig[0]:.4e} ... {mf.sig[-1]:.4e} arcmin^2; vs MMFModel/SZModel spline at the nodes: max rel "
        f"{np.abs(mf.sigma_Y(mf.th)/mf.sig-1).max():.1e}")

# ------------------------------------------------------------------ (1) NUFFT read-out
log("\n(1) NUFFT read-out of the filtered map")
rng = np.random.default_rng(11)
for m in W.MODES:
    mf = MF[m]
    for th in (0.4, 1.0, 4.0, 16.0):
        j = int(np.argmin(np.abs(mf.th - th)))
        thj = mf.th[j]
        x0, y0 = rng.random(2) * G.L
        Y = 1e-4
        mp = P.paint(G, np.array([x0]), np.array([y0]), np.array([Y]), np.array([thj]))
        yk = G.rfft(mp)
        c = mf.coef(j, blk.take(yk))
        # irfft map with the same (block-truncated) coefficients
        K2 = np.zeros(yk.shape, complex)
        Tfull = mf.F.yresp_transfer(extra=P.presmooth(G))
        from szsim import signal as sg
        K2 = ((mf.sig[j] * V.ARCMIN ** 2) ** 2 * sg.template_hankel(G.ell * thj * V.ARCMIN) * Tfull) * yk
        mask = (np.abs(np.fft.fftfreq(G.n, 1.0 / G.n))[:, None] <= W.KB) & (np.arange(G.n // 2 + 1)[None, :] <= W.KB)
        Ymap = G.irfft(np.where(mask, K2, 0)) / V.ARCMIN ** 2
        ix = rng.integers(0, G.n, 400); iy = rng.integers(0, G.n, 400)
        f = blk.evaluate(c, (ix + 0.5) * G.pix, (iy + 0.5) * G.pix)
        e1 = np.abs(f - Ymap[iy, ix]).max() / Y
        resp = blk.evaluate(c, [x0], [y0])[0] / Y
        log(f"   [{m:8s}] node {thj:7.3f}': |NUFFT - irfft| at 400 pixel centres max {e1:.1e} Y; response at the "
            f"cluster's own continuous position {resp:.6f} (exact 1 up to the self-response R_j(theta_j))")
        del K2, Ymap, Tfull, mp, yk

# ------------------------------------------------------------------ (2) self response
log("\n(2) self response R_j(theta): node-j filter on a painted unit-Y halo of size theta at its own position")
for m in W.MODES:
    mf = MF[m]
    lt, R = mf.self_response_table(gm, nsub=16)
    Rjj = np.array([np.interp(mf.lth[j], lt[j], R[j]) for j in range(len(mf.th))])
    inner = (mf.th >= 0.25) & (mf.th <= 60)
    log(f"   [{m:8s}] R_j(theta_j) - 1: nodes 0.25-60' max |.| {np.abs(Rjj-1)[inner].max():.1e}; all nodes "
        f"min {np.min(Rjj-1):+.1e} max {np.max(Rjj-1):+.1e}")
    W.save_self(os.path.join(W.OUT, f"self_{m}.npz"), lt, R, mf.th)
    # same quantity by the continuum Hankel route (own rho table at s = 0 is built in (3)); here: build's sigma check
    log(f"      R_j over the ln-theta windows: min {min(r.min() for r in R):.4f}, max {max(r.max() for r in R):.4f}")
log(f"   [{time.time()-t0:.0f}s]")

# ------------------------------------------------------------------ (3) own rho tables
log("\n(3) own Hankel rho_j(theta_B, s) tables (dominance rule only)")
for m in W.MODES:
    mf = MF[m]
    ta = time.time()
    tab, thB, S_dev = W.build_rho_table(mf, gm)
    E = W.envelope(tab)
    np.savez(os.path.join(W.OUT, f"rho_{m}.npz"), tab=tab, E=E, thB=thB, s=W.s_grid(), th=mf.th, sig=mf.sig)
    log(f"   [{m:8s}] table {tab.shape} built in {time.time()-ta:.0f}s; MMF y-response S(l) = sum_i w_i B_i f_i: "
        f"max |S - 1| = {S_dev:.1e}")
    # (a) vs the build's cross_response (build template on both sides) at s = 0 and a few s
    model = mf.model
    worst = 0.0
    for thA in (0.4, 1.0, 4.0, 16.0):
        j = int(np.argmin(np.abs(mf.th - thA)))
        for thb in (0.3, 1.0, 5.0, 20.0):
            iB = int(np.argmin(np.abs(thB - thb)))
            ss = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
            mine = np.array([W._rho_interp(tab[j], np.log(thB[iB]), s_) for s_ in ss])
            if m == "white":
                from szsim import signal as sg
                from scipy.special import j0
                inst = mf.F.inst
                l = np.arange(0.0, 30000.5, 1.0); inv = np.interp(l, inst.ell, inst.inv_Ny)
                kern = l * inv * sg.template_hankel(l * mf.th[j] * V.ARCMIN) * sg.template_hankel(l * thB[iB] * V.ARCMIN)
                ref = np.array([np.trapezoid(kern * j0(l * s_ * V.ARCMIN), l) / (2 * np.pi) for s_ in ss]) * (mf.sig[j] * V.ARCMIN ** 2) ** 2
            else:
                ref = model.cross_response(mf.th[j], thB[iB], ss)
            e = np.abs(mine - ref).max() / max(1.0, np.abs(ref).max())
            worst = max(worst, e)
    log(f"      (a) vs the build's response with the build's template on both sides (thA 0.4-16', thB 0.3-20', s 0-16'): "
        f"max |diff| / max(1, |rho|) = {worst:.1e}  (template ours vs build + S(l))")
    # (b) vs a painted halo on the map, read by NUFFT
    worst = 0.0
    for thA, thb, Yb in ((0.4, 0.4, 3e-5), (1.0, 2.0, 1e-4), (1.0, 0.3, 1e-5), (4.0, 1.0, 1e-4), (16.0, 8.0, 1e-3)):
        j = int(np.argmin(np.abs(mf.th - thA)))
        xb, yb = rng.random(2) * G.L
        mp = P.paint(G, np.array([xb]), np.array([yb]), np.array([Yb]), np.array([thb]))
        c = mf.coef(j, blk.take(G.rfft(mp)))
        ss = np.array([0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0])
        ang = rng.random(ss.size) * 2 * np.pi
        f = blk.evaluate(c, (xb + ss * np.cos(ang)) % G.L, (yb + ss * np.sin(ang)) % G.L) / Yb
        mine = np.array([W._rho_interp(tab[j], np.log(thb), s_) for s_ in ss])
        e = np.abs(mine - f).max() / max(1.0, np.abs(f).max())
        worst = max(worst, e)
        log(f"      (b) thA={mf.th[j]:6.3f}' thB={thb:5.2f}': map rho(s=0,1,2,5,20') = "
            f"{f[0]:+.4f} {f[3]:+.4f} {f[5]:+.4f} {f[7]:+.4f} {f[10]:+.2e}; table - map max |.|/max(1,|rho|) = {e:.1e}")
    log(f"      (b) worst table vs map: {worst:.1e}")
    # (c) vs step 1's stored rho tables (blendlib)
    try:
        sys.path.insert(0, f"{W.ROOT}/analysis/blending")
        import blendlib as bl
        T1 = bl.RhoTable(np.load(f"{W.ROOT}/data_real/rho_mmf_{m}.npz")) if m != "white" else bl.load_rho_table()
        worst = 0.0
        for thA in (0.4, 1.0, 4.0, 16.0):
            j = int(np.argmin(np.abs(mf.th - thA)))
            for thb in (0.3, 1.0, 5.0, 20.0):
                for s_ in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
                    a = bl.rho_at(T1.rho, T1.lth0, T1.dlth, T1.R_BEAM, float(mf.th[j]), float(thb), float(s_))
                    b = W._rho_interp(tab[j], np.log(thb), s_)
                    worst = max(worst, abs(a - b) / max(1.0, abs(a)))
        log(f"      (c) vs step 1's stored rho table (blendlib.rho_at; its interpolation error is up to 6.4e-3 rho(0)): "
            f"max |diff|/max(1,|rho|) = {worst:.1e}")
    except Exception as ex:                                               # noqa: BLE001
        log(f"      (c) skipped: {ex}")
    del tab, E
log(f"   [{time.time()-t0:.0f}s]  peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")

# ------------------------------------------------------------------ (4) timing
log("\n(4) timing of one patch")
S = V.PatchSampler(G.L)
ta = time.time(); h = S.draw(np.random.default_rng([1, 2, 3])); t1 = time.time() - ta
keep = h["th"] <= 60.0
ta = time.time(); mp = P.paint(G, h["x"][keep], h["y"][keep], h["Y"][keep], h["th"][keep]); t2 = time.time() - ta
ta = time.time(); yk = G.rfft(mp); yb = blk.take(yk); t3 = time.time() - ta
ta = time.time(); c = MF["baseline"].coef(20, yb); f = blk.evaluate(c, h["x"][:300], h["y"][:300]); t4 = time.time() - ta
log(f"   {h['x'].size:,} halos ({(~keep).sum()} with theta500 > 60'); draw {t1:.2f}s, paint {t2:.2f}s, rfft + block {t3:.2f}s, "
    f"one node coefficients + NUFFT (300 points) {t4:.3f}s")
log(f"[w01] done {time.time()-t0:.0f}s; peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
