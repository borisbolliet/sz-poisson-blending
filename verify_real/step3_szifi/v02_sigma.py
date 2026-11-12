"""v02: check (a), part 1 -- sigma(theta) and the template, SZiFi conventions vs the build (no S/N maps yet).

All SZiFi behaviour is our own emulation (vsz_lib docstring); SZiFi itself could not be run (permission refused).
(0) framework checks: our zero-padded convolution equals scipy.signal.fftconvolve(mode='same'); our MMF with the build's
    template and covariance reproduces the build's sigma and S/N maps; the SZiFi centring + 'same' crop shifts the
    S/N map by exactly half a pixel.
(1) analytic CMB + noise covariance: sigma(theta) with SZiFi's sampled template (converted to unit Y500) and with the
    sampled template but the build's beams, against the build's sigma_Y; the template ratio in l bands.
(2) estimated covariance, 10 tiles per noise level, the build's template: SZiFi's nearest-bin rule (ties to the lower
    bin) and SZiFi's own spectra (all modes of a bin, so the two bins straddling l = 100 and 20000 are diluted)
    against the build's iteration-0 covariance; r before the iteration with SZiFi's template.
Usage: python -I -B v02_sigma.py
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "2"
sys.path.insert(0, HERE)
import numpy as np
from scipy import fft as sfft
from scipy import signal as ssig
from scipy import integrate
import vsz_lib as V
from szsim import signal as sgl

F, S = V.F, V.S
t0 = time.time()
g = V.szgrid()
G = F.grid()
TH = V.THETAS
TILES = [0, 33, 66, 99, 132, 165, 198, 231, 264, 297]
print(f"v02: SZiFi-convention sigma checks; scales {np.round(TH, 3).tolist()}")
assert np.array_equal(G.m_idx, g.m_idx), "band membership differs"
print(f"  in-band rfft modes: {g.m_idx.size} (identical set for the build's n^2 rule and SZiFi's float l)")

# ------------------------------------------------------------------ (0) framework checks
print("(0) framework checks")
rng = np.random.default_rng(1)
for n in (8, 9, 64):
    a, b = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    P = 2 * n
    mine = sfft.irfft2(sfft.rfft2(a, s=(P, P)) * sfft.rfft2(b, s=(P, P)), s=(P, P))
    s0 = (2 * n - 1 - n) // 2
    mine = mine[s0:s0 + n, s0:s0 + n]
    print(f"  zero-padded convolution vs scipy.signal.fftconvolve(mode='same'), n = {n}: max |diff| = "
          f"{np.abs(mine - ssig.fftconvolve(a, b, mode='same')).max():.1e}")
X, comps = V.make_tile(0, "baseline")
st = V.load_stored_tile(0)
C1 = st["baseline_it1_C"]
fb = F.Filter(G, Cbin=C1)                                    # the build's own filter (iteration-1 covariance)
cx = fb.CX(X)
mine = V.SZFilter(Cbin=C1, mbin=G.m_bin, g=g)               # our MMF, the build's bin assignment
Xm = V.to_modes(X, g)
maps = V.lfiltered_maps(X, g)
lin = V.LinearConv(maps)
for j in (0, 7, 14):
    Tc = V.template_modes(TH[j], "build", g)                 # build template with SZiFi centring phase
    T0 = Tc * np.conj(g.m_phase_c)[None, :]                  # centred
    Fm0, D0 = mine.weights(T0)
    q_b = fb.yhat_map(cx, j) / fb.sigma[j]
    q_m = V.q_circular(Fm0, D0, Xm, g=g)
    Fmc, Dc = mine.weights(Tc)
    q_ref = V.q_circular(Fm0, D0, Xm, shift_half=True, g=g)  # build's map at n - 1/2
    q_c = np.roll(V.q_circular(Fmc, Dc, Xm, g=g), (-(V.N // 2 - 1), -(V.N // 2 - 1)), axis=(0, 1))
    full = ssig.fftconvolve(maps[2], (sfft.irfft2(V.from_modes(Fmc[2], g), s=(V.N, V.N)) * V.N ** 2), mode="same")
    tmp = sfft.irfft2(sfft.rfft2(maps[2], s=(2 * V.N, 2 * V.N)) *
                      sfft.rfft2(sfft.irfft2(V.from_modes(Fmc[2], g), s=(V.N, V.N)) * V.N ** 2, s=(2 * V.N, 2 * V.N)),
                      s=(2 * V.N, 2 * V.N))[V.N // 2 - 1:V.N // 2 - 1 + V.N, V.N // 2 - 1:V.N // 2 - 1 + V.N]
    print(f"  theta = {TH[j]:.2f}': sigma ours / build - 1 = {np.sqrt(1 / D0) / fb.sigma[j] - 1:+.1e}; "
          f"max |q ours - q build| = {np.abs(q_m - q_b).max():.1e}; SZiFi-centred template, periodic, rolled by "
          f"N/2-1 vs build map at n-1/2: {np.abs(q_c - q_ref).max():.1e}; one channel, our padded convolution vs "
          f"scipy fftconvolve 'same' (full size): {np.abs(tmp - full).max() / np.abs(full).max():.1e} (relative)")
del fb, cx, mine, lin, maps, X, comps, Xm

# ------------------------------------------------------------------ (1) analytic covariance: the template
print("(1) analytic CMB + noise covariance: sigma(theta) / the build's sigma_Y (both per unit Y500)")
s_tab, p_tab = V.projected_table()
Ysph1 = integrate.quad(lambda x: 4 * np.pi * x * x * V.gnfw_P(x), 0, 1, limit=400, epsabs=0, epsrel=1e-12)[0]
Ysph5 = integrate.quad(lambda x: 4 * np.pi * x * x * V.gnfw_P(x), 0, 5, limit=400, epsabs=0, epsrel=1e-12)[0]
I_trap = 2 * np.pi * integrate.trapezoid(p_tab * s_tab, s_tab)
sp = sgl.projected_shape(np.array([0.0, 0.5, 1.0, 2.0, 4.0]))
mine_p = np.interp([0.0, 0.5, 1.0, 2.0, 4.0], s_tab, p_tab) / Ysph1
print(f"  our projected profile vs the build's (signal.projected_shape) at s = 0, 0.5, 1, 2, 4: ratio - 1 = "
      + ", ".join(f"{a / b - 1:+.1e}" for a, b in zip(mine_p, sp)))
print(f"  int 2 pi s p(s) ds (trapezoid on SZiFi's 1000 nodes) / int 4 pi x^2 P dx (0..5) - 1 = {I_trap / Ysph5 - 1:+.1e};"
      f" Y5R500/Y500 = {Ysph5 / Ysph1:.5f} (build: {sgl.RATIO_5R500:.5f})")
Y500_per_y0 = lambda th: th ** 2 * Ysph1 / p_tab[0]          # arcmin^2 of spherical Y500 per unit central y
for mode in V.MODES:
    fa = F.Filter(G, mode=mode, analytic=True)
    sb = fa.sigma_arcmin2
    mine = V.SZFilter(analytic_mode=mode, g=g)
    rows = {"ours, build template": [], "SZiFi template": [], "SZiFi sampling, build beams": []}
    tl = {}
    for j, th in enumerate(TH):
        Tb = V.template_modes(th, "build", g)
        rows["ours, build template"].append(V.sigma_from(Tb, mine) / V.AR ** 2)
        Ts = V.template_modes(th, "szifi", g)
        rows["SZiFi template"].append(V.sigma_from(Ts, mine) * Y500_per_y0(th))
        Tbb = V.template_modes(th, "szifi_bb", g)
        rows["SZiFi sampling, build beams"].append(V.sigma_from(Tbb, mine) * Y500_per_y0(th))
        if j in (0, 3, 6, 9, 14):
            # template ratio (beam-free sampled template / build Fourier template), unit Y500, by l band
            ys = Ts[0] / V.szifi_beam(g.m_L, V.FWHM[0]) * np.conj(g.m_phase_c)        # remove beam and centring
            ys = ys * (G.Om / V.AR ** 2) / Y500_per_y0(th)                           # -> continuous FT per unit Y500
            gb = F.gt_lin(G.m_L * th * V.AR)
            out = []
            for lo, hi in ((100, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 15000), (15000, 20000)):
                m = (g.m_L >= lo) & (g.m_L < hi) & (np.abs(gb) > 1e-3 * gb.max())
                out.append(np.median(ys.real[m] / gb[m]) - 1 if m.any() else np.nan)
            tl[th] = out
        del Tb, Ts, Tbb
    print(f"  {mode}: theta [']               " + " ".join(f"{t:7.2f}" for t in TH))
    for k, v in rows.items():
        print(f"    {k:28s} " + " ".join(f"{a / b:7.4f}" for a, b in zip(v, sb)))
    print(f"    template ratio (SZiFi sampled / build), median - 1 in l bands 100-1e3, 1e3-3e3, 3e3-6e3, 6e3-1e4, "
          f"1e4-1.5e4, 1.5e4-2e4 (|g| > 1e-3 g(0)):")
    for th, v in tl.items():
        print(f"      theta = {th:5.2f}': " + ", ".join(f"{x:+.1e}" for x in v))
    del fa, mine
L = np.array([3000., 10000., 20000.])
for fw in (0.9, 1.4, 7.4):
    s2 = (fw * V.AR / np.sqrt(8 * np.log(2))) ** 2
    print(f"  beam convention: B_SZiFi / B_build = exp(-sigma_b^2 l / 2) for FWHM {fw}': at l = 3000, 1e4, 2e4: "
          + ", ".join(f"{np.exp(-0.5 * s2 * l):.6f}" for l in L))

# ------------------------------------------------------------------ (2) estimated covariance conventions
print("(2) estimated covariance (iteration 0), the build's template: SZiFi bin conventions vs the build's")
diff_bin = g.m_near != G.m_bin
print(f"  in-band modes whose SZiFi nearest-centre bin differs from the build's floor(|k|/72): {diff_bin.sum()} of "
      f"{diff_bin.size} ({100 * diff_bin.mean():.3f}%); all at |k| = 72 m exactly: "
      f"{np.all(np.abs(G.m_L[diff_bin] / 72 - np.rint(G.m_L[diff_bin] / 72)) < 1e-9)}")
lo_frac = (G.W * ((g.binm == 1) & G.sel)).sum() / (G.W * (g.binm == 1)).sum()
hi_frac = (G.W * ((g.binm == 277) & G.sel)).sum() / (G.W * (g.binm == 277)).sum()
print(f"  in-band fraction of the modes of bin 1 (72-144): {lo_frac:.4f}; of bin 277 (19944-20016): {hi_frac:.4f}")
res = {m: dict(rB=[], rC=[], rSZ=[], rbuild=[], dil1=[], dil277=[]) for m in V.MODES}
for pid in TILES:
    st = V.load_stored_tile(pid)
    Xt, _ = V.tsz_tile(pid)
    for mode in V.MODES:
        X, _ = V.make_tile(pid, mode, Xt=Xt)
        CA = st[f"{mode}_it0_C"]
        fA = V.SZFilter(Cbin=CA, mbin=G.m_bin, g=g)
        fB = V.SZFilter(Cbin=CA, mbin=g.m_near, g=g)
        CC = V.binned_spectra(np.stack([X[i] * g.band for i in range(6)]), g)
        fC = V.SZFilter(Cbin=CC, mbin=g.m_near, g=g)
        sA, sB, sC = np.zeros(len(TH)), np.zeros(len(TH)), np.zeros(len(TH))
        for j, th in enumerate(TH):
            T = V.template_modes(th, "build", g)
            sA[j], sB[j], sC[j] = V.sigma_from(T, fA), V.sigma_from(T, fB), V.sigma_from(T, fC)
            del T
        res[mode]["rB"].append(sB / sA)
        res[mode]["rC"].append(sC / sA)
        res[mode]["rbuild"].append(sA * 1.0)
        sSZ = np.array([V.sigma_from(V.szifi_T(j, g), fC) for j in range(len(TH))])
        res[mode]["rSZ"].append(sSZ / V.szifi_sigma_analytic(mode, g))
        d = np.arange(6)
        res[mode]["dil1"].append(np.mean(CC[1, d, d] / CA[1, d, d]))
        res[mode]["dil277"].append(np.mean(CC[277, d, d] / CA[277, d, d]))
        chk = np.abs(sA / V.AR ** 2 / st[f"{mode}_it0_sig"] - 1).max()
        print(f"  tile {pid:3d} {mode:8s}: our sigma with the stored iteration-0 covariance / stored sigma - 1: "
              f"max {chk:.1e}  [{time.time() - t0:.0f} s]", flush=True)
        del X, fA, fB, fC
    del Xt
for mode in V.MODES:
    R = {k: np.array(v) for k, v in res[mode].items()}
    sY = F.Filter(G, mode=mode, analytic=True).sigma
    rb = R["rbuild"] / sY[None, :]
    print(f"  {mode} ({len(TILES)} tiles): theta [']                 " + " ".join(f"{t:7.2f}" for t in TH))
    for lab, k in (("SZiFi nearest rule / build", "rB"), ("SZiFi spectra+rule / build", "rC")):
        print(f"    sigma {lab:28s} mean " + " ".join(f"{x:7.5f}" for x in R[k].mean(0)))
        print(f"    {'':34s} min  " + " ".join(f"{x:7.5f}" for x in R[k].min(0)))
        print(f"    {'':34s} max  " + " ".join(f"{x:7.5f}" for x in R[k].max(0)))
    print(f"    r before iteration, build (its template)    " + " ".join(f"{x:7.4f}" for x in rb.mean(0)))
    print(f"    r before iteration, SZiFi conventions       " + " ".join(f"{x:7.4f}" for x in R['rSZ'].mean(0)))
    print(f"    difference SZiFi - build: mean " + " ".join(f"{x:+7.4f}" for x in (R['rSZ'] - rb).mean(0))
          + f"; max |.| over tiles and scales {np.abs(R['rSZ'] - rb).max():.4f}")
    print(f"    dilution of the edge bins (SZiFi spectrum / build, mean of the 6 auto spectra): bin 1 "
          f"{np.mean(R['dil1']):.4f} [{np.min(R['dil1']):.4f}-{np.max(R['dil1']):.4f}], bin 277 "
          f"{np.mean(R['dil277']):.4f} [{np.min(R['dil277']):.4f}-{np.max(R['dil277']):.4f}]")
np.savez(os.path.join(V.OUTD, "v02_sigma.npz"), **{f"{m}_{k}": np.array(v) for m in V.MODES for k, v in res[m].items()})
print(f"done [{time.time() - t0:.0f} s]; peak RSS {V.peak_rss_gb():.2f} GB")
