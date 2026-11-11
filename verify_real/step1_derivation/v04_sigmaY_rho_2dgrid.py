"""v04: MMF sigma_Y(theta500) and cross-response rho by DIRECT 2D grid sums (no Hankel / J0 integral),
with my own N_y (Woodbury, CLASS CMB), compared with the builder's stored data_real/sigmaY_mmf.npz and the
values printed in the builder's logs.
  method (a): 2D half-plane l-grid (dl = 10, cross-check dl = 5), template FT g(|l| theta) from the v03 2D-FFT table;
  method (b): fully direct per theta: pixelised template image (pixel theta/16, capped at 0.5'), rfft2, sum of
              |FT|^2 / N_y(|l|) over the modes with 100 <= |l| <= 20000 (exact 1/N_y for every mode).
Also: N_y algebra (Woodbury vs brute-force 6x6 solve), N_y values vs the builder's log, compensation (1/N_y at l->0),
CMB-spectrum sensitivity (CLASS vs builder CAMB; tail beyond l = 6000 zeroed or x3), variants, DC response of the
old white filter, rho for the builder's s04 configurations.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vlib as V
V.set_threads(4)
import numpy as np

t0 = time.time()
AM = V.AM
B = np.load(os.path.join(V.ROOT, "data_real", "sigmaY_mmf.npz"))
TH_B = B["theta500"]

print("=== (0) N_y algebra and values ===")
lt = np.array([150, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000], float)
for mode in ("baseline", "goal"):
    a = V.inv_Ny(lt, mode); b = V.inv_Ny_solve(lt, mode)
    print(f"[{mode}] Woodbury vs 6x6 solve: max |rel| = {np.max(np.abs(a/b-1)):.1e}")
    print(f"[{mode}] N_y [sr] at l={lt.astype(int).tolist()}: {np.array2string(1/a, precision=3)}")
print("builder s02 (3): baseline N_y = [1.550e-14 3.153e-16 3.303e-17 5.090e-18 1.492e-18 1.152e-18 5.769e-18 5.699e-17 4.417e-15]")
print("                 goal     N_y = [1.548e-14 3.114e-16 3.142e-17 3.927e-18 1.132e-18 6.157e-19 2.512e-18 2.396e-17 2.066e-15]")
for mode in ("baseline", "goal"):
    sig = V.FWHM_AM * AM / np.sqrt(8 * np.log(2))
    for l in (3000, 6000, 20000):
        Bl = np.exp(-0.5 * l * l * sig ** 2)
        Nd = ((V.WHITE[mode] * AM) ** 2 + V.NRED_PAPER * V.A_SR / V.T_EFF * (l / 1e3) ** -3.5) / Bl ** 2
        print(f"[{mode}] C_CMB/min_i(N_ii/B_i^2) at l={l}: {V.cmb_cl(np.array([l]))[0]/Nd.min():.2e}")
ll = np.array([1.0, 5.0, 10.0, 50.0])
iv = V.inv_Ny(ll, "baseline")
print(f"compensation: 1/N_y at l = 1, 5, 10, 50: {np.array2string(iv, precision=3)} sr^-1; log-slope l=1..5: "
      f"{np.log(iv[1]/iv[0])/np.log(5):.3f}; 1/N_y(0) = {V.inv_Ny(np.array([0.0]), 'baseline')[0]}")
print("   builder: [2.050e+08 4.102e+10 3.031e+11 8.077e+12], slope 3.29")

print("\n=== (1) sigma_Y by method (a): 2D l-grid + 2D-FFT template ===")
G10 = V.LGrid2D(dl=10.0)
print(f"   grid dl=10: {len(G10.l):,} half-plane modes (dl=5 checks are summed in chunks, never stored)")
# NOTE: a first run of this script stored the full dl=5 grid and its N_y temporaries (peak 10 GB, over the 5 GB
# budget); it was rewritten with chunked sums.  Its numbers were identical to the ones below.
cache = {}


def inv_on(G, key, **kw):
    k = (id(G), key)
    if k not in cache:
        cache[k] = V.inv_Ny_chunked(G.l, **kw)
    return cache[k]


def sig_chunked(th, mode, dl=5.0, lmin=100.0, lmax=20000.0):
    th = np.atleast_1d(th); acc = np.zeros(len(th))
    for lx, ly, l, w in V.grid_modes_chunks(dl, lmin, lmax):
        iv = V.inv_Ny(l, mode)
        for k, t in enumerate(th):
            acc[k] += np.sum(w * V.g_u(l * t * AM) ** 2 * iv)
    return 1 / np.sqrt(acc * dl * dl / (2 * np.pi) ** 2) / AM ** 2


def rho_chunked(mode, tA, tB, s_am, dl=5.0):
    acc = np.zeros(len(s_am)); accA = 0.0
    for lx, ly, l, w in V.grid_modes_chunks(dl, 100.0, 20000.0):
        iv = V.inv_Ny(l, mode); gA = V.g_u(l * tA * AM)
        accA += np.sum(w * gA * gA * iv)
        base = w * gA * V.g_u(l * tB * AM) * iv
        for k, s_ in enumerate(s_am):
            acc[k] += np.sum(base * np.cos(lx * s_ * AM))
    return acc / accA


def sig_a(G, th, invN):
    return np.array([1 / np.sqrt(G.sum(V.g_u(G.l * t * AM) ** 2 * invN)) / AM ** 2 for t in np.atleast_1d(th)])


TH = np.array([0.251, 0.501, 1.0, 1.995, 3.162, 5.012, 10.0, 19.953, 50.119])
res = {}
for mode in ("baseline", "goal"):
    iv10 = inv_on(G10, mode, mode=mode)
    s10 = sig_a(G10, TH_B, iv10); res[mode] = s10
    s5 = sig_chunked(TH, mode)
    sb = B[f"{mode}_l100"]
    print(f"[{mode}] dl=5 vs dl=10: max |rel| = {np.max(np.abs(s5/np.interp(TH, TH_B, s10)-1)):.1e} (at the 9 printed theta)")
    print(f"[{mode}] mine (a) vs builder sigmaY_mmf.npz over 61 theta in [0.1, 100]': max |rel| = {np.max(np.abs(s10/sb-1)):.2e}; "
          f"theta<=50': {np.max(np.abs((s10/sb-1)[TH_B<=50.2])):.2e}")
    print(f"   theta  : " + " ".join(f"{t:9.3f}" for t in TH))
    print(f"   mine(a): " + " ".join(f"{v:9.3e}" for v in np.interp(TH, TH_B, s10)))
    print(f"   builder: " + " ".join(f"{v:9.3e}" for v in np.interp(TH, TH_B, sb)))
# white (old setup), l from 0 (DC cell included) to 40000
Gw = V.LGrid2D(dl=20.0, lmin=0.0, lmax=40000.0)
ivw = V.inv_Ny_white(Gw.l)
sw = sig_a(Gw, TH_B, ivw); res["white"] = sw
del Gw, ivw
print(f"[white] mine (a, dl=20, l<=40000) vs builder 'white' over 61 theta: max |rel| = {np.max(np.abs(sw/B['white']-1)):.2e}")

print("\n=== (2) sigma_Y by method (b): fully direct template images ===")


def sig_b(th, mode=None, white=False, N=4096, lmin=100.0, lmax=20000.0, cmb="class"):
    # pixel: resolve the template (theta/16) AND keep l_Nyq >= 4 l_max (aliasing); box >= 12 theta
    d = min(th / 16.0, np.pi / (4 * lmax) / AM)
    if N * d < 12 * th:
        N = 8192
    i = np.arange(N); x = np.where(i < N // 2, i, i - N) * d
    img = np.empty((N, N))
    for r0 in range(0, N, 512):
        img[r0:r0 + 512] = V.tau_shape(np.hypot(x[r0:r0 + 512, None], x[None, :]) / th) / th ** 2
    F = np.fft.rfft2(img) * d * d
    del img
    Lr = N * d * AM
    kx = np.arange(N // 2 + 1); ky = np.where(i < N // 2, i, i - N)
    LX = 2 * np.pi / Lr * kx[None, :]; LY = 2 * np.pi / Lr * ky[:, None]
    L = np.hypot(LX, LY)
    if white:
        sel = L <= lmax
        iv = V.inv_Ny_white(L[sel])
    else:
        sel = (L >= lmin) & (L <= lmax)
        iv = V.inv_Ny_chunked(L[sel], mode=mode, cmb=cmb)
    w = np.where(np.broadcast_to(kx[None, :], L.shape)[sel] == 0, 1.0, 2.0)
    w = np.where(np.broadcast_to(kx[None, :], L.shape)[sel] == N // 2, 1.0, w)
    S = np.sum(w * F.real[sel] ** 2 * iv) / Lr ** 2
    del F, L, LX, LY
    return 1 / np.sqrt(S) / AM ** 2, d, 2 * np.pi / Lr, np.pi / (d * AM)


for mode in ("baseline", "goal"):
    rows = []
    for th in TH:
        sb_, d, dl, lny = sig_b(th, mode)
        sa = np.interp(th, TH_B, res[mode])
        rows.append((th, sb_, sa, np.interp(th, TH_B, B[f"{mode}_l100"]), d, dl, lny))
    print(f"[{mode}] theta  pixel  dl   l_Nyq | sigma(b)   sigma(a)   builder  | (b)/(a)-1  (b)/builder-1")
    for th, sb_, sa, bb, d, dl, lny in rows:
        print(f"   {th:7.3f} {d:6.4f} {dl:6.1f} {lny:8.0f} | {sb_:.4e} {sa:.4e} {bb:.4e} | {sb_/sa-1:+.1e}   {sb_/bb-1:+.1e}")
rows = []
for th in TH:
    sb_, d, dl, lny = sig_b(th, white=True, lmax=40000.0)
    rows.append((th, sb_, np.interp(th, TH_B, B["white"])))
print("[white] (b) vs builder: " + "  ".join(f"{th:.2f}': {s/b-1:+.1e}" for th, s, b in rows))
try:
    sys.path.insert(0, V.ROOT)
    from szsim import signal as sgold      # the OLD validated single-frequency model (not the builder's new code)
    so = sgold.SZModel().sigma_Y(TH)
    print("[white] (a) vs old szsim SZModel.sigma_Y: " + " ".join(f"{v:+.1e}" for v in np.interp(TH, TH_B, sw) / so - 1))
except Exception as e:
    print("old SZModel not available:", e)

print("\n=== (3) ratios, equivalent white noise, variants, CMB sensitivity (method a, dl = 10) ===")
rb = np.interp(TH, TH_B, res["baseline"]) / np.interp(TH, TH_B, res["white"])
rg = np.interp(TH, TH_B, res["goal"]) / np.interp(TH, TH_B, res["white"])
print("   ratio baseline/white: " + " ".join(f"{v:.3f}" for v in rb))
print("   ratio goal/white    : " + " ".join(f"{v:.3f}" for v in rg))
print("   builder baseline/white: 0.783 0.801 0.859 1.015 1.249 1.705 3.319 7.733 28.277 ; goal 0.575 0.596 0.654 0.796 1.005 1.411 2.873 7.014 26.878")
var = {"noCMB": dict(cmb="none"), "noATM": dict(atm=False), "atmcorr0": dict(r_atm=0.0), "145only": dict(channels=[3]),
       "93+145": dict(channels=[2, 3]), "CAMB_builder": dict(cmb="camb_builder")}
for k, kw in var.items():
    s = sig_a(G10, TH, V.inv_Ny_chunked(G10.l, mode="baseline", **kw))
    bk = f"baseline_{k}" if f"baseline_{k}" in B.files else None
    line = " ".join(f"{v:9.3e}" for v in s)
    cmp = (" | vs builder: " + " ".join(f"{v:+.1e}" for v in s / np.interp(TH, TH_B, B[bk]) - 1)) if bk else \
          (" | rel to CLASS: " + " ".join(f"{v:+.1e}" for v in s / np.interp(TH, TH_B, res['baseline']) - 1))
    print(f"   baseline {k:13s}: {line}{cmp}")
# CMB tail sensitivity: modify the CLASS spectrum above l = 6000
le = np.arange(0, 26001).astype(float); clc = V.cmb_cl(le)
for tag, fac in (("tail>6000 = 0", 0.0), ("tail>6000 x3", 3.0)):
    V._CMB["mod"] = (le, np.where(le > 6000, fac * clc, clc))
    s = sig_a(G10, TH, V.inv_Ny_chunked(G10.l, mode="baseline", cmb="mod"))
    print(f"   baseline CMB {tag:14s}: rel to CLASS " + " ".join(f"{v:+.1e}" for v in s / np.interp(TH, TH_B, res['baseline']) - 1))
G36 = V.LGrid2D(dl=10.0, lmin=36.0)
s36 = sig_a(G36, TH, V.inv_Ny_chunked(G36.l, mode="baseline")); del G36
print("   baseline lmin=36 vs 100: " + " ".join(f"{v:+.1e}" for v in s36 / np.interp(TH, TH_B, res['baseline']) - 1))

print("\n=== (4) DC response of the old white filter: sigma^2 RATIO^2 / N_w ===")
for th in (1.0, 4.0, 16.0):
    s = np.interp(th, TH_B, res["white"]) * AM ** 2
    print(f"   theta500 = {th:4.1f}': DC = {s**2 * V.RATIO**2 / (4e-6*AM)**2 / AM**2:8.2f} arcmin^2  (builder 10.54 / 69.31 / 936.75)")

print("\n=== (5) cross-response rho on the 2D l-grid (method a, dl = 10; dl = 5 check) vs builder s04 log ===")
seps = np.array([0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0])
BLD = {"baseline": {(1, 1): [1.0000e+00, 5.7290e-01, -1.5100e-01, 7.6905e-03, -7.0119e-03, 1.5031e-04, -7.3473e-05, -2.3054e-05],
                    (2, 4): [3.9895e-01, 2.7737e-01, -1.6275e-02, -1.8631e-02, -4.8841e-03, 1.0273e-04, -2.9341e-05, -1.2072e-05],
                    (4, 2): [2.3746e+00, 1.6509e+00, -9.6873e-02, -1.1089e-01, -2.9070e-02, 6.1144e-04, -1.7464e-04, -7.1852e-05],
                    (8, 8): [1.0000e+00, 8.1733e-01, 2.4101e-01, -6.6575e-02, -3.0621e-02, 7.0014e-04, 9.9207e-07, -4.3522e-05],
                    (2, 16): [1.9473e-02, 1.5035e-02, 2.5388e-03, -1.2345e-03, -4.4382e-04, 1.0023e-05, 8.9633e-07, -1.6336e-06]},
       "goal": {(1, 1): [1.0000e+00, 5.2961e-01, -1.7144e-01, 2.2708e-02, -2.6980e-03, -2.7161e-05, -7.1598e-05, -1.2456e-05],
                (2, 4): [3.8349e-01, 2.5420e-01, -3.1692e-02, -9.1392e-03, -2.6534e-03, 2.5949e-06, -3.4054e-05, -7.5953e-06],
                (4, 2): [2.4681e+00, 1.6360e+00, -2.0396e-01, -5.8818e-02, -1.7077e-02, 1.6700e-05, -2.1917e-04, -4.8882e-05],
                (8, 8): [1.0000e+00, 7.9477e-01, 1.8499e-01, -6.6908e-02, -2.1855e-02, 3.2080e-04, -5.8353e-05, -2.8957e-05],
                (2, 16): [1.7317e-02, 1.2868e-02, 1.2669e-03, -9.1720e-04, -2.6303e-04, 3.4101e-06, -7.7930e-08, -1.2035e-06]}}


def rho_a(G, mode, tA, tB, s_am, invN):
    sA2 = (1 / G.sum(V.g_u(G.l * tA * AM) ** 2 * invN))
    base = G.w * V.g_u(G.l * tA * AM) * V.g_u(G.l * tB * AM) * invN
    return np.array([sA2 * G.norm * np.sum(base * np.cos(G.lx * s * AM)) for s in s_am])


worst = 0.0
for mode in ("baseline", "goal"):
    for (a, b), ref in BLD[mode].items():
        r10 = rho_a(G10, mode, a, b, seps, inv_on(G10, mode, mode=mode))
        r5 = rho_chunked(mode, a, b, seps)
        dmax = np.max(np.abs(r10 - np.array(ref))); worst = max(worst, dmax)
        print(f"[{mode}] thA={a} thB={b}: mine {np.array2string(r10, precision=4, max_line_width=200)}")
        print(f"{'':22s}builder {np.array2string(np.array(ref), precision=4, max_line_width=200)}  max|diff| {dmax:.1e}; "
              f"dl5-dl10 {np.max(np.abs(r5-r10)):.1e}")
print(f"   worst |mine - builder| over the 10 configurations x 8 separations: {worst:.2e}")
np.savez(os.path.join(HERE, "sigmaY_mine.npz"), theta500=TH_B, baseline=res["baseline"], goal=res["goal"], white=res["white"])
print(f"total {time.time()-t0:.0f}s")
