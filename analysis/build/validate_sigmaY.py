"""Validate the matched-filter sigma_Y(theta500):
 (1) Hankel (l-space) result vs an independent real-space 2D-FFT calculation on a pixel grid
     sigma_Y^-2 = sum_pix (tau*B)^2 * pix_area / sigma_n^2   (Parseval, white noise);
 (2) Monte Carlo: inject a cluster into white-noise maps, apply the matched filter,
     measure mean and std of Y_hat vs the input Y and the predicted sigma_Y.
Prints q for reference (M500, z).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from szsim import signal as sg, cosmo_hmf as ch

FIG = os.path.join(os.path.dirname(__file__), "../../figures/build")
m = sg.SZModel()
d = np.load(sg._HANKEL_CACHE)
sn, fn = d["sn"], d["fn"]
print(f"RATIO Y5R500/Y500 (Arnaud GNFW, 5R500 truncation) = {sg.RATIO_5R500:.5f}")
print(f"Hankel g(0) = {d['g'][0]:.5f} (should equal ratio); rel diff = {d['g'][0]/sg.RATIO_5R500-1:.2e}")
print(f"point-source sigma_Y = {m.sigma_Y_point:.4e} arcmin^2; table sigma_Y(1e-3') = {m.sigma_Y(1e-3):.4e}")


def template_map(theta500, pix, N):
    """Pixel-sampled unit-Y500 template [1/arcmin^2]; returns map (y per unit Y500)."""
    x = (np.arange(N) - N // 2) * pix
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X, Y) / theta500
    # supersample the central pixels to handle the cusp
    t = np.interp(r, sn, fn, right=0.0) / theta500 ** 2
    c = N // 2
    ss = 8
    for di in range(-3, 4):
        for dj in range(-3, 4):
            off = (np.arange(ss) + 0.5) / ss - 0.5
            xx = (di + off[:, None]) * pix; yy = (dj + off[None, :]) * pix
            t[c + di, c + dj] = np.mean(np.interp(np.hypot(xx, yy) / theta500, sn, fn, right=0.0)) / theta500 ** 2
    return t


def beam_convolve(t, pix):
    N = t.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=pix * ch.ARCMIN)  # rad^-1 = multipole l
    L2 = k[:, None] ** 2 + k[None, :] ** 2
    B = np.exp(-0.5 * L2 * m.sigma_beam_rad ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(t)) * B)), B


print("\n(1) Hankel vs real-space 2D FFT")
print(f"{'theta500':>9} {'pix':>6} {'N':>5} {'sum tau':>9} {'sigY_hankel':>12} {'sigY_2D':>12} {'rel diff':>9}")
rows = []
for th, pix, N in [(0.3, 0.02, 2048), (1.0, 0.05, 2048), (3.0, 0.1, 2048), (10.0, 0.25, 2048), (30.0, 0.5, 2048)]:
    t = template_map(th, pix, N)
    tb, _ = beam_convolve(t, pix)
    s2 = (tb ** 2).sum() * pix ** 2 / m.noise_y_arcmin ** 2  # arcmin^-4
    sY2D = 1 / np.sqrt(s2)
    sYH = m.sigma_Y(th)
    rows.append((th, sYH, sY2D))
    print(f"{th:9.2f} {pix:6.2f} {N:5d} {t.sum()*pix**2:9.5f} {sYH:12.5e} {sY2D:12.5e} {sY2D/sYH-1:9.2e}")

print("\n(2) Monte Carlo matched filter on white-noise maps")
rng = np.random.default_rng(42)
pix, N, nmc = 0.25, 256, 20000
for th in [0.5, 2.0, 6.0]:
    t = template_map(th, pix, N)
    tb, _ = beam_convolve(t, pix)
    tb = np.fft.fftshift(tb)
    sig_pix = m.noise_y_arcmin / pix  # y-noise per pixel
    Yin = 10 * m.sigma_Y(th)
    tt = (tb ** 2).sum()
    Yhat = np.empty(nmc)
    for i in range(nmc):
        dmap = Yin * tb + sig_pix * rng.standard_normal((N, N))
        Yhat[i] = (tb * dmap).sum() / tt
    sY_pred_pix = sig_pix / np.sqrt(tt)
    sdev = Yhat.std(ddof=1)
    err_sdev = sdev / np.sqrt(2 * (nmc - 1))
    bias = (Yhat.mean() - Yin)
    print(f"theta500={th:4.1f}': Y_in={Yin:.4e}  <Y_hat>-Y_in = {bias:+.3e} ({bias/(sdev/np.sqrt(nmc)):+.2f} sigma)  "
          f"std(Y_hat)={sdev:.4e} +- {err_sdev:.1e}; sigma_Y(Hankel)={m.sigma_Y(th):.4e} -> "
          f"({(sdev-m.sigma_Y(th))/err_sdev:+.2f} sigma); pixel-grid prediction {sY_pred_pix:.4e} ({(sdev-sY_pred_pix)/err_sdev:+.2f} sigma)")

print("\nReference q = Y500_med / sigma_Y(theta500) (no scatter)")
print(f"{'M500[Msun]':>11} {'z':>5} {'theta500[\']':>11} {'Y500[arcmin2]':>14} {'sigma_Y':>11} {'q':>8}")
for M in [1e14, 2e14, 3e14, 6e14, 1e15]:
    for z in [0.1, 0.3, 0.5, 1.0, 2.0]:
        th = m.theta500(M, z); Y = m.Y500_med(M, z)
        print(f"{M:11.1e} {z:5.2f} {th:11.3f} {Y:14.4e} {m.sigma_Y(th):11.4e} {Y/m.sigma_Y(th):8.2f}")

# figure: sigma_Y(theta500)
tg = np.geomspace(1e-2, 300, 300)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].loglog(tg, m.sigma_Y(tg), label="Hankel (table)")
ax[0].loglog([r[0] for r in rows], [r[2] for r in rows], "o", label="2D FFT check")
ax[0].axhline(m.sigma_Y_point, ls=":", c="k", label="point-source limit")
ax[0].set_xlabel(r"$\theta_{500}$ [arcmin]"); ax[0].set_ylabel(r"$\sigma_Y$ [arcmin$^2$]"); ax[0].legend()
zg = np.linspace(0.02, 3, 200)
for M in [1e14, 3e14, 1e15]:
    ax[1].semilogy(zg, m.q_mean(M, zg), label=f"M500={M:.0e}")
ax[1].axhline(5, ls="--", c="k")
ax[1].set_xlabel("z"); ax[1].set_ylabel("mean q (median Y)"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "sigmaY_and_q.png"), dpi=120)
print("wrote", os.path.join(FIG, "sigmaY_and_q.png"))
