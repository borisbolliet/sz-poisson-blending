"""Validate SZModel.cross_response (Hankel) against a direct 2D pixel-space matched filter:
place a unit-Y500 cluster of size theta_B at separation r from A, beam-convolve, and apply
A's matched filter (white noise => filter = beam-convolved template of A):
   Y_hat_A = sum(t_A * t_B(shifted)) / sum(t_A^2)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
from szsim import signal as sg, cosmo_hmf as ch

m = sg.SZModel()
pix, N = 0.05, 2048


def beam_template(theta500, shift_pix=0):
    x = (np.arange(N) - N // 2) * pix
    X, Y = np.meshgrid(x - shift_pix * pix, x, indexing="ij")
    ss = 4  # 4x4 supersampling of every pixel (cusp-safe)
    t = np.zeros((N, N))
    off = (np.arange(ss) + 0.5) / ss - 0.5
    for a in off:
        for b in off:
            t += sg.template_real(np.hypot(X + a * pix, Y + b * pix), theta500)
    t /= ss * ss
    k = 2 * np.pi * np.fft.fftfreq(N, d=pix * ch.ARCMIN)
    B = np.exp(-0.5 * (k[:, None] ** 2 + k[None, :] ** 2) * m.sigma_beam_rad ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(t) * B))


print(f"{'thA':>5} {'thB':>5} {'sep':>5} {'rho_Hankel':>11} {'rho_2D':>11} {'abs diff':>9}")
maxd = 0
for thA in [1.0, 3.0]:
    tA = beam_template(thA)
    nA = (tA * tA).sum()
    for thB in [0.5, 1.0, 3.0]:
        for sep in [0.0, 1.0, 2.0, 4.0, 8.0]:
            tB = beam_template(thB, shift_pix=int(round(sep / pix)))
            r2d = (tA * tB).sum() / nA
            rh = m.cross_response(thA, thB, sep)[0]
            maxd = max(maxd, abs(r2d - rh))
            print(f"{thA:5.1f} {thB:5.1f} {sep:5.1f} {rh:11.5f} {r2d:11.5f} {r2d-rh:+9.1e}")
print(f"max |rho_2D - rho_Hankel| = {maxd:.2e}")
