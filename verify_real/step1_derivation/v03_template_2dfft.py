"""v03: Fourier transform g(u) of the unit-Y500 Arnaud template (truncated at 5 R500) from DIRECT 2D FFTs of
pixelised template images (no Hankel / J0 quadrature).  Two images: theta500 = 32 px (fine in u, u <= 25) and
theta500 = 256 px (u up to 300); modes binned in |u| (all directions -> anisotropy measured).
Output g_of_u_2dfft.npz; checks: g(0) = RATIO, overlap of the two images, old szsim Hankel table (data only).
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vlib as V
V.set_threads(4)
import numpy as np

N = 4096


def image_ft(P):
    """Point-sampled template image, theta500 = P pixels, centre on pixel (0,0) with periodic coordinates.
    Returns |u| and Re(FT) for the half plane (rfft2 layout); FT normalised so that FT(0) = sum tau dA."""
    i = np.arange(N); x = np.where(i < N // 2, i, i - N).astype(float)
    t0 = time.time()
    img = np.empty((N, N))
    for r0 in range(0, N, 512):                       # rows in chunks (memory)
        r = np.hypot(x[r0:r0 + 512, None], x[None, :]) / P
        img[r0:r0 + 512] = V.tau_shape(r) / P ** 2       # tau per pixel^2 (theta500 = P px)
    F = np.fft.rfft2(img)
    del img
    kx = np.arange(N // 2 + 1); ky = np.where(i < N // 2, i, i - N)
    U = 2 * np.pi * P / N * np.hypot(ky[:, None], kx[None, :])
    print(f"   image theta500 = {P} px: built+FFT {time.time()-t0:.1f}s; sum(tau dA) = {F[0,0].real:.10f} "
          f"(RATIO {V.RATIO:.10f}, rel {F[0,0].real/V.RATIO-1:+.2e}); max |Im| = {np.abs(F.imag).max():.1e}", flush=True)
    return U.ravel(), F.real.ravel()


def radial(U, G, umax, du):
    sel = U <= umax
    U, G = U[sel], G[sel]
    b = np.floor(U / du + 0.5).astype(int)            # bins centred on multiples of du
    n = np.bincount(b); su = np.bincount(b, U); sg = np.bincount(b, G); sg2 = np.bincount(b, G * G)
    ok = n > 0
    um = su[ok] / n[ok]; gm = sg[ok] / n[ok]
    sd = np.sqrt(np.maximum(sg2[ok] / n[ok] - gm ** 2, 0))
    return um, gm, sd, n[ok]


print("=== template FT from 2D FFT images ===")
U1, G1 = image_ft(32)
u1, g1, s1, n1 = radial(U1, G1, 25.0, 2 * np.pi * 32 / N / 2)
del U1, G1
U2, G2 = image_ft(256)
u2, g2, s2, n2 = radial(U2, G2, 300.0, 2 * np.pi * 256 / N / 2)
del U2, G2
print(f"   anisotropy (rms over directions / g(0)): fine image u<=5 max {np.max(s1[u1<=5])/V.RATIO:.1e}, "
      f"u<=25 max {np.max(s1)/V.RATIO:.1e}; coarse image u<=100 max {np.max(s2[u2<=100])/V.RATIO:.1e}")
# overlap check
uo = np.linspace(8, 24, 9)
a = np.interp(uo, u1, g1); b = np.interp(uo, u2, g2)
print("   overlap u =", uo, "\n      fine/coarse - 1:", np.round(a / b - 1, 5), "\n      (fine - coarse)/g(0):", np.round((a - b) / V.RATIO, 7))
# combine: fine image up to u = 12, coarse image beyond
u = np.concatenate([u1[u1 <= 12.0], u2[u2 > 12.0]])
g = np.concatenate([g1[u1 <= 12.0], g2[u2 > 12.0]])
o = np.argsort(u); u, g = u[o], g[o]
# make strictly increasing u for the spline
keep = np.concatenate([[True], np.diff(u) > 1e-9]); u, g = u[keep], g[keep]
u[0] = 0.0
np.savez(os.path.join(HERE, "g_of_u_2dfft.npz"), u=u, g=g)
print(f"   saved g_of_u_2dfft.npz: {len(u)} nodes, u in [0, {u[-1]:.1f}], g(0) = {g[0]:.8f}, g(umax)/g(0) = {g[-1]/g[0]:.2e}")

print("\n=== comparison with the old szsim Hankel table (J0 quadrature; data/template_hankel_arnaud_5R500.npz) ===")
d = np.load(os.path.join(V.ROOT, "data", "template_hankel_arnaud_5R500.npz"))
uo, go = d["u"], d["g"]
V._GU = None
for uu in [0.0, 0.01, 0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
    print(f"   u = {uu:7.2f}: g_2dfft = {float(V.g_u(uu)):.6e}   g_hankel = {float(np.interp(uu, uo, go)):.6e}   "
          f"diff/g(0) = {(float(V.g_u(uu)) - float(np.interp(uu, uo, go)))/V.RATIO:+.2e}")
uu = np.linspace(0, 200, 4001)
dd = (V.g_u(uu) - np.interp(uu, uo, go)) / V.RATIO
print(f"   max |g_2dfft - g_hankel| / g(0) over u in [0, 200]: {np.max(np.abs(dd)):.2e}  (at u = {uu[np.argmax(np.abs(dd))]:.2f});"
      f" over u in [0, 20]: {np.max(np.abs(dd[uu<=20])):.2e}")
