"""c01: checks of the verifier's own numerics (vc_model.py) against independent references.

(1) Arnaud shape: g(0) = Y(5R500)/Y500 (two different integrals); the central value f(0) and the projected shape f(s)
    against step 1's stored table data/template_hankel_arnaud_5R500.npz (sn, fn: step 1's projected unit-Y500 profile)
    and g(u) against its stored Hankel table (u, g).  Read only, as a cross-check of OUR integrals.
(2) tSZ SED (own formula) and SO table values against the build's data_real/step3_szifi_real/instrument.npz and
    against SZiFi's SOgoal_simple values (typed here from szifi/expt.py, read not imported).
(3) Painting: one cluster painted with vc_model.paint_tsz_rfft; its real-space integral and its central value
    against Y500 * Y5/Y500 and y0 = Y500 f(0) / theta500^2 (beam- and band-limited, so the centre is checked on a
    large cluster only).
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B c01_model_check.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                   # noqa: E402
import vc_model as M                                 # noqa: E402

t0 = time.time()
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
print("(1) Arnaud+10 shape, truncated at 5 R500, unit spherical Y500")
print(f"  Y(5R500)/Y500 = {M.RATIO5:.10f};  g(0) by projection-slice quadrature = {M.ghat(0.0)[0]:.10f}  "
      f"(rel. diff {M.ghat(0.0)[0] / M.RATIO5 - 1:+.1e})")
print(f"  f(0) = {M.F_CENTRE:.10f};  f(1e-8) = {M.F_NORM:.10f}  (f(1e-8)/f(0) - 1 = {M.F_NORM / M.F_CENTRE - 1:+.2e})")
st = np.load(os.path.join(ROOT, "data", "template_hankel_arnaud_5R500.npz"))
print(f"  step 1's table: ratio {float(st['ratio']):.10f} (ours/step1 - 1 = {M.RATIO5 / float(st['ratio']) - 1:+.1e}); "
      f"fn(0) = {st['fn'][0]:.10f} (ours/step1 - 1 = {M.F_CENTRE / st['fn'][0] - 1:+.1e})")
ss = np.array([0.01, 0.1, 0.3, 1.0, 2.0, 4.0, 4.9])
ours = M.proj(ss)
theirs = np.interp(ss, st["sn"], st["fn"])
print("  f(s) at s = " + " ".join(f"{x:5.2f}" for x in ss))
print("    ours/step1 - 1:   " + " ".join(f"{x:+.1e}" for x in ours / theirs - 1))
uu = np.array([0.1, 1.0, 3.0, 10.0, 30.0, 60.0])
gs = np.interp(uu, st["u"], st["g"])
go = M.ghat(uu)
print("  g(u) at u = " + " ".join(f"{x:6.1f}" for x in uu))
print("    ours:  " + " ".join(f"{x:+.5e}" for x in go))
print("    step1: " + " ".join(f"{x:+.5e}" for x in gs) + "   (step 1 interpolates linearly on its 700-point table)")

print("\n(2) instrument")
sed = M.tsz_sed_uK(M.FREQS_GHZ)
inst = np.load(os.path.join(ROOT, "data_real", "step3_szifi_real", "instrument.npz"))
print("  our SED [uK/y]:          " + " ".join(f"{x:13.6f}" for x in sed))
print("  build instrument.npz:    " + " ".join(f"{x:13.6f}" for x in inst["tsz_sed_uK"]))
print(f"    max |ours/build - 1| = {np.abs(sed / inst['tsz_sed_uK'] - 1).max():.1e}")
sys.path.insert(0, "/Users/boris/GitHub/szifi")
from szifi import expt                                # noqa: E402  (read SZiFi's SOgoal_simple values at run time)
sz_goal_sed = np.asarray(expt.experiment("SOgoal_simple").tsz_sed, float)
print(f"    max |ours/SZiFi SOgoal_simple - 1| = {np.abs(sed / sz_goal_sed - 1).max():.1e}")
for k, ours_v in (("fwhm_arcmin", M.FWHM), ("white_baseline", M.WHITE["baseline"]), ("white_goal", M.WHITE["goal"]),
                  ("freqs_GHz", M.FREQS_GHZ)):
    print(f"  {k:15s} ours {ours_v.tolist()}  build {inst[k].tolist()}  identical {np.array_equal(ours_v, inst[k])}")

print("\n(3) painting check: one cluster, theta500 = 5.676', Y500 = 1e-3 arcmin^2 at index (600.5, 400.5)")
th, Y = 5.676185956352383, 1e-3
X = M.paint_tsz_rfft(np.array([600.5]), np.array([400.5]), np.array([Y]), np.array([th]))
sed = M.tsz_sed_uK(M.FREQS_GHZ)
from scipy import fft as sfft                         # noqa: E402
m145 = sfft.irfft2(X[3], s=(M.N, M.N)) / sed[3]       # y map seen at 145 GHz (beam 1.4', band 100-20000)
tot = m145.sum() * M.OMEGA_PIX_ARCMIN2
print(f"  sum of the band-limited y map x pixel area = {tot:.6e} arcmin^2 (l = 0 removed, so ~0 expected: "
      f"{tot / (Y * M.RATIO5):+.1e} of Y5)")
X0 = X[3, 0, 0]
print(f"  rfft coefficient at k = 0 is zeroed by the band: {X0}")
# reconstruct the l = 0 limit from the k -> 0 coefficient extrapolation: X(k) * Omega / (sed B) -> Y500 * g(0)
m0, m1, n2 = M.rfft_grid()
k1 = X[3, 0, 3] / sed[3] * M.OMEGA_PIX_ARCMIN2 / np.exp(-2j * np.pi * 3 * 400.5 / M.N)
print(f"  |X(m=(0,3))| Omega / SED / phase = {k1.real:.6e} (Y500 g(108 theta) B = "
      f"{Y * M.ghat(108. * th * M.ARC)[0] * M.beam_paint(108., 1.4):.6e})")
mid = m145[598:604, 398:404]
print(f"  peak of the 145 GHz y map near the centre: {mid.max():.4e} at offsets "
      f"{np.unravel_index(mid.argmax(), mid.shape)} from (598, 398);  y0_true = Y500 f(0)/theta^2 = "
      f"{Y * M.F_CENTRE / th ** 2:.4e} (the map is beam-smoothed and has no l < 100, so lower)")
print(f"  symmetric about (600.5, 400.5): max |m[600-d] - m[601+d]| / max = "
      f"{np.abs(m145[600 - np.arange(50), 400] - m145[601 + np.arange(50), 400]).max() / m145.max():.1e}")
print(f"\ndone [{time.time() - t0:.0f} s]")

print("\n(4) g(u) by a second, independent route: 2 pi int_0^5 s f(s) J0(u s) ds with f(s) from vc_model.proj on 20001")
print("    Gauss-Legendre-panel nodes (s = 5 (1 - (1 - t)^2) clusters nodes at the sqrt edge)")
from scipy.special import j0                          # noqa: E402
xg, wg = np.polynomial.legendre.leggauss(20)
ed = np.linspace(0., 1., 401)
tt = (0.5 * (ed[1:] - ed[:-1])[:, None] * xg[None, :] + 0.5 * (ed[1:] + ed[:-1])[:, None]).ravel()
wt = (0.5 * (ed[1:] - ed[:-1])[:, None] * wg[None, :]).ravel()
s_n = 5. * (1. - (1. - tt) ** 2)
w_n = wt * 10. * (1. - tt)
f_n = M.proj(s_n)
for u in (0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 60.0):
    g2 = np.sum(w_n * 2 * np.pi * s_n * f_n * j0(u * s_n))
    print(f"  u = {u:5.1f}: projection-slice {M.ghat(u)[0]:+.8e}   J0 route {g2:+.8e}   rel. diff {M.ghat(u)[0] / g2 - 1:+.1e}")
print(f"done (4) [{time.time() - t0:.0f} s]")
