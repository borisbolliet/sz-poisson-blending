"""V1: build independent tables; check RATIO, painted-profile normalisation, sigma_k (discrete and Hankel) against
the claimed sigma_Y values, R_k(0;theta_k)=1, and the l=0 bias ybar*kappa/sigma (C3, C4)."""
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
import indep_lib as I

t0 = time.time()
print(f"RATIO Y5R500/Y500 (own quad) = {I.RATIO:.6f}  (claimed 1.79644)")
# projected profile normalisation: int 2 pi s f ds over [0,5] must equal RATIO
s = np.linspace(0, 5, 2_000_001); print(f"int 2pi s f(s) ds / RATIO = {np.trapezoid(2*np.pi*s*I.f_of(s), s)/I.RATIO:.6f}")
# int over R500 cylinder should exceed 1 (cylinder > sphere)
print(f"g(0)/RATIO = {I.g_of(0.0)/I.RATIO:.6f}")
tab, rmax = I.paint_table()
print(f"paint table built  t={time.time()-t0:.0f}s")
r = np.linspace(0, 1, I.NRP)
integ = np.array([np.trapezoid(2*np.pi*r*rmax[i]*tab[i]*rmax[i], r) for i in range(I.NT)]) / I.RATIO
print(f"beam-convolved painted profile integral / RATIO: min {integ.min():.6f} max {integ.max():.6f}  (theta {I.TNODES[0]:.3f}..{I.TNODES[-1]:.1f}')")
Tft, sig, Tsum = I.templates()
print(f"templates built t={time.time()-t0:.0f}s")
sh = I.sigma_Y_hankel(I.THK)
claimed = {0: 5.6306e-06, 5: 1.0624e-05, 11: 4.3083e-05}
for k in range(I.K):
    c = claimed.get(k)
    print(f"  k={k:2d} theta_k={I.THK[k]:6.3f}'  sigma discrete {sig[k]:.5e}  sigma own-Hankel {sh[k]:.5e}  ratio-1 {sig[k]/sh[k]-1:+.2e}"
          + (f"   claimed {c:.4e}  (mine/claimed-1 = {sig[k]/c-1:+.2e})" if c else "")
          + f"   sum(T)*pix^2/RATIO = {Tsum[k]/I.RATIO:.6f}")
# library comparison values: sigma_Y(1')=7.23e-6, (3')=1.46e-5, (10')=4.31e-5 and point source 4.693e-6
print("own Hankel sigma_Y(1', 3', 10') =", I.sigma_Y_hankel(np.array([1.0, 3.0, 10.0])), " point-source:", I.sigma_Y_hankel(np.array([1e-4]))[0])
R, rR = I.resp_table(sig)
print(f"response table built t={time.time()-t0:.0f}s")
for k in [0, 5, 11]:
    v = I.tab_eval(R[k], rR, np.log(I.THK[k]), 0.0)
    print(f"  R_k(0; theta_k) k={k}: {v:.6f}")
# l=0 bias with the claimed ybar of patch 0
ybar = 6.0601e-07
kap = sig ** 2 * I.RATIO / I.NW
print("ybar*kappa_k/sigma_k for ybar=6.0601e-7 (claimed 0.383 ... 2.931):", np.array2string(ybar * kap / sig, precision=3))
# discrete kappa: Y_hat response (no l=0 removal) to uniform y=1 = sum(T)/sum(T^2) = Tsum/pix^2 * sig^2/sigpix^2
kap_d = Tsum / I.PIX ** 2 * sig ** 2 / I.SIGPIX ** 2
print("discrete kappa / analytic kappa - 1:", np.array2string(kap_d / kap - 1, precision=2))
