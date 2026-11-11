"""Validate replib.naive_exact: speed, convergence, comparison to cosmolib binned counts, smoothness."""
import os, sys, time
os.environ["NUMBA_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import replib as R

fid = np.array(R.FIDP)
t = time.time(); N = R.naive_exact(fid); t1 = time.time() - t
t = time.time(); N = R.naive_exact(fid); t2 = time.time() - t
print(f"[timing] naive_exact: first {t1:.2f}s, repeat {t2:.2f}s")
Nhi = R.naive_exact(fid, ngl=24, nM=1000, ngh=48)
m = Nhi > 1
print(f"[convergence] (12,500,32) vs (24,1000,48): total rel {N.sum()/Nhi.sum()-1:+.2e}, max |rel| bins N>1 {np.abs(N/Nhi-1)[m].max():.2e}")
Ncl = R.G["Nzq_n"][[tuple(np.round(q, 6)) for q in R.G["params"]].index(tuple(np.round(R.FIDP, 6)))]
r = Ncl / Nhi
print(f"[cosmolib binning] N_tot cosmolib {Ncl.sum():.1f} exact {Nhi.sum():.1f} rel {Ncl.sum()/Nhi.sum()-1:+.2e}")
print("  per-bin cosmolib/exact - 1 (rows z, cols q):")
print(np.array2string(np.round(r - 1, 4), max_line_width=200))
print(f"  max |rel| over bins N>20: {np.abs(r-1)[Nhi>20].max():.4f}; over q<50 columns: {np.abs(r-1)[:, :-1][Nhi[:, :-1]>20].max():.4f}")
chi_bin = ((Ncl - Nhi) ** 2 / Nhi)[Nhi > 1e-3].sum()
print(f"  chi2 of cosmolib-vs-exact naive counts, full sky: {chi_bin:.1f}  (ratio C_fid unchanged by construction)")
# smoothness: deviance along sigma8 and 1-b lines, data = C_fid * N_exact(fid)
Dd = R.C_FID * N
for a, hs in ((0, 2e-4), (2, 8e-4)):
    xs = np.arange(-6, 7) * hs
    cs = []
    for dx in xs:
        th = fid.copy(); th[a] += dx; cs.append(R.deviance(Dd, R.naive(th), 1.0))
    cs = np.array(cs)
    p = np.polyfit(xs, cs, 4); res = cs - np.polyval(p, xs)
    print(f"[smoothness] deviance along {R.PN[a]} (13 pts, step {hs}): range {cs.min():.3f}-{cs.max():.3f}, rms resid to quartic {res.std():.2e}")
# Jacobian step sensitivity
for a in range(4):
    out = []
    for f in (0.5, 1.0, 2.0):
        tp, tm = fid.copy(), fid.copy(); tp[a] += f * R.H_STEP[a]; tm[a] -= f * R.H_STEP[a]
        out.append((R.naive_exact(tp) - R.naive_exact(tm)) / (2 * f * R.H_STEP[a]))
    print(f"[jacobian] {R.PN[a]}: max rel diff step x0.5 vs x1 {np.abs(out[0]/out[1]-1)[N>20].max():.2e}, x2 vs x1 {np.abs(out[2]/out[1]-1)[N>20].max():.2e}")
# smooth HMF vs colossus
import cosmolib as cl
from colossus.cosmology import cosmology
c = R.FastCosmo(0.81, 0.31); Rr = np.array([0.5, 2.0, 8.0, 15.0, 25.0])
s_my = np.sqrt(c._sig2(Rr)[0] * c._norm); cosmology.setCurrent(c.c); s_col = c.c.sigma(Rr, 0.0)
print("[hmf] sigma(R) mine/colossus - 1 at R =", Rr, "Mpc/h:", np.round(s_my / s_col - 1, 6))
z = np.array([0.05, 0.5, 1.0, 2.0]); lnM = np.log(np.array([1e13, 1e14, 5e14, 1e15, 3e15]))
print("[hmf] d2N mine/colossus - 1 (rows z=0.05,0.5,1,2; cols M=1e13,1e14,5e14,1e15,3e15):")
print(np.round(c.d2N(z, lnM) / cl.Cosmo(0.81, 0.31).d2N(z, lnM) - 1, 4))
Oms = 0.31 + np.arange(-6, 7) * 2e-4
v1 = np.array([np.log(cl.Cosmo(0.81, Om).d2N(z, lnM)).ravel() for Om in Oms])
v2 = np.array([np.log(R.FastCosmo(0.81, Om).d2N(z, lnM)).ravel() for Om in Oms])
for v, nm in ((v1, "colossus"), (v2, "replib  ")):
    print(f"[hmf smoothness in Om, 13 pts step 2e-4] {nm}: max rms resid of ln d2N to cubic {max((v[:,k]-np.polyval(np.polyfit(Oms,v[:,k],3),Oms)).std() for k in range(v.shape[1])):.2e}")
