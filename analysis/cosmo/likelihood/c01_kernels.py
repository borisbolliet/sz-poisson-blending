"""Blended kernel table G[i,k,t] at the fiducial cosmology from the b03 rate histograms
(data/blending/analytic.npz: neighbours M>1e13, full z resolution).  Plus numerical checks:
 (a) zero-rate G reproduces Phi(t) (pure noise) -> interpolation error of the counts machinery
 (b) P_dom against b03, mean of y against -0 (map mean removed: <delta> - mu = 0 for the untruncated sum)
Timing of each step is printed (cost per cosmology)."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
import time
import numpy as np
from scipy.special import ndtr
import cclib as cc

A = np.load(f"{cc.ROOT}/data/blending/analytic.npz")
R = dict(W=A["W"], WX=A["WX"], WX2=A["WX2"], CAP=A["CAP"], CAPX=A["CAPX"], mu=A["mu_exact"].sum(1), th=A["th"])
assert np.allclose(A["QNODES"], cc.QNODES)
print("mu(theta_A) [SNR units, map-mean shift]:", np.round(R["mu"][::4], 3), "at theta", np.round(R["th"][::4], 2))
t0 = time.time()
G, Pdom = cc.build_G(R)
tG = time.time() - t0
print(f"build_G (40 theta x 259 q nodes, FFT 2^17): {tG:.1f}s")
print(f"P_dom vs b03 (floor 1e13): max |diff| = {np.abs(Pdom - A['Pdom'][0]).max():.2e}")
# untruncated mean check at large q (x_cut = 500): E[y] = E[n] + E[delta_tr] - mu ~ -(mean of x>=500 part)
tg = cc.TGRID
for i in (5, 15, 25, 35):
    k = len(cc.QNODES) - 1
    F = G[i, k] / Pdom[i, k]
    pdf = np.diff(F); tm = 0.5 * (tg[1:] + tg[:-1])
    print(f"   theta={R['th'][i]:6.2f}': q_A=3000 node: <t> = {np.sum(pdf*tm):+.4f} (expect ~0), sd = {np.sqrt(np.sum(pdf*tm**2)-np.sum(pdf*tm)**2):.3f},"
          f" P_dom(q=5) = {Pdom[i, 49]:.4f}, P_dom(q=1) = {Pdom[i, 9]:.4f}")
np.savez(f"{cc.OUT}/G_fid_b03rates.npz", G=G, Pdom=Pdom, mu=R["mu"], th=R["th"], qnodes=cc.QNODES, tgrid=cc.TGRID)

# (a) numerical check: zero rates
Z = dict(W=np.zeros_like(A["W"][:2]), WX=np.zeros_like(A["W"][:2]), WX2=np.zeros_like(A["W"][:2]),
         CAP=np.zeros((2, 3)), CAPX=np.zeros((2, 3)), mu=np.zeros(2))
G0, _ = cc.build_G(Z, verbose=False)
print(f"zero-rate G vs Phi(t): max |diff| = {np.abs(G0[0] - ndtr(cc.TGRID)[None, :]).max():.2e}")
G0 = np.repeat(G0[:1], 40, axis=0)
pop = cc.population(cc.FID, zedges=np.array([0.01, 3.0]))
E = np.array([4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 100, np.inf])
t1 = time.time()
Ng = cc.counts_blended(G0, pop, E, 1)[0]
tc = time.time() - t1
Na = cc.counts_gauss(pop, E, 1)[0]
print(f"counts via G-interpolation (zero rate) vs exact Phi, per q bin (rel): {np.array2string(Ng/Na-1, precision=2)}")
print(f"counts_blended time for {pop['q'].size:,} points x {len(E)} edges: {tc:.2f}s")
