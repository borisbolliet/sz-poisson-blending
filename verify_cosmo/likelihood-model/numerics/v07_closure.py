"""Re-run of the c01 numerical closure (zero-rate kernel vs Phi; G-interpolated vs exact naive counts) without
overwriting data_cosmo, plus a pure-Gaussian kernel test with a NON-zero shift and width (y = N(-0.3, 1.5^2)):
G built by build_G from a single small-x rate (Gaussian path) vs the exact Gaussian counts."""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.special import ndtr
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/likelihood")
import cclib as cc
A = np.load(f"{cc.ROOT}/data/blending/analytic.npz")
Z = dict(W=np.zeros_like(A["W"][:2]), WX=np.zeros_like(A["W"][:2]), WX2=np.zeros_like(A["W"][:2]),
         CAP=np.zeros((2, 3)), CAPX=np.zeros((2, 3)), mu=np.zeros(2))
G0, _ = cc.build_G(Z, verbose=False)
print(f"zero-rate G vs Phi(t): max |diff| = {np.abs(G0[0] - ndtr(cc.TGRID)[None, :]).max():.2e}")
G0 = np.repeat(G0[:1], 40, axis=0)
pop = cc.population(cc.FID, zedges=np.array([0.01, 3.0]))
E = np.array([4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 100, np.inf])
Ng = cc.counts_blended(G0, pop, E, 1)[0]; Na = cc.counts_gauss(pop, E, 1)[0]
print(f"G-interp (zero rate) vs exact, rel per q bin: {np.array2string(Ng/Na-1, precision=2)}")
# Gaussian kernel with shift and extra variance via the 'small-x' (Gaussian) branch: one tiny-x bin with
# mean mu_s=0.2 and var v_s=1.25 -> y ~ N(0.2 - mu, 1 + 1.25); set mu = 0.5 -> N(-0.3, 2.25)
W = np.zeros_like(A["W"][:2]); WX = np.zeros_like(W); WX2 = np.zeros_like(W)
W[:, 0, 1] = 1e6; WX[:, 0, 1] = 0.2; WX2[:, 0, 1] = 1.25
Gg, Pd = cc.build_G(dict(W=W, WX=WX, WX2=WX2, CAP=np.zeros((2, 3)), CAPX=np.zeros((2, 3)), mu=np.full(2, 0.5)), verbose=False)
print(f"Gaussian-branch G vs Phi((t+0.3)/1.5): max |diff| = {np.abs(Gg[0] - ndtr((cc.TGRID + 0.3) / 1.5)[None, :]).max():.2e}; P_dom {Pd.min():.6f}")
Gg = np.repeat(Gg[:1], 40, axis=0)
Nb = cc.counts_blended(Gg, pop, E, 1)[0]; Ne = cc.counts_gauss(pop, E, 1, shift=-0.3, var=2.25)[0]
print(f"G-interp (shift -0.3, var 2.25) vs exact, rel per q bin: {np.array2string(Nb/Ne-1, precision=2)}")
