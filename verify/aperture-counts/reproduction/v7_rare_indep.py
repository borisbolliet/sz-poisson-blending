"""V7: independent check of C11 (P(at least one close pair / triple anywhere on the sky) ~ 1-exp(-E)).
Pure Poisson skies with my own Nbar (v1) for M500>3e14 (N ~ Poisson(Nbar), isotropic positions from normalised
Gaussians), brute-force sorted-window pair and triangle counting.  Seeds 60000+i."""
import sys, os, json, time
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_common import SortedPts, counts, triangles

H = os.path.dirname(os.path.abspath(__file__))
NB = json.load(open(os.path.join(H, "nbar_indep.json")))["M3e14"]
AM = np.pi / 180 / 60
PT = np.array([0.1, 0.3, 0.5, 1.0]); TT = np.array([3, 5, 7, 10.0])
NSKY = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
t0 = time.time()
pc = np.zeros((NSKY, len(PT))); tc = np.zeros((NSKY, len(TT)))
for i in range(NSKY):
    rng = np.random.default_rng(60000 + i)
    n = rng.poisson(NB)
    g = rng.standard_normal((n, 3)); v = g / np.linalg.norm(g, axis=1)[:, None]
    S = SortedPts(v, np.arange(n))
    pc[i] = counts(v, np.arange(n), S, PT).sum(0) / 2
    tc[i] = [triangles(S, t) for t in TT]
print(f"{NSKY} skies, Nbar(M>3e14) = {NB:.2f} (mine), {time.time()-t0:.0f}s")
P3 = 1 - 3 * np.sqrt(3) / (4 * np.pi)
zs = []
for lab, th, C, kind in [("pairs", PT, pc, 2), ("triples", TT, tc, 3)]:
    for j, t in enumerate(th):
        mu = NB * np.sin(t * AM / 2) ** 2
        E = NB * mu / 2 if kind == 2 else NB * mu ** 2 * P3 / 6
        pm = np.mean(C[:, j] > 0); pe = np.sqrt(max(pm * (1 - pm), 1 / NSKY) / NSKY); pa = 1 - np.exp(-E)
        em = C[:, j].std(ddof=1) / np.sqrt(NSKY)
        zs.append((pm - pa) / pe)
        print(f"{lab:7s} {t:5.2f}': E={E:.4f} MC mean={C[:,j].mean():.4f}+-{em:.4f} (z={(C[:,j].mean()-E)/em:+.2f})"
              f"  P(exists) 1-exp(-E)={pa:.4f} MC={pm:.4f}+-{pe:.4f} z={(pm-pa)/pe:+.2f}")
zs = np.array(zs)
print(f"P(exists) z: rms={np.sqrt(np.mean(zs**2)):.2f} max|z|={np.abs(zs).max():.2f}")
