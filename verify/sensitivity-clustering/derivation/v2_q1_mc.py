"""V2: independent full-sky MC of Q1 (own sampler, own q). Floor M>1e14. P(>=k other detections within theta | detection)
vs Poisson CCDF with own analytic Sigma_det. Configs: base, beam 5, planck-like, noise x0.5."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import poisson, chi2
import own_model as om
from own_sampler import Sampler

NSKY = int(sys.argv[1]) if len(sys.argv) > 1 else 20
S = Sampler(Mmin=1e14, nz=900, nM=300)
print(f"own sampler N(M>1e14) expected = {S.Ntot:.1f}")
ynP = 33 / (2.7255e6 * 1.0404)
CF = {"base": (1.4, 4e-6), "noise_x0.5": (1.4, 2e-6), "beam_5": (5.0, 4e-6), "planck_like": (7.3, 1.1638e-5)}
TH = [1.0, 2.0, 5.0, 10.0, 30.0]
zs = []
thg = np.geomspace(0.01, 300, 600)
for ic, (name, (beam, noise)) in enumerate(CF.items()):
    sYg = om.sigmaY(thg, beam, noise)
    # analytic N_det on the sampler's own cells (same integrand)
    zc = 0.5 * (S.ze[1:] + S.ze[:-1]); Mc = np.exp(0.5 * (S.le[1:] + S.le[:-1]))
    Pd = om.pdet_grid(zc, Mc, beam, noise, 0.173, 5.0)
    Nan = (S.mu_full * Pd).sum(); Sig = Nan / (4 * np.pi)
    rng = np.random.default_rng(777 + ic)
    Nd = []; num = {(t, k): 0 for t in TH for k in (1, 2, 3)}; per = {(t, k): [] for t in TH for k in (1, 2, 3)}
    for s in range(NSKY):
        z, M = S.draw(rng, 1.0)
        Y = om.Ymed(M, z) * np.exp(0.173 * rng.standard_normal(len(z)))
        th = om.theta500(M, z)
        q = Y / np.exp(np.interp(np.log(th), np.log(thg), np.log(sYg)))
        det = q + rng.standard_normal(len(z)) > 5.0
        nd = det.sum(); Nd.append(nd)
        u = rng.random(nd); v = rng.random(nd)
        ph = 2 * np.pi * u; ct = 2 * v - 1; st = np.sqrt(1 - ct * ct)
        xyz = np.column_stack([st * np.cos(ph), st * np.sin(ph), ct])
        tr = cKDTree(xyz)
        for t in TH:
            nn = tr.query_ball_point(xyz, 2 * np.sin(t * om.ARC / 2), return_length=True) - 1
            for k in (1, 2, 3):
                per[(t, k)].append(((nn >= k).sum(), nd))
    Nd = np.array(Nd)
    print(f"{name:12s} N_det MC {Nd.mean():.1f} +- {np.sqrt(Nan/NSKY):.1f} vs analytic {Nan:.1f} ({(Nd.mean()-Nan)/np.sqrt(Nan/NSKY):+.2f} sig); Sigma = {Sig*om.ARC**2*3600:.4f}/deg2")
    for t in TH:
        lam = Sig * 2 * np.pi * (1 - np.cos(t * om.ARC))
        for k in (1, 2, 3):
            a = np.array(per[(t, k)], float)
            if a[:, 0].sum() < 10: continue
            r = a[:, 0].sum() / a[:, 1].sum()
            # error: sky-to-sky scatter of per-sky ratios
            # error: each k-tuple is counted (k+1) times -> var ~ (k+1) * count (Poisson clusters)
            e = np.sqrt((k + 1) * a[:, 0].sum()) / a[:, 1].sum()
            pa = poisson.sf(k - 1, lam); zz = (r - pa) / e; zs.append(zz)
            print(f"   theta={t:4.0f}' k>={k}: MC {r:.4e} +- {e:.1e}  Poisson {pa:.4e} ({zz:+.2f} sig)  Sigma*pi*theta^2 = {Sig*np.pi*(t*om.ARC)**2:.4e}")
zs = np.array(zs)
print(f"ALL: {len(zs)} tests, chi2 = {np.sum(zs**2):.1f}, p = {chi2.sf(np.sum(zs**2), len(zs)):.3f}, max|z| = {np.abs(zs).max():.2f}")
