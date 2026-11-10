"""V2b: N_det(q>5, M>1e14) of own sampler, many skies (no positions), base and planck-like."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import own_model as om
from own_sampler import Sampler
S = Sampler(Mmin=1e14, nz=900, nM=300)
thg = np.geomspace(0.01, 300, 600)
zc = 0.5 * (S.ze[1:] + S.ze[:-1]); Mc = np.exp(0.5 * (S.le[1:] + S.le[:-1]))
for name, beam, noise, seed in [("base", 1.4, 4e-6, 31), ("planck_like", 7.3, 1.1638e-5, 32)]:
    sYg = om.sigmaY(thg, beam, noise)
    Nan = (S.mu_full * om.pdet_grid(zc, Mc, beam, noise, 0.173, 5.0)).sum()
    rng = np.random.default_rng(seed); Nd = []
    for s in range(150):
        z, M = S.draw(rng, 1.0)
        q = om.Ymed(M, z) * np.exp(0.173 * rng.standard_normal(len(z))) / np.exp(np.interp(np.log(om.theta500(M, z)), np.log(thg), np.log(sYg)))
        Nd.append((q + rng.standard_normal(len(z)) > 5).sum())
    Nd = np.array(Nd)
    print(f"{name}: 150 skies N_det MC {Nd.mean():.1f} +- {np.sqrt(Nan/150):.1f} vs analytic {Nan:.1f} ({(Nd.mean()-Nan)/np.sqrt(Nan/150):+.2f} sig); var/mean {Nd.var(ddof=1)/Nd.mean():.2f} +- {np.sqrt(2/149):.2f}")
