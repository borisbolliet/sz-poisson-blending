"""V07: compare the bright/large-halo population: my model grid vs analysis catalogue sky 1."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import indep_model as im
from szsim import sampler
ze, lme, dens = im.hmf_grid(); cc = im.cell_counts(ze, lme, dens)
zc = 0.5 * (ze[1:] + ze[:-1]); Mc = np.exp(0.5 * (lme[1:] + lme[:-1])); Z, MM = np.meshgrid(zc, Mc, indexing="ij")
TH = im.theta500(MM, Z); Q = im.Ymed(MM, Z) / im.sigmaY_fast(TH)
AREA = 4 * np.pi / im.ARCMIN ** 2
cat = sampler.load_catalog("/Users/boris/GitHub/sz-poisson-blending/data/catalogs/cat_M1e13_seed1.npz", fields=["theta500", "q", "z", "M500"])
for t in (5, 10, 20, 40):
    print(f"N(theta500>{t}'): mine expected {cc[TH>t].sum()*AREA:9.1f}   sky1 {np.sum(cat['theta500']>t)}")
for qq in (10, 50, 200):
    # median-Y q; scatter ignored here for mine (approx), so compare also q_med in catalogue is not stored -> just report
    print(f"N(q>{qq}) sky1 {np.sum(cat['q']>qq)}")
print("N(z<0.05) mine", cc[Z < 0.05].sum() * AREA, "sky1", np.sum(cat["z"] < 0.05))
print("N(z<0.05, M>1e14) mine", cc[(Z < 0.05) & (MM > 1e14)].sum() * AREA, "sky1", np.sum((cat["z"] < 0.05) & (cat["M500"] > 1e14)))
