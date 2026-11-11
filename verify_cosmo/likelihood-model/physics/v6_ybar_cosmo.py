"""V6: cosmology dependence of the map-mean shift: ybar500 (M>1e13, 0.01<z<3) at the fiducial, sigma8+0.03, Om0+0.02
(my HMF integral); mu scales linearly with ybar (DC identity checked in v3)."""
import numpy as np, vlib as v
AREA = 4 * np.pi / v.ARC ** 2
nd = v.nodes(nzper=6, nsub=6, nM=200, ngh=24)
for lab, kw in (("fid", {}), ("s8+0.03", dict(sigma8=0.8402)), ("Om+0.02", dict(Om0=0.3311))):
    p = v.population(nd=nd, **kw)
    print(f"{lab:8s} ybar500 = {np.sum(p['w']*p['Y'])/AREA:.4e}")
