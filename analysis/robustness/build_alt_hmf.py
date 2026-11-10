"""Build HMF tables for alternative colossus models at 500c (bocquet16, despali16) on the default grid."""
import time, numpy as np
from szsim import cosmo_hmf as ch
T0 = ch.default_table()
print("tinker08 N(>1e13), N(>1e14), N(>3e14):", T0.N_above(1e13), T0.N_above(1e14), T0.N_above(3e14))
for model in ["bocquet16", "despali16"]:
    t = time.time()
    T = ch.HMFTable(model=model)
    print(f"{model}: built in {time.time()-t:.1f}s; N(>1e13)={T.N_above(1e13):.6g} N(>1e14)={T.N_above(1e14):.6g} N(>3e14)={T.N_above(3e14):.6g}")
    for M in [1e14, 3e14, 1e15]:
        for z in [0.1, 0.5, 1.0, 2.0]:
            r = ch.dndlnM_Mpc3(np.array([M]), z, model=model)[0] / ch.dndlnM_Mpc3(np.array([M]), z)[0]
            print(f"   ratio {model}/tinker08 dn/dlnM at M={M:.0e}, z={z}: {r:.3f}")
