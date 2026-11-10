"""REPAIR (C7): filtered-map variance of the [1e13,1e14) halo component (A+B) over S1/S2/ALL patches vs the analytic
Poisson one-halo variance stored by analysis_noise.py (noise_confusion.json, an_AB)."""
import json, numpy as np
D = "/Users/boris/GitHub/sz-poisson-blending/data/q2_maps"
an = np.array(json.load(open(f"{D}/noise_confusion.json"))["confusion"]["an_AB"])
V = []
for p in range(324):
    with np.load(f"{D}/patch_out/patch_{p:03d}.npz") as f:
        V.append(f["st_var_AB"] / f["sig"] ** 2)
V = np.array(V)
for name, P in [("S1", slice(0, 108)), ("S2", slice(108, 324)), ("ALL", slice(0, 324))]:
    m = V[P].mean(0); e = V[P].std(0, ddof=1) / np.sqrt(V[P].shape[0])
    print(f"{name:3s}: " + "  ".join(f"k={k}: {m[k]:.4f}+-{e[k]:.4f} vs {an[k]:.4f} ({(m[k]-an[k])/e[k]:+.2f}s, ratio {m[k]/an[k]:.4f})" for k in (0, 5, 11)))
