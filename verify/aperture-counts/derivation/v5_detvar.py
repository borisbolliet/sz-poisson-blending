"""Sky-to-sky variance of detected counts with the M>1e13 floor (Poisson thinning => var/mean must be 1)."""
import numpy as np, sys
from szsim.sampler import draw_catalog
NS = int(sys.argv[1]); out = []
for s in range(NS):
    q = draw_catalog(950001+s, Mmin=1e13, fields=("q_obs",))['q_obs']
    out.append([(q > c).sum() for c in (4, 5, 6, 7)])
    print(s, out[-1], flush=True)
a = np.array(out, float); NB = np.array([132126.5, 85913.4, 61630.3, 46575.1])
for k, c in enumerate((4, 5, 6, 7)):
    v = a[:, k]; e = v.std(ddof=1)/np.sqrt(NS)
    print(f"det{c}: mean {v.mean():.1f}+-{e:.1f} analytic {NB[k]:.1f} z={(v.mean()-NB[k])/e:+.2f}  var/mean {v.var(ddof=1)/v.mean():.3f}+-{np.sqrt(2/(NS-1)):.3f}")
