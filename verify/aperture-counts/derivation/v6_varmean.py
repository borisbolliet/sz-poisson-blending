"""C7: mean/mu and var/mean of random-aperture counts from the 60 independent skies (histograms truncated at >=6,
so only populations with P(N>=6) < 1e-4 are used)."""
import numpy as np
from scipy import stats
r = dict(np.load('v2_combined.npz')); ARC = np.pi/180/60
NB = dict(det5=85913.4, M1e14=396387.8, M3e13=4598099.4)
n = np.arange(7)
for pop in ['det5', 'M1e14', 'M3e13']:
    for t in [1., 2., 5., 10.]:
        mu = NB[pop]*np.sin(t*ARC/2)**2
        if stats.poisson.sf(5, mu) > 1e-4: continue
        h = r[f'rand|{pop}|{t}']  # (nsky,7)
        m = (h*n).sum(1); v = (h*n**2).sum(1) - m**2
        vm = v/m
        print(f"{pop} {t}': mu={mu:.4f} mean/mu={m.mean()/mu:.4f}+-{m.std(ddof=1)/np.sqrt(len(m))/mu:.4f}  var/mean={vm.mean():.4f}+-{vm.std(ddof=1)/np.sqrt(len(vm)):.4f} (z={(vm.mean()-1)/(vm.std(ddof=1)/np.sqrt(len(vm))):+.2f})")
