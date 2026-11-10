"""Check of the M>1e13 N_det(q>5) offsets seen in the shared-seed passes (planck_like -2.77 sigma, base +2.07):
fresh szsim draws (new seeds) for planck-like and base, and the same seeds 2001-2005 to confirm the pass numbers."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/robustness")
import configs as C
from szsim import sampler, signal as sg
t0 = time.time()
for name, beam, noise in (("planck_like", 7.3, C.PLANCK_NOISE_Y), ("base", 1.4, 4e-6)):
    m = sg.SZModel(beam_fwhm_arcmin=beam, noise_y_arcmin=noise)
    Nan = m.expected_detections(5.0); Nan_lo = m.expected_detections(5.0, Mmax=1e14)
    for label, seeds in (("seeds 2001-2005 (as passes)", range(2001, 2006)), ("fresh seeds 7001-7010", range(7001, 7011))):
        nd, nlo = [], []
        for s in seeds:
            c = sampler.draw_catalog(s, Mmin=1e13, model=m, fields=("M500", "q_obs"))
            d = c["q_obs"] > 5; nd.append(d.sum()); nlo.append((d & (c["M500"] < 1e14)).sum()); del c
        nd = np.array(nd); nlo = np.array(nlo); n = len(nd)
        print(f"{name:12s} {label}: N_det {nd.mean():.1f} vs {Nan:.1f} ({(nd.mean()-Nan)/np.sqrt(Nan/n):+.2f} sigma); "
              f"from M<1e14: {nlo.mean():.2f} vs {Nan_lo:.2f} ({(nlo.mean()-Nan_lo)/np.sqrt(Nan_lo/n):+.2f} sigma)  [{time.time()-t0:.0f}s]", flush=True)
