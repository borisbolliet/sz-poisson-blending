"""[added after verification] MC check of the detections that come from halos with true q < 0.5 (which
run_pass.py does not follow): count (q < 0.5) & (q_obs > 5) in full-sky M>1e13 draws and compare with
blendlib.ndet_lowq. Usage: python check_lowq_det.py"""
import os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blendlib as bl, configs as C
from szsim import sampler, signal as sg, cosmo_hmf as ch

T = ch.default_table()
for name, beam, noise, seeds in (("base", 1.4, 4e-6, [3101, 3102, 3103]), ("planck_like", 7.3, C.PLANCK_NOISE_Y, [3201, 3202, 3203])):
    m = sg.SZModel(beam_fwhm_arcmin=beam, noise_y_arcmin=noise)
    an = bl.ndet_lowq(m, [5.0], T, 3.0)[0]
    an_full = bl.ndet_lowq(m, [5.0], T, 3.0, zs=1, ms=1)[0]
    cnt = []
    for s in seeds:
        cat = sampler.draw_catalog(s, Mmin=1e13, model=m, table=T)
        cnt.append(int(np.sum((cat["q"] < 0.5) & (cat["q_obs"] > 5.0))))
        del cat
    k = np.sum(cnt)
    print(f"{name}: MC noise-only detections per sky {cnt} -> mean {k/len(seeds):.2f} ± {np.sqrt(k)/len(seeds):.2f}; "
          f"analytic N_lowq {an:.2f} (full-resolution grid {an_full:.2f}); "
          f"({(k/len(seeds)-an)/(np.sqrt(an/len(seeds))):+.2f}σ); total N_det an {m.expected_detections(5.0, table=T):.1f}", flush=True)
