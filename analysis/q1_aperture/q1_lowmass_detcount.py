"""Repair phase: is the expected detected count Nbar(det_q) right for the full (M>1e13) population?
The floor-1e14 part is checked by q1_palm_tension.py (1000 skies).  Here we draw only 1e13<=M<1e14 halos on new
seeds (sampler with Mmax=1e14) and count q_obs>q, compared with SZModel.expected_detections(q, Mmax=1e14).
usage: python q1_lowmass_detcount.py seed0 seed1"""
import os, sys, json, time
import numpy as np
from szsim import sampler, signal as sg, cosmo_hmf as ch
from q1_common import *
T = ch.default_table(); m = sg.SZModel()
E = {q: m.expected_detections(q, T, Mmax=1e14) for q in (4, 5, 6)}
NB = json.load(open(os.path.join(DATA, "nbar.json")))
for q in (4, 5, 6):
    print(f"q>{q}: expected low-mass (M<1e14) detections {E[q]:.2f}; check det{q} - det{q}_floor1e14 = {NB[f'det{q}']-NB[f'det{q}_floor1e14']:.2f}")
rows = []
for s in range(int(sys.argv[1]), int(sys.argv[2])):
    t0 = time.time()
    cat = sampler.draw_catalog(s, Mmin=1e13, Mmax=1e14, fields=("q_obs",))
    rows.append([(cat["q_obs"] > q).sum() for q in (4, 5, 6)])
    print(f"seed {s}: N(M<1e14)={len(cat['q_obs'])} det4/5/6 = {rows[-1]}  t={time.time()-t0:.1f}s", flush=True)
rows = np.array(rows, float); R = len(rows)
for i, q in enumerate((4, 5, 6)):
    x = rows[:, i]
    print(f"q>{q}: MC mean {x.mean():.2f} +- {np.sqrt(E[q]/R):.2f} vs expected {E[q]:.2f}: rel {x.mean()/E[q]-1:+.4f}, z={(x.mean()-E[q])/np.sqrt(E[q]/R):+.2f};"
          f" var/mean={x.var(ddof=1)/E[q]:.2f}")
