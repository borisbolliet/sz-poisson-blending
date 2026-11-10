"""Repair phase: the low tail P(N<=1) of M>3e13 neighbours at 10' (mu=9.73), where the ready-to-use table shows
P(>=2) z = -2.2..-3.2 for detected centres in the 120-sky set.  Prints P(N=0), P(N=1), P(N<=1) per centre type and
seed set with sky-scatter errors and z vs Poisson (unconditional) and vs Bin(N_realised - [centre in pop], f)."""
import os, glob, json
import numpy as np
from scipy import stats
from q1_common import *
from q1_mc_fullsky import FINE
NB = json.load(open(os.path.join(DATA, "nbar.json")))
files = sorted(glob.glob(os.path.join(DATA, "fullsky", "sky_*.npy")))
S = [np.load(f, allow_pickle=True).item() for f in files]
seeds = np.array([s["seed"] for s in S])
for t in [7.5, 10, 15]:
    j = int(np.argmin(abs(FINE - t))); f = np.sin(t * ARCMIN / 2) ** 2
    mu = NB["M3e13"] * f
    an0, an1 = np.exp(-mu), mu * np.exp(-mu)
    print(f"theta={t}': mu={mu:.4f}  Poisson P(0)={an0:.4e} P(1)={an1:.4e} P(<=1)={an0+an1:.4e}")
    for key in ["rand|M3e13", "det4|M3e13", "det5|M3e13", "det6|M3e13", "det5_zc<0.5|M3e13", "det5_zc>1|M3e13", "det5_q>20|M3e13"]:
        for lab, m in [("orig 40", seeds <= 1040), ("new 80", seeds > 1040), ("all 120", seeds > 0)]:
            SS = [s for s, k in zip(S, m) if k]; R = len(SS)
            F = np.array([s["cdf"][key][:, j] for s in SS])
            nc = np.array([s["ncen"][key] for s in SS], float)
            le1 = 1 - F[:, 1]
            # events per sky
            ev = le1 * nc
            mc = ev.sum() / nc.sum()
            r = ev - (an0 + an1) * nc
            z = r.sum() / (r.std(ddof=1) * np.sqrt(R))
            zbin = (ev.sum() - (an0 + an1) * nc.sum()) / np.sqrt((an0 + an1) * nc.sum())
            print(f"   {key:20s} {lab:8s} P(<=1) MC={mc:.4e} rel={mc/(an0+an1)-1:+.3f} events={ev.sum():8.0f}"
                  f" z(sky scatter)={z:+.2f} z(Poisson count)={zbin:+.2f}")
