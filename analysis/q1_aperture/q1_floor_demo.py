"""Repair phase: reproduce the C12 error-bar pitfall.  det5|M3e13, P(N=0) at 10' (mu=9.73, ~5 events/sky): z with the
plain sky-to-sky standard error vs with the binomial floor, for the first 6 skies (seeds 1001-1006) and for 40/120 skies."""
import os, glob, json
import numpy as np
from q1_common import *
from q1_mc_fullsky import FINE
NB = json.load(open(os.path.join(DATA, "nbar.json")))
S = [np.load(f, allow_pickle=True).item() for f in sorted(glob.glob(os.path.join(DATA, "fullsky", "sky_*.npy")))]
j = int(np.argmin(abs(FINE - 10))); A = np.exp(-mu_of(NB["M3e13"], 10))
for R in (6, 40, 120):
    P0 = np.array([1 - s["cdf"]["det5|M3e13"][0, j] for s in S[:R]]); nc = np.array([s["ncen"]["det5|M3e13"] for s in S[:R]])
    se = P0.std(ddof=1) / np.sqrt(R); fl = np.sqrt(A * (1 - A) / (nc.mean() * R))
    print(f"R={R:3d}: events/sky={np.round(P0*nc).astype(int)[:6]}... P0 MC={P0.mean():.3e} analytic={A:.3e}"
          f"  z(no floor)={(P0.mean()-A)/se:+.2f}  z(floored)={(P0.mean()-A)/max(se, fl):+.2f}")
