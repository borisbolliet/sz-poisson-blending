"""Analyse q1_void_tail.py output: P(N=0) and P(N<=1) of M>=3e13 neighbours (Nbar from nbar.json 'M3e13'),
detected centres (q>4,5,6, centre excluded) and random centres; z from sky scatter of per-sky residuals,
unconditional Poisson and conditional Bin(N - [centre in B], f); paired det - rand difference per sky."""
import os, glob, json
import numpy as np
from scipy import stats
from q1_common import *
from q1_void_tail import TH, NRAND
NB = json.load(open(os.path.join(DATA, "nbar.json")))
S = [np.load(f, allow_pickle=True).item() for f in sorted(glob.glob(os.path.join(DATA, "void_tail", "sky_*.npy")))]
R = len(S); N = np.array([s["N"] for s in S], float)
print(f"R={R} floor-3e13 skies (seeds {S[0]['seed']}..{S[-1]['seed']}); mean N={N.mean():.1f} Nbar={NB['M3e13']:.1f}"
      f" z={(N.mean()-NB['M3e13'])/np.sqrt(NB['M3e13']/R):+.2f}")
f = np.sin(TH * ARCMIN / 2) ** 2
def zres(ev, nc, p):
    r = ev - p * nc; return r.sum() / (r.std(ddof=1) * np.sqrt(len(r)))
for j, t in enumerate(TH):
    mu = NB["M3e13"] * f[j]
    for kind, an in [("n0", np.exp(-mu)), ("le1", np.exp(-mu) * (1 + mu))]:
        print(f"theta={t}' mu={mu:.4f} P({'N=0' if kind=='n0' else 'N<=1'}) Poisson={an:.4e}")
        er = np.array([s[f"rand_{kind}"][j] for s in S], float); nr = np.full(R, NRAND, float)
        cr = stats.binom.cdf(0 if kind == "n0" else 1, N, f[j])
        print(f"   rand   : MC={er.sum()/nr.sum():.4e} rel={er.sum()/nr.sum()/an-1:+.4f} events={er.sum():.0f}"
              f" z={zres(er, nr, an):+.2f} z_cond={zres(er, nr, cr):+.2f}")
        for q in (4, 5, 6):
            ed = np.array([s[f"det{q}_{kind}"][j] for s in S], float); nd = np.array([s[f"n{q}"] for s in S], float)
            cd = stats.binom.cdf(0 if kind == "n0" else 1, N - 1, f[j])
            # paired per-sky difference of fractions det - rand
            dd = ed / nd - er / nr
            print(f"   det{q}   : MC={ed.sum()/nd.sum():.4e} rel={ed.sum()/nd.sum()/an-1:+.4f} events={ed.sum():.0f}"
                  f" z={zres(ed, nd, an):+.2f} z_cond={zres(ed, nd, cd):+.2f};  paired det-rand: {dd.mean()/an:+.4f}"
                  f" +- {dd.std(ddof=1)/np.sqrt(R)/an:.4f} (rel), z={dd.mean()/(dd.std(ddof=1)/np.sqrt(R)):+.2f}")
