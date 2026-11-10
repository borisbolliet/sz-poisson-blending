"""Repair phase: Hotelling T^2 for random-centred and self (det_q|det_q) combos with the prediction CONDITIONAL on each
sky's realised population size N (binomial law: random centre -> Bin(N, f); member centre -> Bin(N-1, f)),
same <=8 bins as q1_calibrated_tests.py.  Separates 'the Poisson law fails' from 'this sky set happens to have a
high/low total count'.  usage: Q1_SEEDS=a-b python q1_calibrated_conditional.py"""
import os, glob, json
import numpy as np
from scipy import stats
from q1_common import *
from q1_mc_fullsky import FINE
from q1_calibrated_tests import S, R, iT, NB, hot_bins, hotelling
f = np.sin(THETAS * ARCMIN / 2) ** 2
print(f"conditional-on-N Hotelling, R={R} (seeds {S[0]['seed']}..{S[-1]['seed']})")
for key in S[0]["cdf"]:
    c, p = key.split("|")
    if not (c == "rand" or c == p):
        continue
    N = np.array([s["npop"][p] for s in S], float)
    Nm = N - (0 if c == "rand" else 1)
    Fs = np.array([s["cdf"][key][:, iT] for s in S])
    Fan = np.array([P_geq(k + 1, mu_of(NB[p], THETAS)) for k in range(KMAX)])
    ncen = np.mean([s["ncen"][key] for s in S])
    sel = hot_bins(Fan, ncen)
    Dc = np.array([[Fs[r, k, j] - stats.binom.sf(k, Nm[r], f[j]) for (k, j) in sel] for r in range(R)])
    Du = np.array([[Fs[r, k, j] - Fan[k, j] for (k, j) in sel] for r in range(R)])
    _, _, pu = hotelling(Du); _, _, pc = hotelling(Dc)
    print(f"{key:24s} mean N={N.mean():11.1f} Nbar={NB[p]:11.1f} z_N={(N.mean()-NB[p])/np.sqrt(NB[p]/R):+5.2f}"
          f"  Hotelling p: unconditional={pu:.3f}  conditional-on-N={pc:.3f}")
