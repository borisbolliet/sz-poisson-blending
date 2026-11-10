"""Analytic ingredients for Q1: expected full-sky counts Nbar of every population used,
surface densities, mu(theta) and Poisson P(N); flat-sky vs spherical-cap areas; the
triangle constant p3 for survey-wide triple counts.  Writes data/q1_aperture/nbar.json."""
import json, os, time
import numpy as np
from szsim import cosmo_hmf as ch, signal as sg, sampler
from q1_common import *

t0 = time.time()
T = ch.default_table(); m = sg.SZModel()
nb = {}
# mass-floor populations (sampler cell sums == table trapezoid)
for lab, M in [("M1e13", 1e13), ("M3e13", 3e13), ("M1e14", 1e14), ("M2e14", 2e14), ("M3e14", 3e14)]:
    nb[lab] = sampler.expected_count(Mmin=M)
for (z1, z2) in ZSLICES:
    nb[f"M1e14_z{z1}-{z2}"] = sampler.expected_count(Mmin=1e14, zmin=z1, zmax=z2)
    nb[f"M3e13_z{z1}-{z2}"] = sampler.expected_count(Mmin=3e13, zmin=z1, zmax=z2)
# detected populations: q_obs > q_cut, all halos M>1e13 (floor 1e13), and floor 1e14 / 3e14 variants
for q in [4, 5, 6, 7]:
    nb[f"det{q}"] = m.expected_detections(q, T)
    nb[f"det{q}_floor1e14"] = m.expected_detections(q, T, Mmin=1e14)
for q in [5]:
    for (z1, z2) in ZSLICES:
        nb[f"det{q}_z{z1}-{z2}"] = m.expected_detections(q, T, zmin=z1, zmax=z2)
nb["undet5_M1e14"] = nb["M1e14"] - nb["det5_floor1e14"]
nb["det7_floor3e14"] = m.expected_detections(7, T, Mmin=3e14)
os.makedirs(DATA, exist_ok=True)
json.dump(nb, open(os.path.join(DATA, "nbar.json"), "w"), indent=1)

print("== Expected full-sky counts and surface densities ==")
print(f"{'population':24s} {'Nbar':>14s} {'Sigma[deg^-2]':>14s} {'Sigma[arcmin^-2]':>17s}  mu(1')    mu(5')    mu(10')   mu(60')")
for k, v in nb.items():
    s = v / SKY_DEG2
    print(f"{k:24s} {v:14.2f} {s:14.5f} {s/3600:17.4e}  " + "  ".join(f"{mu_of(v, t):.3e}" for t in [1, 5, 10, 60]))

print("\n== det share from 1e13<M<3e13 and z-slice closure ==")
for q in [4, 5, 6]:
    print(q, "share(M<1e14) =", 1 - nb[f"det{q}_floor1e14"] / nb[f"det{q}"])
print("det5 z-slices sum / det5 =", sum(nb[f"det5_z{a}-{b}"] for a, b in ZSLICES) / nb["det5"])

print("\n== Poisson P(N) around a point, det5 population ==")
for t in [1, 2, 5, 10, 30, 60]:
    mu = mu_of(nb["det5"], t)
    print(f"theta={t:5.1f}'  mu={mu:.4e}  " + " ".join(f"P({n})={P_N(n, mu):.4e}" for n in range(6)))

print("\n== flat-sky pi theta^2 vs exact cap 2pi(1-cos theta) ==")
print(f"{'theta':>10s} {'flat/cap-1':>12s} {'theta^2/12':>12s}")
for tdeg in [1/60, 1, 5, 10, 19.8, 30, 60, 90, 120, 180]:
    t = tdeg * 60
    r = flat_area(t) / cap_area_stable(t) - 1
    print(f"{tdeg:9.3f}d {r:12.4e} {(tdeg*np.pi/180)**2/12:12.4e}")
# theta at which the flat approximation errs by 0.1%, 1%, 10%
from scipy.optimize import brentq
for eps in [1e-3, 1e-2, 0.1]:
    th = brentq(lambda d: flat_area(d * 60) / cap_area_stable(d * 60) - 1 - eps, 1e-3, 179.9)
    print(f"flat-sky area error {eps:.0e} reached at theta = {th:.3f} deg")

# triangle constant: two points uniform in a disk of radius 1, P(|y-z| < 1)
p3_exact = 1 - 3 * np.sqrt(3) / (4 * np.pi)
rng = np.random.default_rng(0)
n = 4_000_000
r = np.sqrt(rng.random((2, n))); ph = 2 * np.pi * rng.random((2, n))
d = np.hypot(r[0] * np.cos(ph[0]) - r[1] * np.cos(ph[1]), r[0] * np.sin(ph[0]) - r[1] * np.sin(ph[1]))
p3_mc = np.mean(d < 1)
print(f"\np3 = 1 - 3sqrt3/(4pi) = {p3_exact:.6f};  MC (4e6) = {p3_mc:.6f} +- {np.sqrt(p3_mc*(1-p3_mc)/n):.6f}"
      f"  -> {(p3_mc-p3_exact)/np.sqrt(p3_mc*(1-p3_mc)/n):+.2f} sigma")
print("time", time.time() - t0)
