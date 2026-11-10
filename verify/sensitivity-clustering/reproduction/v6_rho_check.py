"""Sanity of the own response table: vs szsim exact cross_response, identity int rho d2s = sigma^2 RATIO^2/N_w,
and the analysis' quoted examples rho(3',0.5',0)=2.19, rho(2',2',2')=0.47."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import indep_model as im
from szsim import signal as sg
inst = im.Instrument(1.4, 4e-6); m = sg.SZModel()
thn = np.geomspace(0.04, 150, 130); sepn = np.concatenate([np.linspace(0, 20, 401), np.geomspace(20, 700, 181)[1:]])
rho = inst.rho_table(thn, sepn)
rng = np.random.default_rng(5); md = []
for _ in range(30):
    ia, ib = rng.integers(20, 110, 2); s = sepn[rng.integers(0, 300, 3)]
    ex = m.cross_response(thn[ia], thn[ib], s); tb = rho[ia, ib, np.searchsorted(sepn, s)]
    md.append(np.abs(tb - ex)[ex > 0.02] / ex[ex > 0.02])
md = np.concatenate(md)
print(f"own table (nodes) vs szsim cross_response: max rel diff {md.max():.2e}, median {np.median(md):.2e} (rho>0.02)")
for ia in (30, 60, 90):
    for ib in (30, 60, 90):
        integ = np.trapezoid(2 * np.pi * sepn * rho[ia, ib], sepn) * im.ARC ** 2
        tgt = (inst.sigY(thn[ia]) * im.ARC ** 2) ** 2 * im.RATIO ** 2 / inst.Nw_sr
        print(f"   thA={thn[ia]:.2f} thB={thn[ib]:.2f}: int rho d2s / target = {integ/tgt:.5f}")
print("exact own (direct) rho(3,0.5,0), rho(2,2,2):")
for a, b, s in ((3, 0.5, 0), (2, 2, 2)):
    l, w = inst.l, inst.wl * inst.l * inst.B2
    ga, gb = im.g(l * a * im.ARC), im.g(l * b * im.ARC)
    print(f"   {np.sum(w*ga*gb*np.special.j0(l*s*im.ARC)) if False else np.sum(w*ga*gb*__import__('scipy.special').special.j0(l*s*im.ARC))/np.sum(w*ga*ga):.4f}")
