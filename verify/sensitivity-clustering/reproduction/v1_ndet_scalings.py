"""V1: independent selection integral -> Sigma_det(q>5) per configuration (C2), noise/beam scalings (C3),
ybar (C4), and sanity of own sigma_Y vs the build report numbers."""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import indep_model as im

t0 = time.time()
DEG2 = (180 / np.pi) ** 2
print(f"RATIO_5R500 (own) = {im.RATIO:.5f}")
base = im.Instrument(1.4, 4e-6)
print("own sigma_Y(1',3',10') =", [f"{base.sigY(t):.4e}" for t in (1, 3, 10)], " (build report 7.23e-6, 1.46e-5, 4.31e-5)")
# Planck-like noise conversion
x = 143 / 56.7798; f = x / np.tanh(x / 2) - 4
yP = 33e-6 / (2.7255 * abs(f))
print(f"Planck-like y-noise: f(143)={f:.4f}, y = {yP:.4e} arcmin (claimed 1.164e-5)")

cfg = {
    "base": dict(beam=1.4, noise=4e-6), "noise_x0.5": dict(beam=1.4, noise=2e-6), "noise_x2": dict(beam=1.4, noise=8e-6),
    "beam_5": dict(beam=5.0, noise=4e-6), "planck_like": dict(beam=7.3, noise=yP),
    "scatter_0": dict(sig=0.0), "scatter_0.3": dict(sig=0.3), "hmf_bocquet16": dict(model="bocquet16"),
    "hmf_despali16": dict(model="despali16"), "zmax_1": dict(zmax=1.0), "zmax_2": dict(zmax=2.0)}
claimed13 = {"base": 2.083, "noise_x0.5": 6.038, "noise_x2": 0.638, "beam_5": 0.798, "planck_like": 0.095, "scatter_0": 2.010,
             "scatter_0.3": 2.234, "hmf_bocquet16": 1.894, "hmf_despali16": 1.855, "zmax_1": 1.843, "zmax_2": 2.078}
claimed14 = {"base": 2.0381, "noise_x0.5": 5.5091, "noise_x2": 0.6328, "beam_5": 0.7756, "planck_like": 0.0935, "scatter_0": 1.9719,
             "scatter_0.3": 2.1700, "hmf_bocquet16": 1.8561, "hmf_despali16": 1.8134, "zmax_1": 1.7999, "zmax_2": 2.0337}
insts = {}
out = {}
for name, c in cfg.items():
    key = (c.get("beam", 1.4), c.get("noise", 4e-6))
    if key not in insts:
        insts[key] = im.Instrument(*key)
    inst = insts[key]
    kw = dict(sig=c.get("sig", 0.173), model=c.get("model", "tinker08"), zmax=c.get("zmax", 3.0))
    n13 = im.N_det(inst, 5.0, Mmin=1e13, **kw)
    n14 = im.N_det(inst, 5.0, Mmin=1e14, **kw)
    s13, s14 = n13 / (4 * np.pi * DEG2), n14 / (4 * np.pi * DEG2)
    out[name] = dict(N13=n13, N14=n14)
    print(f"{name:14s} N_det(M>1e13) = {n13:10.1f}  Sigma = {s13:.4f}/deg2 (claimed {claimed13[name]:.3f}, ratio {s13/claimed13[name]:.4f}) | "
          f"N_det(M>1e14) = {n14:10.1f} Sigma = {s14:.4f} (claimed {claimed14[name]:.4f}, ratio {s14/claimed14[name]:.4f})", flush=True)

print("\nnoise scaling (beam 1.4'):")
fg = np.array([0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0])
Nf = np.array([im.N_det(im.Instrument(1.4, 4e-6 * f), 5.0) for f in fg])
print("  Sigma_det:", ", ".join(f"{f:g}:{n/(4*np.pi*DEG2):.3f}" for f, n in zip(fg, Nf)))
print("  local slopes:", ", ".join(f"{np.sqrt(fg[i]*fg[i+1]):.2f}:{np.log(Nf[i+1]/Nf[i])/np.log(fg[i+1]/fg[i]):.3f}" for i in range(len(fg) - 1)))
print("beam scaling (noise 4e-6):")
bg = np.array([0.5, 1.0, 1.4, 2.0, 3.0, 5.0, 7.3, 10.0])
Nb = np.array([im.N_det(im.Instrument(b, 4e-6), 5.0) for b in bg])
print("  Sigma_det:", ", ".join(f"{b:g}:{n/(4*np.pi*DEG2):.3f}" for b, n in zip(bg, Nb)))
print("  local slopes:", ", ".join(f"{np.sqrt(bg[i]*bg[i+1]):.2f}:{np.log(Nb[i+1]/Nb[i])/np.log(bg[i+1]/bg[i]):.3f}" for i in range(len(bg) - 1)))

print("\nybar (own):")
for name, kw in (("M>1e13", dict()), ("M>1e14", dict(Mmin=1e14)), ("M>3e13", dict(Mmin=3e13)), ("zmax1", dict(zmax=1.0)),
                 ("despali16", dict(model="despali16")), ("bocquet16", dict(model="bocquet16")), ("sig0", dict(sig=0.0))):
    print(f"   {name:10s} ybar = {im.ybar(**kw):.5e}")
print(f"mu at theta=1' base (sigma RATIO ybar/Nw) = {base.mu_coef(1.0)*im.ybar():.4f}")
json.dump(out, open(os.path.join(im.CACHE, "v1_ndet.json"), "w"))
print(f"done {time.time()-t0:.0f}s")
