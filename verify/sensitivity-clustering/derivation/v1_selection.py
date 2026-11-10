"""V1: independent checks of RATIO, sigma_Y, HMF vs szsim spot values, N_det / Sigma_det for all configs, slopes."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import own_model as om

print(f"RATIO Y5R500/Y500 = {om.RATIO:.5f}")
for t in [1, 3, 10]:
    print(f"sigma_Y({t}') beam1.4 noise4e-6 = {om.sigmaY(t,1.4,4e-6)[0]:.4e}")
print(f"point limit: {om.sigmaY(1e-4,1.4,4e-6)[0]:.4e}")
# Planck y-noise
x = 143 / (6.62607015e-34 * 1e9 / (1.380649e-23 * 2.7255)) ** -1 if False else 143e9 * 6.62607015e-34 / (1.380649e-23 * 2.7255)
f = x / np.tanh(x / 2) - 4
ynP = 33 / (2.7255e6 * abs(f))
print(f"x(143)={x:.4f} f={f:.4f} Planck y-noise = {ynP:.4e} arcmin")
# spot checks of q
for M, z in [(2e14, 0.1), (2e14, 0.5), (3e14, 1.0), (1e14, 0.1)]:
    q = om.Ymed(M, z) / om.sigmaY(om.theta500(M, z), 1.4, 4e-6)[0]
    print(f"q_mean(M={M:.0e}, z={z}) = {q:.2f}, theta500 = {om.theta500(M,z):.3f}'")

z, lnM, d2N = om.grid()
M = np.exp(lnM); wz = om.trapw(z); wM = om.trapw(lnM)
for Mmin in [1e13, 1e14, 3e14]:
    print(f"N(M>{Mmin:.0e}) = {wz @ d2N @ (wM*(M>=Mmin*0.9999)):.1f}  (grid trapz, M-edge approx)")
FSKY_DEG2 = 4 * np.pi * (180 / np.pi) ** 2

def Ndet(beam, noise, sig, qc=5.0, Mmin=1e13, zmax=3.0, model="tinker08", dd=None):
    zz, ll, D = (z, lnM, d2N) if dd is None else dd
    MM = np.exp(ll)
    Pd = om.pdet_grid(zz, MM, beam, noise, sig, qc)
    w = om.trapw(ll) * (MM >= Mmin * 0.9999)
    wzz = om.trapw(zz) * (zz <= zmax + 1e-9)
    return wzz @ (D * Pd) @ w

res = {}
cfgs = {"base": (1.4, 4e-6, 0.173), "noise_x0.5": (1.4, 2e-6, 0.173), "noise_x2": (1.4, 8e-6, 0.173),
        "beam_5": (5.0, 4e-6, 0.173), "planck_like": (7.3, ynP, 0.173), "scatter_0": (1.4, 4e-6, 0.0),
        "scatter_0.3": (1.4, 4e-6, 0.3)}
for k, (b, n, s) in cfgs.items():
    a = Ndet(b, n, s); c = Ndet(b, n, s, Mmin=1e14)
    print(f"{k:12s} N_det(M>1e13)={a:9.1f} Sigma={a/FSKY_DEG2:.4f}/deg2 | N_det(M>1e14)={c:9.1f} Sigma={c/FSKY_DEG2:.4f}")
for zm in [1.0, 2.0]:
    a = Ndet(1.4, 4e-6, 0.173, zmax=zm); c = Ndet(1.4, 4e-6, 0.173, Mmin=1e14, zmax=zm)
    print(f"zmax={zm}: N_det(M>1e13)={a:9.1f} Sigma={a/FSKY_DEG2:.4f} | M>1e14 {c:9.1f} Sigma={c/FSKY_DEG2:.4f}")
for mod in ["bocquet16", "despali16"]:
    dd = om.grid(nz=300, nM=300, model=mod)
    a = Ndet(1.4, 4e-6, 0.173, dd=dd); c = Ndet(1.4, 4e-6, 0.173, Mmin=1e14, dd=dd)
    print(f"{mod}: N_det(M>1e13)={a:9.1f} Sigma={a/FSKY_DEG2:.4f} | M>1e14 {c:9.1f} Sigma={c/FSKY_DEG2:.4f}")
# slopes
fs = np.array([0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0])
Nf = np.array([Ndet(1.4, 4e-6 * f, 0.173) for f in fs])
print("Sigma vs noise f:", dict(zip(fs, (Nf / FSKY_DEG2).round(3))))
print("local slopes:", np.round(np.diff(np.log(Nf)) / np.diff(np.log(fs)), 3), "at", np.round(np.sqrt(fs[1:]*fs[:-1]), 2))
bs = np.array([0.5, 1.0, 1.4, 2.0, 3.0, 5.0, 7.3, 10.0])
Nb = np.array([Ndet(b, 4e-6, 0.173) for b in bs])
print("Sigma vs beam:", dict(zip(bs, (Nb / FSKY_DEG2).round(3))))
print("local slopes:", np.round(np.diff(np.log(Nb)) / np.diff(np.log(bs)), 3), "at", np.round(np.sqrt(bs[1:]*bs[:-1]), 2))
