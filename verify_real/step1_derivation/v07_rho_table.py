"""v07: my own cross-response table rho(theta_A, theta_B, s) for the direct per-object sums (v08).
Built from MY N_y (Woodbury, CLASS CMB), MY g(u) (2D FFT) with a 1D radial quadrature (dl = 1) in PHYSICAL s
(no scaled-s mapping), and validated against my 2D l-grid route (v04 method a) at several configurations.
Nodes: 151 theta in [0.2', 80'] (dln theta = 0.04), 671 s in [0, 800'] (0.04' steps to 8').
Also: trilinear interpolation accuracy vs exact, and support-truncation effects (variance beyond
S = 5 thA + 5 thB + M0 and the truncated DC response), weighted by a fresh catalogue, for M0 = 30, 60, 100, 150'.
Output rho_table_{baseline,goal,white}.npz (float32 rho).
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vlib as V
V.set_threads(4)
import numpy as np
from scipy.special import j0

t0 = time.time()
AM = V.AM
TH = np.geomspace(0.2, 80.0, 151)
S = np.concatenate([np.arange(0, 8, 0.04), np.arange(8, 40, 0.2), np.arange(40, 200, 1.0), np.arange(200, 800.01, 4.0)])
print(f"nodes: {len(TH)} theta (dln = {np.log(TH[1]/TH[0]):.4f}), {len(S)} s")


def build(mode):
    if mode == "white":
        ell = np.arange(0.0, 40000.5, 1.0); inv = V.inv_Ny_white(ell)
    else:
        ell = np.arange(100.0, 20000.5, 1.0); inv = V.inv_Ny(ell, mode)
    w = V.trap_w(ell) * ell / (2 * np.pi) * inv            # sr^-1 per unit
    G = V.g_u(np.outer(TH * AM, ell))                       # (NT, nl)
    sig2 = 1.0 / (G * G * w[None, :]).sum(1)                # sr^2
    J = j0(np.outer(ell, S * AM))                           # (nl, NS)
    R = np.empty((len(TH), len(TH), len(S)), np.float32)
    for i in range(len(TH)):
        H = G * (w * G[i])[None, :]
        R[i] = (sig2[i] * (H @ J)).astype(np.float32)
    del J
    # exact int rho^2 d^2s (Parseval, 1D) and DC (= 0 for MMF, sigma^2 RATIO^2/N_w for white)
    P = np.empty((len(TH), len(TH)))
    for i in range(len(TH)):
        P[i] = sig2[i] ** 2 * (G * G * (w * inv * G[i] ** 2)[None, :]).sum(1) / AM ** 2
    return R, np.sqrt(sig2) / AM ** 2, P, ell, inv


def rho_exact(mode, ell, inv, tA, tB, s_am):
    w = V.trap_w(ell) * ell / (2 * np.pi) * inv
    gA = V.g_u(ell * tA * AM); gB = V.g_u(ell * tB * AM)
    s2 = 1.0 / np.sum(w * gA * gA)
    return s2 * (j0(np.outer(np.atleast_1d(s_am) * AM, ell)) @ (w * gA * gB))


def trilin(R, tA, tB, s):
    lt0, dlt = np.log(TH[0]), np.log(TH[1] / TH[0])
    fa = np.clip((np.log(tA) - lt0) / dlt, 0, len(TH) - 1.000001); ia = fa.astype(int); wa = fa - ia
    fb = np.clip((np.log(tB) - lt0) / dlt, 0, len(TH) - 1.000001); ib = fb.astype(int); wb = fb - ib
    k = np.clip(np.searchsorted(S, s, side="right") - 1, 0, len(S) - 2); ws = (s - S[k]) / (S[k + 1] - S[k])
    out = 0.0
    for da, fa_ in ((0, 1 - wa), (1, wa)):
        for db, fb_ in ((0, 1 - wb), (1, wb)):
            out = out + fa_ * fb_ * ((1 - ws) * R[ia + da, ib + db, k] + ws * R[ia + da, ib + db, k + 1])
    return out


from szsim import sampler
cat = sampler.draw_catalog(20261111, fields=("Y500", "theta500"))
Yc = cat["Y500"].astype(np.float64); tc = cat["theta500"].astype(np.float64); del cat
# Y and Y^2 per theta_B node cell (nearest node in ln theta) for the truncation weights
ib = np.clip(np.rint((np.log(tc) - np.log(TH[0])) / np.log(TH[1] / TH[0])).astype(int), 0, len(TH) - 1)
WY = np.bincount(ib, Yc, len(TH)); WY2 = np.bincount(ib, Yc * Yc, len(TH))
del Yc, tc, ib
A_SKY = 4 * np.pi / AM ** 2
rng = np.random.default_rng(5)
ring = 2 * np.pi * S * V.trap_w(S)                          # d^2s weights (trapezoid in s)

for mode in ("baseline", "goal", "white"):
    ts = time.time()
    R, sig, P, ell, inv = build(mode)
    print(f"\n[{mode}] table built {time.time()-ts:.0f}s; rho(th,th,0) - 1: max {np.max(np.abs(np.diagonal(R[:, :, 0]) - 1)):.1e}")
    # (1) vs my 2D-grid route at the builder's s04 configurations (nearest table nodes)
    if mode != "white":
        seps = np.array([0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0])
        worst = 0.0
        for a, b in ((1, 1), (2, 4), (4, 2), (8, 8), (2, 16)):
            i = int(np.argmin(np.abs(np.log(TH / a)))); j = int(np.argmin(np.abs(np.log(TH / b))))
            r2d = np.zeros(len(seps)); accA = 0.0
            for lx, ly, l, w in V.grid_modes_chunks(10.0, 100.0, 20000.0):
                iv = V.inv_Ny(l, mode); gA = V.g_u(l * TH[i] * AM)
                accA += np.sum(w * gA * gA * iv)
                base = w * gA * V.g_u(l * TH[j] * AM) * iv
                for k, s_ in enumerate(seps):
                    r2d[k] += np.sum(base * np.cos(lx * s_ * AM))
            r2d /= accA
            rt = trilin(R, np.full(len(seps), TH[i]), np.full(len(seps), TH[j]), seps)   # 2.5' is not a node
            worst = max(worst, np.max(np.abs(rt - r2d)))
            print(f"   ({TH[i]:.3f}', {TH[j]:.3f}'): table {np.array2string(rt, precision=5, max_line_width=250)}\n"
                  f"   {'':18s}2D grid {np.array2string(r2d, precision=5, max_line_width=250)}")
        print(f"   max |table(1D quadrature) - 2D grid| = {worst:.1e}")
    # (2) trilinear interpolation accuracy vs exact at random points
    n = 400
    tA = np.exp(rng.uniform(np.log(0.25), np.log(60), n)); tB = np.exp(rng.uniform(np.log(0.25), np.log(60), n))
    err = []; errn = []
    for k in range(n):
        smax = min(5 * tA[k] + 5 * tB[k] + 60, 790)
        ss = np.concatenate([rng.uniform(0, min(3 * (tA[k] + tB[k]) + 3, smax), 6), rng.uniform(0, smax, 2)])
        ex = rho_exact(mode, ell, inv, tA[k], tB[k], ss)
        it = trilin(R, np.full(len(ss), tA[k]), np.full(len(ss), tB[k]), ss)
        r0 = abs(rho_exact(mode, ell, inv, tA[k], tB[k], [0.0])[0])
        err.append(np.max(np.abs(it - ex)) / max(1.0, r0))       # units of max(rho_AA(0)=1, |rho_AB(0)|)
        errn.append(np.max(np.abs(it - ex)) / r0)
    err = np.array(err); errn = np.array(errn)
    print(f"   interpolation |rho_interp - rho_exact|, {n} random (thA, thB) in [0.25', 60']^2 x 8 s: in units of "
          f"max(1, |rho_AB(0)|): median {np.median(err):.1e}, 95% {np.percentile(err, 95):.1e}, max {err.max():.1e}; "
          f"in units of |rho_AB(0)|: median {np.median(errn):.1e}, 95% {np.percentile(errn, 95):.1e}, max {errn.max():.1e}")
    # (3) truncation: fraction of the variance beyond S and truncated DC, catalogue-weighted, per theta_A
    showA = [int(np.argmin(np.abs(TH - t))) for t in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)]
    for M0 in (30.0, 60.0, 100.0, 150.0):
        fr = []; mu = []
        for i in showA:
            Sab = 5 * TH[i] + 5 * TH + M0
            inside = S[None, :] < Sab[:, None]
            tot = (WY2 * P[i]).sum()                                          # = A_sky var (Parseval, exact)
            within = (WY2 * ((R[i].astype(np.float64) ** 2 * ring[None, :] * inside).sum(1))).sum()
            dc = (WY * ((R[i].astype(np.float64) * ring[None, :] * inside).sum(1))).sum() / A_SKY / sig[i]
            fr.append(1 - within / tot); mu.append(dc)
        print(f"   M0 = {M0:5.0f}': var beyond support / total at thA = 0.5..32': " + " ".join(f"{x:+.1e}" for x in fr) +
              " | truncated-kernel mean shift [sigma]: " + " ".join(f"{x:+.1e}" for x in mu))
    np.savez(os.path.join(HERE, f"rho_table_{mode}.npz"), th=TH, s=S, rho=R, sigY=sig, P=P)
    del R
print(f"\ntotal {time.time()-t0:.0f}s")
