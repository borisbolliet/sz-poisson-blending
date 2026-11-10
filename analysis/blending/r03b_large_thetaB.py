"""[repair] Neighbours with theta500 > 80' (beyond the rho-table range 0.25'-80'): the trilinear lookup clamps
thB to 80' while the support S uses the true thB, i.e. it stretches the 80' kernel.  Quantify: their expected
full-sky number, the sky fraction within their kernel support, and their share of mu and Var at thA = 1.1', 4.15'
(exact Parseval / DC vs table)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blendlib as bl
from szsim import signal as sg, cosmo_hmf as ch
m = sg.SZModel(); A = dict(np.load(os.path.join(bl.DATA, "analytic.npz"))); TH = A["th"]
ug, _ = sg.template_hankel(); sb = m.sigma_beam_rad; Nw = m.N_white_sr


def K_exact(thA, thBs, nl=40000):
    out = np.empty(len(thBs))
    sA = m.sigma_Y_exact(thA, nl=80000)[0] * ch.ARCMIN ** 2
    for k, tb in enumerate(thBs):
        tA, tB = thA * ch.ARCMIN, tb * ch.ARCMIN
        lm = min(9.0 / sb, ug[-1] / max(tA, tB))
        l = np.linspace(0, lm, nl)
        gA = sg.template_hankel(l * tA); gB = sg.template_hankel(l * tB)
        I = np.trapezoid(l * np.exp(-2 * (l * sb) ** 2) * gA ** 2 * gB ** 2, l) / (2 * np.pi) / Nw ** 2
        out[k] = sA ** 4 * I / ch.ARCMIN ** 2
    return out


def K_table(tab, thA, thBs):
    out = np.empty(len(thBs)); t = tab.t
    dt = np.full(len(t), t[1] - t[0]); dt[0] *= 0.5; dt[-1] *= 0.5
    for k, tb in enumerate(thBs):
        S = 5 * thA + 5 * tb + tab.R_BEAM
        s = S * t ** 2
        r = np.array([bl.rho_interp_py(tab, thA, tb, x) for x in s])
        out[k] = np.sum(4 * np.pi * S * S * t ** 3 * dt * r ** 2)
    return out


thB, YB, cB, wB = bl.neighbour_population(zsub=1, msub=6, ngh=16)
big = thB > 80.0
print(f"expected full-sky number of halos (M>1e13) with theta500 > 80': {wB[big].sum()*bl.ARCMIN2_FULLSKY:.3f}; "
      f"> 40': {wB[thB>40].sum()*bl.ARCMIN2_FULLSKY:.2f}")
tab = bl.load_rho_table()
G = np.geomspace(80.0, thB.max() * 1.001, 60)
for t in (1.1, 4.15):
    i = int(np.argmin(np.abs(TH - t))); thA = TH[i]; sA = float(m.sigma_Y(thA))
    Ke = K_exact(thA, G); Kt = K_table(tab, thA, G)
    S = 5 * thA + 5 * thB[big] + tab.R_BEAM
    frac_area = np.sum(wB[big] * np.pi * S ** 2)    # expected number of such halos whose support covers a point
    ve = np.sum(wB[big] * YB[big] ** 2 * np.exp(np.interp(np.log(thB[big]), np.log(G), np.log(Ke)))) / sA ** 2
    vt = np.sum(wB[big] * YB[big] ** 2 * np.exp(np.interp(np.log(thB[big]), np.log(G), np.log(Kt)))) / sA ** 2
    dc = bl.dc_response(m.sigma_Y_exact(thA, nl=80000), m)[0]
    mue = np.sum(wB[big] * YB[big]) * dc / sA
    # table DC response for these neighbours: int 2pi s rho_tab ds
    def dct(tb):
        S_ = 5 * thA + 5 * tb + tab.R_BEAM; tt = tab.t; dt = np.full(len(tt), tt[1] - tt[0]); dt[0] *= .5; dt[-1] *= .5
        return np.sum(4 * np.pi * S_ ** 2 * tt ** 3 * dt * np.array([bl.rho_interp_py(tab, thA, tb, S_ * x * x) for x in tt]))
    D = np.array([dct(g) for g in G])
    mut = np.sum(wB[big] * YB[big] * np.interp(np.log(thB[big]), np.log(G), D)) / sA
    print(f"thA={thA:.3f}': P(point inside the support of a thB>80' halo) = {frac_area:.4f}; "
          f"mu from them exact {mue:.5f} table {mut:.5f}; Var from them exact {ve:.4f} table {vt:.4f}  [BLENDLIB_FIXED_KERNEL]")
