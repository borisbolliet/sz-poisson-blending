"""[repair after verification] (1) exact DC response at the quoted theta_A; (2) mu at the quoted nodes;
(3) exact (Parseval, untruncated) variance of the neighbour sum per mass class, compared with the
table-kernel variance; (4) Lambda(x>1) per class; (5) b08 z-subsampling ratio.

Variance: Var(delta) = sum_pop w_B (Y_B / sigma_A)^2 K(thA, thB),  K = int rho^2 d^2s
 exact:  K = sigma_A^4 (1/2pi) int dl l B_l^4 g(l thA)^2 g(l thB)^2 / N_w^2       (Parseval)
 table:  K_tab = int 2 pi s rho_tab(s)^2 ds with the trilinear table and the SAME s-quadrature as the
         rate histogram (s = S t^2, trapezoid in t), for the production table (NT=40) and, if present,
         the finer NT=79 table (nodes nest the NT=40 ones).
"""
import os, sys, re
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl
from szsim import signal as sg, cosmo_hmf as ch

m = sg.SZModel()
A = dict(np.load(os.path.join(bl.DATA, "analytic.npz")))
TH = A["th"]

# (1) DC response, exact at the quoted theta
thq = np.array([0.25, 1.0, 3.0, 10.0, 30.0])
dc_ex = bl.dc_response(m.sigma_Y_exact(thq, nl=80000), m)
dc_sp = bl.dc_response(m.sigma_Y(thq), m)
dc_old = np.interp(np.log(thq), np.log(TH), bl.dc_response(m.sigma_Y_exact(TH, nl=80000), m))
print("(1) DC response [arcmin^2] at thA = 0.25, 1, 3, 10, 30'")
print("    exact (sigma_Y_exact):          ", np.round(dc_ex, 3))
print("    library spline sigma_Y:         ", np.round(dc_sp, 3))
print("    old print (interp of node values):", np.round(dc_old, 3), " -> old/exact - 1 =", np.round(dc_old / dc_ex - 1, 4))

# (2) mu at nodes
print("\n(2) mu (monopole shift) at nodes, interpolated-kernel value used everywhere (mu_exact key) and the formula ybar*DC/sigma:")
for t in (0.5, 1, 2, 4, 8, 16):
    i = np.searchsorted(TH, t)
    print(f"    thA={TH[i]:6.3f}': mu = {A['mu_exact'][i].sum():.4f}  formula {A['mu_formula'][i].sum():.4f}  "
          f"(kernel/formula - 1 = {A['mu_exact'][i].sum()/A['mu_formula'][i].sum()-1:+.4f}); classes {np.round(A['mu_exact'][i], 4)}")

# (3) variance
pop = bl.neighbour_population(zsub=1, msub=6, ngh=16)
thB, YB, cB, wB = pop
ug, _ = sg.template_hankel()
sb = m.sigma_beam_rad
Nw = m.N_white_sr


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


tabs = {"NT=40": bl.load_rho_table()}
f79 = os.path.join(bl.DATA, "rho_table_nt79.npz")
if os.path.exists(f79):
    tabs["NT=79"] = bl.RhoTable(np.load(f79))
G = np.geomspace(thB.min() * 0.999, thB.max() * 1.001, 400)
print("\n(3) variance of delta per neighbour class [1e13-3e13, 3e13-1e14, >1e14]")
for t in (1.1, 4.15):
    i = int(np.argmin(np.abs(TH - t))); thA = TH[i]; sigA = float(m.sigma_Y(thA))
    Ke = K_exact(thA, G)
    var = {"exact": np.array([np.sum((wB * YB ** 2 * np.exp(np.interp(np.log(thB), np.log(G), np.log(Ke))))[cB == c]) for c in range(3)]) / sigA ** 2}
    for nm, tb in tabs.items():
        Kt = K_table(tb, thA, G)
        var[nm] = np.array([np.sum((wB * YB ** 2 * np.exp(np.interp(np.log(thB), np.log(G), np.log(Kt))))[cB == c]) for c in range(3)]) / sigA ** 2
        rel = Kt / Ke - 1
        print(f"   thA={thA:.3f}': K_table({nm})/K_exact - 1 over thB grid: min {rel.min():+.4f} max {rel.max():+.4f}; "
              f"at thB in [20,35]': max {rel[(G > 20) & (G < 35)].max():+.4f}")
    v40 = A["WX2"][i].sum(1)
    print(f"   thA={thA:.3f}': exact (untruncated) {np.round(var['exact'], 4)} tot {var['exact'].sum():.4f}; "
          + "; ".join(f"table {nm} (untruncated) {np.round(v, 4)} tot {v.sum():.4f} ({v.sum()/var['exact'].sum()-1:+.3f})" for nm, v in var.items() if nm != "exact")
          + f"; production rate histogram (x<500) {np.round(v40, 4)} tot {v40.sum():.4f}; share M>1e14 (exact) {var['exact'][2]/var['exact'].sum():.3f}")

# (4) Lambda(x>1) per class
print("\n(4) expected number of neighbours with x > 1 per filter, per class (production rate histograms):")
for t in (0.5, 1.1, 4.15):
    i = int(np.argmin(np.abs(TH - t)))
    lam = np.array([bl.PD.lam_above(A["W"][i, c], A["WX"][i, c], 1.0) + A["CAP"][i, c] for c in range(3)])
    print(f"   thA={TH[i]:.3f}': per class {np.round(lam, 4)}, total {lam.sum():.4f}; shares {np.round(lam / lam.sum(), 3)}")

# (5) b08 ratio
txt = open(os.path.join(HERE, "b08_pop_convergence.log")).read()
for line in txt.splitlines():
    if line.startswith("thA=0.524'") and ("zsub=6 msub=6 ngh=16" in line or "zsub=1 msub=3 ngh=32" in line):
        v = float(re.findall(r"floor 1e14:.*P\(>50\)=([0-9.e+-]+)", line)[0])
        print(f"\n(5) {line.split(' (')[0]}: floor 1e14 P(>50) = {v:.4e}")
        if "zsub=6" in line:
            v6 = v
        else:
            print(f"    ratio zsub=6 / zsub=1 = {v6 / v:.4f}")
