"""Cosmology / nuisance dependence of the blending correction and its impact on a cluster-count likelihood.
Inputs: data_cosmo/c02_grid.npz (3^4 grid), data_cosmo/c02_extra.npz (FD steps, Planck +-sigma, random points).
Likelihood: Poisson counts in (z, q_obs) bins, 8 z bins (0.01-3) x 10 q bins (5-inf), Asimov data (no noise),
full sky (N_det ~ 84.5k) and f_sky = 0.4.  All numbers printed.
"""
import os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import numpy as np
import cosmolib as cl
from c02_grid import S8, OM, OMB, SL, FIDP, STEP, SIG_P18

D = "/Users/boris/GitHub/sz-poisson-blending/data_cosmo"
G = dict(np.load(f"{D}/c02_grid.npz"))
E = dict(np.load(f"{D}/c02_extra.npz"))
PN = ["sigma8", "Om", "1-b", "sigma_lnY"]
NODES = [S8, OM, OMB, SL]
tab = cl.rho_tab()[0]
i23 = int(np.searchsorted(tab.th, 2.0))   # theta_A node 2.30'


def gidx(p):
    return [tuple(np.round(q, 6)) for q in G["params"]].index(tuple(np.round(p, 6)))


def eidx(p):
    return [tuple(np.round(q, 6)) for q in E["params"]].index(tuple(np.round(p, 6)))


def obs(S, j):
    return dict(ytot=S["ytot"][j], mu23=S["mu_c"][j, i23].sum(), var5_23=S["var5_c"][j, i23].sum(),
                var500_23=S["var500_c"][j, i23].sum(), var_hi_frac=S["var500_c"][j, i23, 2] / S["var500_c"][j, i23].sum(),
                mu_det=S["mu_det"][j], var5_det=S["var5_det"][j], lam1_23=S["lam1"][j, i23],
                N_single=S["N_single"][j], N_det_b=S["N_det_b"][j], ratio=S["ratio"][j], B2_10=S["B2_10"][j],
                B2_20=S["B2_20"][j], B3=S["B3"][j], lost=S["lost"][j], ybias=S["ybias"][j])


OBS = list(obs(G, 0).keys())
jf = gidx(FIDP)
fid = obs(G, jf)

print("=" * 100)
print(f"theta_A reference node = {tab.th[i23]:.2f}'.  Fiducial (grid centre) {FIDP}")
print("\n[1] one-at-a-time grid values (others at fiducial)")
hdr = f"{'param':10s} {'value':>6s} " + " ".join(f"{k:>10s}" for k in OBS)
print(hdr)
for a in range(4):
    for v in NODES[a]:
        p = list(FIDP); p[a] = v
        o = obs(G, gidx(p))
        print(f"{PN[a]:10s} {v:6.3f} " + " ".join(f"{o[k]:10.4g}" for k in OBS))

print("\n[2] logarithmic slopes d ln(obs)/d ln(param) at the fiducial: (a) small FD steps, (b) grid ends")
print(f"{'obs':10s} " + " ".join(f"{p+' FD':>14s} {p+' grid':>14s}" for p in PN))
slopes = {}
for k in OBS:
    row = []
    for a in range(4):
        pm = [list(FIDP), list(FIDP)]; pm[0][a] -= STEP[a]; pm[1][a] += STEP[a]
        o0, o1 = obs(E, eidx(pm[0]))[k], obs(E, eidx(pm[1]))[k]
        s_fd = (np.log(o1) - np.log(o0)) / (np.log(pm[1][a]) - np.log(pm[0][a])) if o0 > 0 and o1 > 0 else np.nan
        gm = [list(FIDP), list(FIDP)]; gm[0][a] = NODES[a][0]; gm[1][a] = NODES[a][2]
        g0, g1 = obs(G, gidx(gm[0]))[k], obs(G, gidx(gm[1]))[k]
        s_g = (np.log(g1) - np.log(g0)) / (np.log(gm[1][a]) - np.log(gm[0][a])) if g0 > 0 and g1 > 0 else np.nan
        row += [s_fd, s_g]; slopes[(k, a)] = s_fd
    print(f"{k:10s} " + " ".join(f"{x:14.3f}" for x in row))

print("\n[3] derivatives d(obs)/d(param) at the fiducial (central FD), and shift per Planck18 1-sigma")
print(f"{'obs':10s} " + " ".join(f"{'d/d'+p:>14s}" for p in PN) + f" {'x sig8_P18':>12s} {'x Om_P18':>12s}")
for k in ["ratio", "B2_10", "B3", "lost", "ybias", "mu_det", "var5_det"]:
    der = []
    for a in range(4):
        pm = [list(FIDP), list(FIDP)]; pm[0][a] -= STEP[a]; pm[1][a] += STEP[a]
        der.append((obs(E, eidx(pm[1]))[k] - obs(E, eidx(pm[0]))[k]) / (2 * STEP[a]))
    print(f"{k:10s} " + " ".join(f"{x:14.4g}" for x in der) + f" {der[0]*SIG_P18[0]:12.3g} {der[1]*SIG_P18[1]:12.3g}")

# ----------------------------------------------------------------------------- counts, correction, residuals
def Nb(S, j): return S["Nzq_b"][j]
def Nn(S, j): return S["Nzq_n"][j]


Cfid = Nb(G, jf) / Nn(G, jf)
print("\n[4] fiducial correction C(z,q) = N_blended / N_naive (rows z bins, columns q bins)")
print("    z edges", cl.ZEDGES, "\n    q edges", cl.QEDGES)
print(np.round(Cfid, 4))
print("    C(q) summed over z:", np.round(Nb(G, jf).sum(0) / Nn(G, jf).sum(0), 4))
print("    C(z) summed over q:", np.round(Nb(G, jf).sum(1) / Nn(G, jf).sum(1), 4))
print("\n    C(q) summed over z at sigma8 = 0.75 / 0.81 / 0.87 (others fiducial):")
for v in S8:
    p = list(FIDP); p[0] = v; j = gidx(p)
    print(f"     s8={v}: " + " ".join(f"{x:.4f}" for x in Nb(G, j).sum(0) / Nn(G, j).sum(0)))


def resid_stats(Nd, Nm, fsky):
    """Nd: true (blended) counts, Nm: model counts, full-sky; scaled by fsky. Returns dN_tot/N_tot, (dN_tot)/sqrt(N),
    chi2 = sum (Nd-Nm)^2/Nm over bins"""
    Nd, Nm = Nd * fsky, Nm * fsky
    s = Nm > 0
    return (Nd.sum() - Nm.sum()) / Nm.sum(), (Nd.sum() - Nm.sum()) / np.sqrt(Nm.sum()), (((Nd - Nm) ** 2)[s] / Nm[s]).sum()


print("\n[5] residual of a FIXED fiducial correction  model = C_fid(z,q) x N_naive(theta)  vs truth N_blended(theta)")
print("    and (for reference) of ignoring blending (C = 1).  chi2 over 80 (z,q) bins; Poisson sigma of N_tot.")
print(f"{'truth point':38s} {'fixed: dN/N':>11s} {'dN/sqrtN fs1':>12s} {'chi2 fs1':>9s} {'chi2 fs.4':>9s} | "
      f"{'C=1: dN/N':>10s} {'dN/sqrtN':>9s} {'chi2 fs1':>9s} {'chi2 fs.4':>9s}")
rows = []
pts = []
for a in range(2):
    for k in (-3, -1, 1, 3):
        p = list(FIDP); p[a] += k * SIG_P18[a]; pts.append((f"{PN[a]} {k:+d} sig_P18", E, eidx(p)))
for k in (-3, 3):
    p = list(FIDP); p[0] += k * SIG_P18[0]; p[1] -= k * SIG_P18[1]; pts.append((f"s8 {k:+d}s, Om {-k:+d}s (P18)", E, eidx(p)))
for a in range(4):
    for v in (NODES[a][0], NODES[a][2]):
        p = list(FIDP); p[a] = v; pts.append((f"grid {PN[a]}={v}", G, gidx(p)))
pts.append(("fiducial", G, jf))
for name, S, j in pts:
    f1 = resid_stats(Nb(S, j), Cfid * Nn(S, j), 1.0); f4 = resid_stats(Nb(S, j), Cfid * Nn(S, j), 0.4)
    n1 = resid_stats(Nb(S, j), Nn(S, j), 1.0); n4 = resid_stats(Nb(S, j), Nn(S, j), 0.4)
    print(f"{name:38s} {f1[0]:+11.2e} {f1[1]:+12.2f} {f1[2]:9.2f} {f4[2]:9.2f} | {n1[0]:+10.4f} {n1[1]:+9.1f} {n1[2]:9.1f} {n4[2]:9.1f}")
chi_all = np.array([resid_stats(Nb(G, j), Cfid * Nn(G, j), 1.0)[2] for j in range(len(G["params"]))])
dn_all = np.array([resid_stats(Nb(G, j), Cfid * Nn(G, j), 1.0)[0] for j in range(len(G["params"]))])
print(f"    all 81 grid points, fixed correction: max chi2 (full sky) = {chi_all.max():.1f} at {G['params'][chi_all.argmax()]}, "
      f"max |dN/N| = {np.abs(dn_all).max():.4f} at {G['params'][np.abs(dn_all).argmax()]}")

# ----------------------------------------------------------------------------- emulator test
lnC = np.log(np.where(G["Nzq_n"] > 0, G["Nzq_b"] / np.where(G["Nzq_n"] > 0, G["Nzq_n"], 1), 1.0))   # (81, nz, nq)
idx = np.array([[NODES[a].index(round(float(p[a]), 6)) for a in range(4)] for p in G["params"]])
lnCg = np.zeros((3, 3, 3, 3) + lnC.shape[1:])
for j in range(len(lnC)):
    lnCg[tuple(idx[j])] = lnC[j]


def lag(x, nodes):
    x0, x1, x2 = nodes
    return np.array([(x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2)), (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2)),
                     (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))])


def emu_C(p):
    L = [lag(p[a], NODES[a]) for a in range(4)]
    return np.exp(np.einsum("a,b,c,d,abcd...->...", *L, lnCg))


print("\n[6] emulator test: tensor-product quadratic interpolation of ln C(z,q) on the 3^4 grid, at points NOT on the grid")
print(f"{'truth point':48s} {'emu: dN/N':>10s} {'chi2 fs1':>9s} {'chi2 fs.4':>9s} | {'fixed: dN/N':>11s} {'chi2 fs1':>9s}")
emu_rows = []
for j, p in enumerate(E["params"]):
    em = emu_C(p) * Nn(E, j)
    e1 = resid_stats(Nb(E, j), em, 1.0); e4 = resid_stats(Nb(E, j), em, 0.4)
    f1 = resid_stats(Nb(E, j), Cfid * Nn(E, j), 1.0)
    emu_rows.append((p, e1, f1))
    print(f"{str(tuple(np.round(p, 4))):48s} {e1[0]:+10.2e} {e1[2]:9.3f} {e4[2]:9.3f} | {f1[0]:+11.2e} {f1[2]:9.2f}")

# ----------------------------------------------------------------------------- Fisher and parameter biases
def blended_derivs():
    d = []
    for a in range(4):
        pm = [list(FIDP), list(FIDP)]; pm[0][a] -= STEP[a]; pm[1][a] += STEP[a]
        d.append((Nb(E, eidx(pm[1])) - Nb(E, eidx(pm[0]))) / (2 * STEP[a]))
    return np.array(d)


dNb = blended_derivs()
N0 = Nb(G, jf)
mask = N0 > 1e-3
CASES = {"A: {s8,Om} free; 1-b, slnY fixed": ([0, 1], {}),
         "B: all 4 free; priors s(1-b)=0.04, s(slnY)=0.05": ([0, 1, 2, 3], {2: 0.04, 3: 0.05}),
         "C: all 4 free, no priors": ([0, 1, 2, 3], {})}


def fisher(dN, N, free, pri, fsky):
    F = fsky * np.einsum("ak,bk->ab", dN[free][:, mask], dN[free][:, mask] / N[mask])
    for ii, a in enumerate(free):
        if a in pri:
            F[ii, ii] += 1 / pri[a] ** 2
    return F


print("\n[7] Fisher forecast (blended model at fiducial; Poisson in 8x10 (z,q) bins). 1-sigma errors:")
SIG = {}
for cname, (free, pri) in CASES.items():
    for fs in (1.0, 0.4):
        F = fisher(dNb, N0, free, pri, fs)
        Ci = np.linalg.inv(F)
        sig = np.sqrt(np.diag(Ci)); SIG[(cname, fs)] = dict(zip(free, sig))
        rho = Ci[0, 1] / sig[0] / sig[1]
        print(f"  {cname:52s} fsky={fs}: " + " ".join(f"s({PN[a]})={s:.4f}" for a, s in zip(free, sig)) + f"  r(s8,Om)={rho:+.2f}")


def fit(model, data, theta0, free, pri, fsky, it=8):
    """Poisson Gauss-Newton (Fisher scoring) fit of model(theta) to Asimov data; priors centred on theta0."""
    th = np.array(theta0, float)
    for _ in range(it):
        M = model(th)
        dM = []
        for a in free:
            tp, tm = th.copy(), th.copy(); tp[a] += STEP[a]; tm[a] -= STEP[a]
            dM.append((model(tp) - model(tm)) / (2 * STEP[a]))
        dM = np.array(dM)
        s = M > 1e-3
        F = fsky * np.einsum("ak,bk->ab", dM[:, s], dM[:, s] / M[s])
        g = fsky * np.einsum("ak,k->a", dM[:, s], (data[s] - M[s]) / M[s])
        for ii, a in enumerate(free):
            if a in pri:
                F[ii, ii] += 1 / pri[a] ** 2
                g[ii] -= (th[a] - theta0[a]) / pri[a] ** 2
        step = np.linalg.solve(F, g)
        th[free] += step
        if np.abs(step).max() < 1e-6:
            break
    return th


_naive_cache = {}


def naive(th):
    key = tuple(np.round(th, 9))
    if key not in _naive_cache:
        _naive_cache[key] = cl.naive_counts(*th)
    return _naive_cache[key]


# sanity: naive_counts reproduces the stored naive counts at the fiducial
print(f"\n    check: naive_counts(fid) vs stored Nzq_n: max rel diff "
      f"{np.max(np.abs(naive(np.array(FIDP)) / np.where(Nn(G, jf) > 0, Nn(G, jf), 1) - 1)[Nn(G, jf) > 1e-3]):.2e}")

print("\n[8] parameter bias (Delta theta / sigma) from Asimov fits.  sigma = Fisher error of that case/fsky.")
print("  (i) IGNORE blending: data = blended counts at fiducial; model = naive single-cluster counts")
t0 = time.time()
for cname, (free, pri) in CASES.items():
    for fs in (1.0, 0.4):
        th = fit(naive, N0, np.array(FIDP), free, pri, fs)
        sig = SIG[(cname, fs)]
        print(f"   {cname:52s} fsky={fs}: " + " ".join(f"d{PN[a]}={th[a]-FIDP[a]:+.4f} ({(th[a]-FIDP[a])/sig[a]:+.2f} sig)" for a in free))
print(f"   ({time.time()-t0:.0f}s)")
print("  (ii) FIXED fiducial correction: data = blended counts at truth point; model = C_fid x naive(theta).")
print("       linear Fisher bias  F^-1 sum dN_b/dtheta (N_true - C_fid N_naive(truth)) / N  (full sky; x sqrt(0.4) for fsky=0.4)")
lin_fixed = {}
for name, S, j in pts:
    if name == "fiducial":
        continue
    dN = Nb(S, j) - Cfid * Nn(S, j)
    out = []
    for cname, (free, pri) in CASES.items():
        F = fisher(dNb, N0, free, pri, 1.0)
        b = np.einsum("ak,k->a", dNb[free][:, mask], dN[mask] / N0[mask])
        db = np.linalg.solve(F, b)
        sig = SIG[(cname, 1.0)]
        lin_fixed[(name, cname)] = db
        out.append(cname[0] + ": " + " ".join(f"{db[ii]/sig[a]:+.3f}" for ii, a in enumerate(free)))
    print(f"   {name:38s} bias/sigma ({PN}): " + " | ".join(out))
# nonlinear check of the linear bias on the largest-residual Planck point and the sigma8=0.87 grid point
for name, S, j in pts:
    if name not in ("sigma8 +3 sig_P18", "grid sigma8=0.87"):
        continue
    tt = S["params"][j]
    free, pri = CASES["A: {s8,Om} free; 1-b, slnY fixed"]
    th = fit(lambda t: Cfid * naive(t), Nb(S, j), tt, free, pri, 1.0, it=5)
    sig = SIG[("A: {s8,Om} free; 1-b, slnY fixed", 1.0)]
    db = lin_fixed[(name, "A: {s8,Om} free; 1-b, slnY fixed")]
    print(f"   nonlinear check {name}, case A: fit bias/sigma s8 {(th[0]-tt[0])/sig[0]:+.3f} Om {(th[1]-tt[1])/sig[1]:+.3f}"
          f"  (linear {db[0]/sig[0]:+.3f}, {db[1]/sig[1]:+.3f})")
print("  (iii) EMULATOR (quadratic on the 3^4 grid): linear Fisher bias at the random test points, case A and B, full sky")
for (p, e1, f1), j in zip(emu_rows, range(len(E["params"]))):
    if j < 18:
        continue
    em = emu_C(p) * Nn(E, j)
    dN = Nb(E, j) - em
    out = []
    for cname, (free, pri) in list(CASES.items())[:2]:
        F = fisher(dNb, N0, free, pri, 1.0)
        b = np.einsum("ak,k->a", dNb[free][:, mask], dN[mask] / N0[mask])
        db = np.linalg.solve(F, b)
        sig = SIG[(cname, 1.0)]
        out.append(cname[0] + ": " + " ".join(f"{db[ii]/sig[a]:+.3f}" for ii, a in enumerate(free)))
    print(f"   {str(tuple(np.round(p, 4))):40s} " + " | ".join(out))
np.savez(f"{D}/c03_summary.npz", Cfid=Cfid, dNb=dNb, N0=N0)
