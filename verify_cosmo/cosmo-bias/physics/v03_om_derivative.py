"""Om-derivative of the counts: smoothness of direct colossus (mine) vs HMF-grid spline (cosmolib).
Scan Om = fid + k*h, fit a low-order polynomial per bin, compare derivatives and Fisher (config b)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/boris/GitHub/sz-poisson-blending/analysis/cosmo/bias")
import vmodel as V, cosmolib as CL
fsky = 0.4; Mo = CL.NaiveModel(5.0)
ks = np.arange(-8, 9); h = 1e-3
def scan(fun, fid):
    out = []
    for k in ks:
        t = fid.copy(); t[0] += k * h; out.append(fun(t).ravel())
    return np.array(out)
Sm = scan(lambda t: V.counts(t, 5.0, fsky), V.FID); St = scan(lambda t: Mo.counts(t, fsky), CL.FID)
for name, S in [("mine(direct colossus)", Sm), ("theirs(grid spline)", St)]:
    tot = S.sum(1)
    c = np.polyfit(ks * h, np.log(tot), 4)
    resid = np.log(tot) - np.polyval(c, ks * h)
    print(f"{name}: total lnN vs Om quartic-fit rms resid {resid.std():.2e}; dlnN/dOm(fid) = {c[-2]:.4f}")
    # per-bin smooth derivative from cubic fit over +-8e-3
    D = np.array([np.polyfit(ks * h, S[:, j], 3)[-2] for j in range(S.shape[1])])
    resb = np.array([np.std(S[:, j] - np.polyval(np.polyfit(ks * h, S[:, j], 3), ks * h)) / max(S[8, j], 1e-30) for j in range(S.shape[1])])
    print(f"   per-bin rel. residual about a cubic in Om (bins mu>10): median {np.median(resb[S[8]>10]):.1e} max {resb[S[8]>10].max():.1e}")
    if name.startswith("mine"): Dm = D
    else: Dt = D
mu_m = V.counts(V.FID, 5.0, fsky).ravel(); mu_t = Mo.counts(CL.FID, fsky).ravel()
print("smooth dN/dOm: max|mine-theirs|/max|theirs| =", np.abs(Dm - Dt).max() / np.abs(Dt).max())
# relative per-bin derivative difference
sel = mu_t > 10
print("per-bin dlnN/dOm diff (mine - theirs) for mu>10: rms", np.std((Dm / mu_m - Dt / mu_t)[sel]), " max", np.abs((Dm / mu_m - Dt / mu_t)[sel]).max())
def jcol(fun, fid, i, st):
    tp = fid.copy(); tm = fid.copy(); tp[i] += st; tm[i] -= st
    return ((fun(tp) - fun(tm)) / (2 * st)).ravel()
fm = lambda t: V.counts(t, 5.0, fsky); ft = lambda t: Mo.counts(t, fsky)
J_s8m = jcol(fm, V.FID, 1, 5e-4); J_bm = jcol(fm, V.FID, 2, 2e-3)
J_s8t = jcol(ft, CL.FID, 1, 4e-4); J_bt = jcol(ft, CL.FID, 2, 1e-3)
g = np.array([0.5 * V.S8(*V.FID[:2]) / V.FID[0], V.S8(*V.FID[:2]) / V.FID[1], 0])
def sg(J, mu):
    F = J.T @ (J / mu[:, None]); C = np.linalg.inv(F); return np.round(np.sqrt(np.diag(C)), 5), round(float(np.sqrt(g @ C @ g)), 5)
print("config b Fisher sig (Om,s8,omb), S8:")
print("  mine, smooth Om-derivative     :", sg(np.stack([Dm, J_s8m, J_bm], 1), mu_m))
print("  theirs, smooth Om-derivative   :", sg(np.stack([Dt, J_s8t, J_bt], 1), mu_t))
print("  mixed: my Om-deriv, their s8/omb:", sg(np.stack([Dm, J_s8t, J_bt], 1), mu_t))
print("  mixed: their Om-deriv, my s8/omb:", sg(np.stack([Dt, J_s8m, J_bm], 1), mu_m))
np.savez("/Users/boris/GitHub/sz-poisson-blending/verify_cosmo/cosmo-bias/physics/v03_scan.npz", Sm=Sm, St=St, ks=ks, h=h)
