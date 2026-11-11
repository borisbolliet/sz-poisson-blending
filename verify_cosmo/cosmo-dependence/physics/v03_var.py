"""Independent confusion variance at theta_A = 2.30': Var(delta) = sum_B n_B Y_B^2 int dA rho^2 1[x<X] / sigma_A^2,
x = rho Y_B/sigma_A.  rho(s) from szsim cross_response (real-space Hankel), HMF from colossus (own grid).
Log slopes in sigma8, Om, (1-b), sigma_lnY with central FD; share from M>1e14."""
import numpy as np
from colossus.lss import mass_function
from v_lib import cosmo, SZM, ARCMIN, C_KMS
thA = 2.30
sA = SZM.sigma_Y(thA)
lthB = np.linspace(np.log(0.03), np.log(150.0), 60)
lX = np.linspace(np.log(1e-12), np.log(1e3), 600)        # threshold on rho
V = np.zeros((len(lthB), len(lX))); V0 = np.zeros(len(lthB))
for i, lt in enumerate(lthB):
    tB = np.exp(lt); S = 5 * thA + 5 * tB + 3.0
    s = np.linspace(0, S, 3001)
    r = SZM.cross_response(thA, tB, s)
    dA = 2 * np.pi * s * np.gradient(s)
    V0[i] = (dA * r ** 2).sum()
    o = np.argsort(r); rs = r[o]; cum = np.concatenate([[0], np.cumsum((dA * r ** 2)[o])])
    V[i] = cum[np.searchsorted(rs, np.exp(lX), side="left")]   # int rho^2 over rho < X
print(f"theta_A={thA}, sigma_A={sA:.4e}; check rho(thA,thA,0)={SZM.cross_response(thA, thA, [0.0])[0]:.4f}")

def var(s8, Om, omb, slnY, nz=300, nM=241):
    c = cosmo(s8, Om); h = c.h
    z = np.linspace(0.01, 3.0, nz); wz = np.full(nz, z[1] - z[0]); wz[[0, -1]] *= .5
    lnM = np.linspace(np.log(1e13), np.log(1e16), nM); M = np.exp(lnM); wM = np.full(nM, lnM[1] - lnM[0]); wM[[0, -1]] *= .5
    chi = c.comovingDistance(0.0, z) / h; DA = chi / (1 + z); E = c.Ez(z)
    dV = C_KMS / c.H0 / E * chi ** 2; rhoc = c.rho_c(z) * 1e9 * h ** 2
    xg, wg = np.polynomial.hermite_e.hermegauss(16); wg /= wg.sum()
    out = dict(v500=0.0, v5=0.0, vall=0.0, v500_hi=0.0)
    for i in range(nz):
        n = mass_function.massFunction(M * h, z[i], mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3 * dV[i] * wz[i] * wM * ARCMIN ** 2  # per arcmin^2 (dV per sr)
        th = (3 * M / (4 * np.pi * 500 * rhoc[i])) ** (1 / 3) / DA[i] / ARCMIN
        Y = 1e-4 * 10 ** -0.19 * E[i] ** (2 / 3) * (omb * M / 6e14) ** 1.79 / DA[i] ** 2 / ARCMIN ** 2
        fb = np.clip((np.log(th) - lthB[0]) / (lthB[1] - lthB[0]), 0, len(lthB) - 1.000001); ib = fb.astype(int); wb = fb - ib
        for g in range(16):
            Yg = Y * np.exp(slnY * xg[g]); w = n * wg[g] * Yg ** 2 / sA ** 2
            for X, key in ((500.0, "v500"), (5.0, "v5")):
                lx = np.log(X * sA / Yg); fx = np.clip((lx - lX[0]) / (lX[1] - lX[0]), 0, len(lX) - 1.000001); ix = fx.astype(int); wx = fx - ix
                Vi = ((1 - wb) * ((1 - wx) * V[ib, ix] + wx * V[ib, ix + 1]) + wb * ((1 - wx) * V[ib + 1, ix] + wx * V[ib + 1, ix + 1]))
                out[key] += (w * Vi).sum()
                if key == "v500": out["v500_hi"] += (w * Vi * (M > 1e14)).sum()
            out["vall"] += (w * ((1 - wb) * V0[ib] + wb * V0[ib + 1])).sum()
    return out

fid = [0.81, 0.31, 0.8, 0.17]; st = [0.01, 0.01, 0.02, 0.03]
o = var(*fid)
print(f"fiducial: Var(x<500)={o['v500']:.4f} Var(x<5)={o['v5']:.4f} Var(all)={o['vall']:.4f}  frac M>1e14 (x<500)={o['v500_hi']/o['v500']:.4f}")
print("  stored (c02 grid, 2.30'): Var(x<500)=2.178 Var(x<5)=0.4961 frac_hi=0.9481")
for a, nm in enumerate(["sigma8", "Om", "1-b", "slnY"]):
    pm = [list(fid), list(fid)]; pm[0][a] -= st[a]; pm[1][a] += st[a]
    o0, o1 = var(*pm[0]), var(*pm[1]); dl = np.log(pm[1][a] / pm[0][a])
    print(f"  slope {nm:7s} Var(x<500) {np.log(o1['v500']/o0['v500'])/dl:6.3f}  Var(x<5) {np.log(o1['v5']/o0['v5'])/dl:6.3f}  "
          f"Var(all) {np.log(o1['vall']/o0['vall'])/dl:6.3f}  frac_hi {np.log((o1['v500_hi']/o1['v500'])/(o0['v500_hi']/o0['v500']))/dl:6.3f}")
