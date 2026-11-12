"""vc06: exact check of the white filter's subtracted mean mu(theta_A) (convention c = 1).
For the uncompensated white filter int d^2s rho_AB = sigma_A^2 RATIO^2 / N_w for every theta_B (the l = 0 mode), so
   mu(theta_A) = Ybar_tot * sigma_A * RATIO^2 / N_w,   Ybar_tot = sum_B n_B <Y_B> per arcmin^2.
Ybar_tot from the sampler's own HMF table (1201 x 1201 trapezoid, <Y> = Y_med e^{s^2/2}).  Compared with
 (a) my table-based expected mean (vc01: derivation's rho table, support s < 5thA + 5thB + 30'),
 (b) the build's white mu (inputs.npz, s09 rate histograms), (c) the realised skies (vc01 logs).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import vclib as V
import numpy as np
from szsim import cosmo_hmf as ch, signal as sg

T = ch.default_table(); m = sg.SZModel()
wz = ch.HMFTable._weights(T.z, T.z[0], T.z[-1]); wM = ch.HMFTable._weights(T.lnM, T.lnM[0], T.lnM[-1])
M = np.exp(T.lnM)
Ytot = 0.0; Ntot = 0.0
for iz, z in enumerate(T.z):
    Ytot += np.sum(T.d2N[iz] * wz[iz] * wM * m.Y500_med(M, z)) * np.exp(0.5 * m.sigma_lnY ** 2)
    Ntot += np.sum(T.d2N[iz] * wz[iz] * wM)
Ybar = Ytot / V.A_SKY_AM2
Nw = 4e-6 ** 2                                     # y^2 arcmin^2
print(f"population (sampler's table): N = {Ntot:,.0f} halos, total Y = {Ytot:.6e} arcmin^2, Ybar = {Ybar:.6e} per arcmin^2")
I = np.load(f"{V.ROOT}/data_real/step2/inputs.npz")
E = np.load(f"{HERE}/out/expected_mean.npz")
for t in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    sA = float(V.sigma_Y("white", t))
    mu_exact = Ybar * sA * sg.RATIO_5R500 ** 2 / Nw
    mu_mine = np.interp(np.log(t), np.log(E["TH"]), E["num"][0]) / sA
    mu_build = np.interp(np.log(t), np.log(I["white_th"]), I["white_mu"])
    print(f"  theta_A = {t:5.1f}': mu exact {mu_exact:.5f} | mine (rho table, truncated) {mu_mine:.5f} ({mu_mine/mu_exact-1:+.2e}) | "
          f"build (s09) {mu_build:.5f} ({mu_build/mu_exact-1:+.2e})")
# derivation table DC integral vs exact at nodes
R3, TH, S, sig = V.load_rho3()
tw = np.empty_like(S); d = np.diff(S); tw[1:-1] = 0.5 * (d[:-1] + d[1:]); tw[0] = 0.5 * d[0]; tw[-1] = 0.5 * d[-1]
ring = 2 * np.pi * S * tw
for i in (20, 40, 60, 80, 100):
    DC_tab = (R3[i, :, :, 0].astype(float) * ring[None, :]).sum(1)          # full table support (800')
    DC_ex = sig["white"][i] ** 2 * sg.RATIO_5R500 ** 2 / Nw
    print(f"  derivation white rho table, theta_A node {TH[i]:.3f}': int rho d^2s / exact - 1 over theta_B nodes: "
          f"min {np.min(DC_tab/DC_ex-1):+.2e} max {np.max(DC_tab/DC_ex-1):+.2e}; table sigma / SZModel sigma - 1 = "
          f"{sig['white'][i]/float(V.sigma_Y('white', TH[i]))-1:+.2e}")
