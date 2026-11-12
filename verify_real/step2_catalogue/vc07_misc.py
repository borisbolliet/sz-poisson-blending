"""vc07: cheap independent checks of step-2 inputs (lens catalogue).
(1) map-sigma ratio r_i = sqrt(1 + var_subthr): Campbell variance of the sub-threshold neighbours (own q_B < 5),
    var/sigma_A^2 = sum_B n_B <Y_B^2 1(q_B < 5)> int rho^2 d^2s / sigma_A^2, with MY population integral (colossus
    directly, my (z, M) grid, the lognormal integral in closed form) and the derivation's exact int rho^2 (P table);
    vs the build's r_i (inputs.npz) at its nodes inside 0.25-80'.  Low-pressure population: Y x c, same set rule.
(2) all-halo confusion variance at theta_A = 1.0069' for S8 = 0.75 / 0.825 / 0.87 (sigma8 at fixed Om) [claim C17].
(3) Poisson tSZ power D_3000 at 143 GHz, fiducial and low pressure ((1-b) = 0.6, Y x c), from one fresh sky, with the
    derivation's g(u) (2D FFT template transform) [claim C18].
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import vclib as V
import numpy as np
from scipy.special import ndtr
from scipy.interpolate import CubicSpline
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf

t0 = time.time()
I = np.load(f"{V.ROOT}/data_real/step2/inputs.npz")
P = {f: np.load(f"{V.DERIV}/rho_table_{f}.npz") for f in ("baseline", "goal")}
TH = P["baseline"]["th"]
lth0, dlth = np.log(TH[0]), np.log(TH[1] / TH[0])
C_KMS = 299792.458
ZG = np.linspace(0.01, 3.0, 900); LMG = np.linspace(np.log(1e13), np.log(1e16), 700)
wz = np.full(len(ZG), ZG[1] - ZG[0]); wz[[0, -1]] *= 0.5
wm = np.full(len(LMG), LMG[1] - LMG[0]); wm[[0, -1]] *= 0.5
SL = 0.173


def population(s8):
    """n(z, M) dz dlnM / A_sky [per arcmin^2], Y_med [arcmin^2], theta500 [arcmin] on my grid (colossus direct)."""
    c = cosmology.setCosmology("planck18", sigma8=float(s8), persistence="")
    h = c.h; M = np.exp(LMG)
    chi = c.comovingDistance(0.0, ZG) / h; DA = chi / (1 + ZG); E = c.Ez(ZG)
    dV = (C_KMS / c.H0) / E * chi ** 2 * 4 * np.pi
    n = np.array([mf.massFunction(M * h, float(z), mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3 for z in ZG]) * dV[:, None]
    n *= wz[:, None] * wm[None, :] / V.A_SKY_AM2
    Y = 1e-4 * 10 ** (-0.19) * E[:, None] ** (2 / 3) * (0.8 * M[None, :] / 6e14) ** 1.79 / DA[:, None] ** 2 / V.AM ** 2
    rhoc = c.rho_c(ZG) * 1e9 * h ** 2
    th = (3 * M[None, :] / (4 * np.pi * 500 * rhoc[:, None])) ** (1 / 3) / DA[:, None] / V.AM
    return n, Y, th


def node_spread(th, w):
    f = np.clip((np.log(th) - lth0) / dlth, 0, len(TH) - 1.000001); i = f.astype(int); a = f - i
    W = np.zeros(len(TH)); np.add.at(W, i.ravel(), (w * (1 - a)).ravel()); np.add.at(W, (i + 1).ravel(), (w * a).ravel())
    return W


def var_at(filt, W2, thA):
    Pt, sig = P[filt]["P"], P[filt]["sigY"]
    v_nodes = (Pt * W2[None, :]).sum(1) / sig ** 2
    return np.interp(np.log(thA), np.log(TH), v_nodes)


n, Y, th = population(0.8102)
print("(1) map-sigma ratio r_i = sqrt(1 + var_subthr) (Campbell, my population integral, derivation's int rho^2)")
for run, c, key in (("baseline", 1.0, "baseline_r_i"), ("goal", 1.0, "goal_r_i"), ("baseline", V.C_LOW, "lowP_r_i")):
    Ym = c * Y
    sy = V.sigma_Y(run, th)
    L = np.log(5 * sy / Ym)                                          # q_B = Ym e^{sl x} < 5  <=>  x < L / sl
    W2 = node_spread(th, n * Ym ** 2 * np.exp(2 * SL ** 2) * ndtr(L / SL - 2 * SL))
    thn = I["baseline_th"]
    inside = (thn >= 0.25) & (thn <= 80.0)
    r_mine = np.sqrt(1 + var_at(run, W2, thn))
    r_b = I[key]
    d = r_mine[inside] / r_b[inside] - 1
    lab = "lowP" if c != 1.0 else run
    print(f"  [{lab:8s}] build r_i range {r_b.min():.4f}-{r_b.max():.4f} (48 nodes 0.1-139'); mine at the {inside.sum()} nodes in 0.25-80': "
          f"{r_mine[inside].min():.4f}-{r_mine[inside].max():.4f}; mine/build - 1: max |.| {np.abs(d).max():.1e}; "
          f"var_sub mine/build - 1 at 0.5, 1, 4, 16': " + " ".join(
              f"{np.interp(np.log(t), np.log(thn), (r_mine**2-1)/(r_b**2-1)-1):+.3f}" for t in (0.5, 1.0, 4.0, 16.0)), flush=True)
    if run == "goal":
        print(f"             goal max r_i: build {r_b.max():.4f}, mine (0.25-80') {r_mine[inside].max():.4f}")
# derivation's catalogue values (one sky) for comparison
cm = np.load(f"{V.DERIV}/confusion_mine.npz")
rb_d = np.sqrt(1 + cm["baseline_subthr"]); rb_b = np.interp(np.log(cm["thA"]), np.log(I["baseline_th"]), I["baseline_r_i"])
print(f"  [baseline] derivation's catalogue sub-threshold variance (one sky, v06): r_i / build - 1 max |.| {np.abs(rb_d/rb_b-1).max():.1e}")

print("\n(2) all-halo confusion variance / sigma_Y^2 at theta_A = 1.0069' (baseline), S8 = 0.75 / 0.825 / 0.87 at Om = 0.3111"
      " [build: 0.147 / 0.334 / 0.522]")
for S8 in (0.75, 0.8249, 0.87):
    s8 = S8 / np.sqrt(0.3111 / 0.3)
    n2, Y2, th2 = (n, Y, th) if abs(S8 - 0.8249) < 1e-3 else population(s8)
    W2 = node_spread(th2, n2 * Y2 ** 2 * np.exp(2 * SL ** 2))
    print(f"  S8 = {S8:.4f} (sigma8 = {s8:.4f}): var / sigma_Y^2 = {var_at('baseline', W2, 1.0069):.4f}", flush=True)

print("\n(3) Poisson tSZ power at l = 3000, 143 GHz, from one fresh sky (seed 4242) [build: 9.07 fiducial, 3.24 low pressure]")
from szsim import sampler
cat = sampler.draw_catalog(4242, fields=("Y500", "theta500"))
g = np.load(f"{V.DERIV}/g_of_u_2dfft.npz")
gs = CubicSpline(g["u"], g["g"])
Lm = 3000.0
S2 = 0.0
for a in range(0, len(cat["Y500"]), 4_000_000):
    y = cat["Y500"][a:a + 4_000_000].astype(np.float64) * V.AM ** 2
    u = Lm * cat["theta500"][a:a + 4_000_000].astype(np.float64) * V.AM
    gg = np.where(u <= g["u"][-1], gs(np.minimum(u, g["u"][-1])), 0.0)
    S2 += np.sum(y * y * gg * gg)
Cl = S2 / (4 * np.pi)
hx = 6.62607015e-34 * 143e9 / (1.380649e-23 * 2.7255)
fnu = 2.7255e6 * (hx / np.tanh(hx / 2) - 4)
D = Lm * (Lm + 1) * Cl / (2 * np.pi) * fnu ** 2
print(f"  sky with {len(cat['Y500']):,} halos: 1e12 l(l+1)C_l/2pi = {1e12*Lm*(Lm+1)*Cl/(2*np.pi):.4f}; D_3000(143 GHz) = {D:.2f} uK^2 "
      f"(fiducial), {D*V.C_LOW**2:.2f} uK^2 (Y x {V.C_LOW:.5f}) = {D*V.C_LOW**2/3.42:.2f} x SPT 3.42; build HMF values "
      f"{float(I['D3000_hmf_fid']):.2f} / {float(I['D3000_hmf_low']):.2f}")
print(f"total {time.time()-t0:.0f}s")
