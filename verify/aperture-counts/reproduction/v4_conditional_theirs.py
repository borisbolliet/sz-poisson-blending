"""V4: re-analysis of the authors' 40 stored skies (no new sampling).
(a) Detected-centred P(>=1 other detected within theta) vs (i) unconditional Poisson with Nbar and (ii) the
    exact conditional law given each sky's realised N: P(>=1 | N) = 1 - (1-p)^(N-1), p = (1-cos theta)/2.
(b) Same for pair counts: E[P | N] = N(N-1) p / 2.
(c) Correlation structure of the 'chi2/dof' tests: correlation of z(P0) and z(P1) and across nested theta.
(d) C12: 6-sky no-floor z for det5|M3e13 P(N=0) at 10'.
(e) Own p3 MC using rejection sampling in a square (different from the authors' polar method)."""
import glob, json, os
import numpy as np
from scipy import stats

D = "/Users/boris/GitHub/sz-poisson-blending/data/q1_aperture"
NB = json.load(open(os.path.join(D, "nbar.json")))
S = [np.load(f, allow_pickle=True).item() for f in sorted(glob.glob(os.path.join(D, "fullsky", "sky_*.npy")))]
R = len(S)
THETAS = np.array([0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60.0])
FINE = np.unique(np.concatenate([THETAS, np.geomspace(0.2, 240, 91)]))
iT = np.array([np.argmin(abs(FINE - t)) for t in THETAS])
AM = np.pi / 180 / 60
p = np.sin(THETAS * AM / 2) ** 2

print("(a) detected-centred P(>=1), MC vs unconditional and vs conditional-on-realised-N")
for q in [4, 5, 6]:
    key = f"det{q}|det{q}"
    F = np.array([s["cdf"][key][0, iT] for s in S])          # (R, nT)
    N = np.array([s["npop"][f"det{q}"] for s in S], float)
    unc = 1 - np.exp(-NB[f"det{q}"] * p)
    cond = np.mean(1 - (1 - p[None, :]) ** (N[:, None] - 1), 0)
    # error: sky-to-sky scatter of (F - cond_s): removes the N-fluctuation component
    Fc = F - (1 - (1 - p[None, :]) ** (N[:, None] - 1))
    e_unc = F.std(0, ddof=1) / np.sqrt(R); e_c = Fc.std(0, ddof=1) / np.sqrt(R)
    for j in [2, 4, 6, 8, 9, 10]:
        print(f"  det{q} {THETAS[j]:4.0f}': MC {F[:,j].mean():.5f}  unc {unc[j]:.5f} z={(F[:,j].mean()-unc[j])/e_unc[j]:+.2f}"
              f"  cond {cond[j]:.5f} z={Fc[:,j].mean()/e_c[j]:+.2f}  (rel dev from cond {Fc[:,j].mean()/cond[j]:+.4f})")

print("\n(b) pairs: MC vs E[P|N] = N(N-1)p/2 per sky")
for q in [4, 5, 6]:
    P = np.array([s[f"pairs_det{q}"] for s in S]); N = np.array([s["npop"][f"det{q}"] for s in S], float)
    Ec = N[:, None] * (N[:, None] - 1) * p[None, :] / 2
    d = P - Ec
    for j in [4, 6, 8, 11]:
        print(f"  det{q} {THETAS[j]:4.0f}': mean(P-E[P|N]) = {d[:,j].mean():+8.2f} +- {d[:,j].std(ddof=1)/np.sqrt(R):.2f}"
              f" (z={d[:,j].mean()/(d[:,j].std(ddof=1)/np.sqrt(R)):+.2f}); rel {d[:,j].mean()/Ec[:,j].mean():+.4f}")

print("\n(c) correlation of per-sky P(N=0) and P(N=1) estimates (rand|det5), and across nested theta")
F = np.array([s["cdf"]["rand|det5"][:, iT] for s in S])
P0 = 1 - F[:, 0]; P1 = F[:, 0] - F[:, 1]
for j in [2, 4, 6, 8]:
    print(f"  theta={THETAS[j]}': corr(P0,P1) over skies = {np.corrcoef(P0[:,j], P1[:,j])[0,1]:+.3f}")
print(f"  corr(P1 at 1', P1 at 2') = {np.corrcoef(P1[:,2], P1[:,4])[0,1]:+.3f};  corr(P1 at 2', 3') = {np.corrcoef(P1[:,4], P1[:,5])[0,1]:+.3f}")
# simulate how a 'diagonal chi2' of strongly correlated bins is distributed: use the empirical correlation of the
# 64 used bins of rand|det5 is not available per-bin; instead illustrate: z(P0) = -z(P1) when P(N>=2)~0
print("  -> at theta<=3' z(P0) ~ -z(P1): each fluctuation enters the diagonal chi2 twice")

print("\n(d) C12: det5|M3e13 P(N=0) at 10', first 6 skies, no binomial floor")
key = "det5|M3e13"; j = 8
P0 = np.array([1 - s["cdf"][key][0, iT[j]] for s in S[:6]]); nc = np.array([s["ncen"][key] for s in S[:6]])
an = np.exp(-NB["M3e13"] * p[j])
print(f"  per-sky events: {np.round(P0*nc).astype(int)}; analytic P0={an:.4e}; MC={P0.mean():.4e}; "
      f"z_nofloor={(P0.mean()-an)/(P0.std(ddof=1)/np.sqrt(6)):+.2f}")

print("\n(e) p3 by rejection sampling in the square [-1,1]^2 (seed 424242)")
rng = np.random.default_rng(424242)
tot = hit = 0
for _ in range(10):
    u = rng.uniform(-1, 1, (4, 2_000_000))
    ok = (u[0] ** 2 + u[1] ** 2 < 1) & (u[2] ** 2 + u[3] ** 2 < 1)
    d2 = (u[0] - u[2]) ** 2 + (u[1] - u[3]) ** 2
    tot += ok.sum(); hit += (ok & (d2 < 1)).sum()
p3 = hit / tot; e = np.sqrt(p3 * (1 - p3) / tot); ex = 1 - 3 * np.sqrt(3) / (4 * np.pi)
print(f"  p3 MC = {p3:.6f} +- {e:.6f}; exact {ex:.6f}; z = {(p3-ex)/e:+.2f}  (n={tot})")
