"""V6: aggregate the independent MC (v3) and compare to the Poisson predictions, using BOTH my independent Nbar
(nbar_indep.json, v1) and the authors' Nbar (nbar.json).  Errors: sky-to-sky std / sqrt(R), floored at the
binomial error of the analytic P with ncen*R trials."""
import glob, json, os
import numpy as np
from scipy import stats

H = os.path.dirname(os.path.abspath(__file__))
S = [np.load(f, allow_pickle=True).item() for f in sorted(glob.glob(os.path.join(H, "indep_mc", "sky_*.npy")))]
R = len(S)
NBi = json.load(open(os.path.join(H, "nbar_indep.json")))
NBa = json.load(open("/Users/boris/GitHub/sz-poisson-blending/data/q1_aperture/nbar.json"))
TH = np.array([0.5, 1, 2, 5, 10.0]); AM = np.pi / 180 / 60; p = np.sin(TH * AM / 2) ** 2
P3 = 1 - 3 * np.sqrt(3) / (4 * np.pi)
mapi = {"d4": "det4", "d5": "det5", "d6": "det6", "m14": "M1e14", "m3": "M3e13", "d5z": "det5_z1.0-3.0"}
print(f"R = {R} independent skies (seeds {S[0]['seed']}..{S[-1]['seed']})")

print("\n== population sizes ==")
for k, lab in mapi.items():
    n = np.array([s["npop"][k] for s in S], float)
    for tag, NB in [("mine", NBi), ("authors", NBa)]:
        ref = NB[lab] if lab in NB else None
        if ref is None:
            continue
        print(f"  {lab:14s} MC mean {n.mean():12.1f}  Nbar[{tag}] {ref:12.1f}  z={(n.mean()-ref)/np.sqrt(ref/R):+.2f}"
              f"  var/mean={n.var(ddof=1)/ref:.2f}+-{np.sqrt(2/(R-1)):.2f}")
Ntot = np.array([s["Ntot"] for s in S], float)
print(f"  M1e13 total    MC mean {Ntot.mean():12.1f}  Nbar[authors] {NBa['M1e13']:.1f} z={(Ntot.mean()-NBa['M1e13'])/np.sqrt(NBa['M1e13']/R):+.2f}")

print("\n== P(>=k) neighbours within theta, MC vs Poisson(mu = Nbar (1-cos)/2) ==")
allz = []
for key in ["rand|d5", "rand|m14", "rand|m3", "d5|d5", "d4|d4", "d6|d6", "d5|m14", "d5|m3", "d5|d5z"]:
    c, pp = key.split("|")
    F = np.array([s[key] for s in S])           # (R, 3, T)
    nc = np.mean([s[f"n_{key}"] for s in S])
    m = F.mean(0); sd = F.std(0, ddof=1) / np.sqrt(R)
    for tag, NB in [("mine", NBi), ("authors", NBa)]:
        mu = NB[mapi[pp]] * p
        A = np.array([stats.poisson.sf(k - 1, mu) for k in (1, 2, 3)])
        e = np.maximum(sd, np.sqrt(A * (1 - A) / (nc * R)))
        ok = A * nc * R >= 10
        z = np.where(ok, (m - A) / e, np.nan)
        if tag == "mine":
            allz.append(z[ok])
        line = "  ".join(f"{TH[j]:g}': P1 {m[0,j]:.4e}/{A[0,j]:.4e}[{z[0,j]:+.1f}] P2 {m[1,j]:.3e}/{A[1,j]:.3e}[{z[1,j]:+.1f}]"
                         for j in range(len(TH)))
        print(f"  {key:9s} [{tag:7s}] {line}")
allz = np.concatenate(allz)
print(f"  all (mine Nbar): n={len(allz)} rms z={np.sqrt(np.mean(allz**2)):.3f} max|z|={np.abs(allz).max():.2f}")

print("\n== Palm: detected-centred minus random-centred P(>=1), paired per sky (relative) ==")
for c in ["d5", "d4", "d6"]:
    pass
D = np.array([s["d5|d5"][0] - s["rand|d5"][0] for s in S]); Fr = np.array([s["rand|d5"][0] for s in S]).mean(0)
nd = np.mean([s["n_d5|d5"] for s in S]); A = 1 - np.exp(-NBi["det5"] * p)
e = np.maximum(D.std(0, ddof=1) / np.sqrt(R), np.sqrt(A * (1 - A) * (1 / nd + 1 / 1e5) / R))
print("  det5: " + ", ".join(f"{TH[j]:g}': {D[:,j].mean()/Fr[j]:+.4f}+-{e[j]/Fr[j]:.4f}" for j in range(len(TH))))

print("\n== detected-centred P(>=1) at 10' conditional on realised N (as in V4) ==")
for k in ["d4", "d5", "d6"]:
    F = np.array([s[f"{k}|{k}"][0] for s in S]); N = np.array([s["npop"][k] for s in S], float)
    Fc = F - (1 - (1 - p[None, :]) ** (N[:, None] - 1))
    print(f"  {mapi[k]}: " + "  ".join(f"{TH[j]:g}': rel dev {Fc[:,j].mean()/F[:,j].mean():+.4f} z={Fc[:,j].mean()/(Fc[:,j].std(ddof=1)/np.sqrt(R)):+.2f}"
                                      for j in range(len(TH))))

print("\n== survey-wide pairs: E = Nbar mu/2, Var = E(1+2mu) ==")
for k in ["d4", "d5", "d6"]:
    P = np.array([s[f"pairs_{k}"] for s in S])
    for tag, NB in [("mine", NBi), ("authors", NBa)]:
        mu = NB[mapi[k]] * p; E = NB[mapi[k]] * mu / 2; V = E * (1 + 2 * mu)
        print(f"  {mapi[k]} [{tag:7s}] " + "  ".join(f"{TH[j]:g}': {P[:,j].mean():.2f}/{E[j]:.2f} z={(P[:,j].mean()-E[j])/np.sqrt(V[j]/R):+.2f} V={P[:,j].var(ddof=1)/V[j]:.2f}"
                                                for j in range(len(TH))))
print(f"  (Var ratio error +-{np.sqrt(2/(R-1)):.2f})")

print("\n== det5 triples (all sep < theta): E = Nbar mu^2 p3/6 ==")
T = np.array([s["tri_d5"] for s in S])
for j, t in enumerate([2, 5, 10.0]):
    mu = NBi["det5"] * np.sin(t * AM / 2) ** 2; E = NBi["det5"] * mu ** 2 * P3 / 6
    print(f"  {t:g}': MC {T[:,j].mean():.3f}+-{T[:,j].std(ddof=1)/np.sqrt(R):.3f}  E {E:.3f}  z={(T[:,j].mean()-E)/max(T[:,j].std(ddof=1)/np.sqrt(R), np.sqrt(E/R)):+.2f}")
