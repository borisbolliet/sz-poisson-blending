"""B1 check: MC pair fractions on the 3 stored full skies (own KD-tree on detections) vs Poisson formula
with the verifier's own Sigma_det (v02), and the size-aware merge radius with own s_infl (v01 kernel).
Analytic f_phys from own colossus population (detection-weighted theta distribution)."""
import numpy as np, healpy as hp
from scipy.spatial import cKDTree
from scipy.special import ndtr
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import v01_kernel as K
cos = cosmology.setCosmology("planck18"); h = cos.h; ARC = np.pi / 10800
d0 = np.load("v01_kernel.npz"); TH, SIG = d0["TH"], d0["SIG"]
sig = lambda t: np.exp(np.interp(np.log(t), np.log(TH), np.log(SIG)))
# s_infl(theta) on a grid, from exact rho(th,th,s)
tg = np.geomspace(0.05, 70, 60); sinf = []
for t in tg:
    s = np.linspace(0, 3 * t + 4, 3001); r = K.rho_exact(t, t, s, K.sigY(t)[0])
    d2 = np.gradient(np.gradient(r, s), s); i = np.nonzero(np.diff(np.sign(d2[5:])))[0][0] + 5; sinf.append(s[i])
sinf = np.array(sinf); s_of = lambda t: np.interp(np.log(t), np.log(tg), sinf)
print("s_infl at 0.3,1,3,10':", np.round(s_of(np.array([0.3, 1, 3, 10])), 3))
# population (same as v02)
def gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n); return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w
lz, wlz = gl(np.log(0.01), np.log(3.0), 300); Z = np.exp(lz); WZ = wlz * Z
lm, wm = gl(np.log(1e13), np.log(1e16), 200); M = np.exp(lm)
gx, gw = np.polynomial.hermite_e.hermegauss(30); gw /= gw.sum()
TT, WW, QQ = [], [], []
for zi, wz in zip(Z, WZ):
    dn = mass_function.massFunction(M * h, zi, mdef="500c", model="tinker08", q_out="dndlnM") * h ** 3
    chi = cos.comovingDistance(0.0, zi) / h; DA = chi / (1 + zi)
    w = dn * 299792.458 / cos.H0 / cos.Ez(zi) * chi ** 2 * wm * wz * 4 * np.pi
    Y = 1e-4 * 10 ** -0.19 * cos.Ez(zi) ** (2 / 3) * (0.8 * M / 6e14) ** 1.79 / DA ** 2 / ARC ** 2
    R = (3 * M / (4 * np.pi * 500 * cos.rho_c(zi) * 1e9 * h ** 2)) ** (1 / 3); th = R / DA / ARC
    TT.append(th); WW.append(w); QQ.append(Y / sig(th))
TT, WW, QQ = map(np.concatenate, (TT, WW, QQ))
QC = (4, 5, 6, 7)
an = {}
for qc in QC:
    pd = sum(ww * ndtr(QQ * np.exp(0.173 * e) - qc) for e, ww in zip(gx, gw))
    wd = WW * pd; Nd = wd.sum()
    # f_phys = < 1 - exp(-sum_B n_B A_cap(sA+sB)) >_A ; n_B per sr
    sA = s_of(TT)
    o = np.argsort(sA); sAo = sA[o]; wdo = wd[o]
    grid = np.linspace(sA.min(), sA.max(), 400)
    lamg = np.array([np.sum(wdo / (4 * np.pi) * 2 * np.pi * (1 - np.cos((g + sAo) * ARC))) for g in grid])
    fphys = np.sum(wd * (1 - np.exp(-np.interp(sA, grid, lamg)))) / Nd
    an[qc] = (Nd, fphys)
    print(f"q_cut={qc}: N_det {Nd:,.1f}; f_phys analytic {fphys:.4e}")
# MC
res = {}
for seed in (1, 2, 3):
    f = np.load(f"/Users/boris/GitHub/sz-poisson-blending/data/catalogs/cat_M1e13_seed{seed}.npz")
    qo = f["q_obs"]; sel = np.nonzero(qo > 4)[0]; del qo
    ra = f["ra"][sel].astype(float); dec = f["dec"][sel].astype(float); th = f["theta500"][sel].astype(float); qo = f["q_obs"][sel].astype(float)
    v = hp.ang2vec(ra, dec, lonlat=True)
    lab = hp.vec2pix(2, v[:, 0], v[:, 1], v[:, 2]) + 48 * (seed - 1)
    for qc in QC:
        s = qo > qc; vv = v[s]; tree = cKDTree(vv); sA = s_of(th[s])
        out = {}
        for tb in (1, 2, 3, 5):
            ch = 2 * np.sin(tb * ARC / 2)
            n = np.array([len(x) - 1 for x in tree.query_ball_point(vv, ch)])
            out[tb] = (n >= 1, n >= 2)
        pairs = tree.query_pairs(2 * np.sin((2 * sA.max()) * ARC / 2), output_type="ndarray")
        sep = 2 * np.arcsin(np.linalg.norm(vv[pairs[:, 0]] - vv[pairs[:, 1]], axis=1) / 2) / ARC
        ok = sep < sA[pairs[:, 0]] + sA[pairs[:, 1]]
        flag = np.zeros(len(vv), bool); flag[pairs[ok, 0]] = True; flag[pairs[ok, 1]] = True
        out["phys"] = (flag, flag)
        res.setdefault(qc, []).append((out, lab[s]))
def jk(x, lab):
    a = np.bincount(lab, weights=x, minlength=144); b = np.bincount(lab, minlength=144)
    r = a.sum() / b.sum(); ok = b > 0; rj = (a.sum() - a[ok]) / (b.sum() - b[ok]); n = ok.sum()
    return r, np.sqrt((n - 1) / n * ((rj - rj.mean()) ** 2).sum())
zs = []
for qc in QC:
    Nd = an[qc][0]; line = f"q_cut={qc}:"
    lab = np.concatenate([l for _, l in res[qc]])
    for tb in (1, 2, 3, 5, "phys"):
        f1 = np.concatenate([o[tb][0] for o, _ in res[qc]]).astype(float)
        r, e = jk(f1, lab)
        if tb == "phys":
            a = an[qc][1]
        else:
            mu = Nd / (4 * np.pi) * 2 * np.pi * (1 - np.cos(tb * ARC)); a = 1 - np.exp(-mu)
        z = (r - a) / e; zs.append(z)
        line += f" {tb}': MC {r:.4e}±{e:.1e} an {a:.4e} ({z:+.2f})"
        if tb in (3, 5) and qc == 5:
            f2 = np.concatenate([o[tb][1] for o, _ in res[qc]]).astype(float)
            r2, e2 = jk(f2, lab); a2 = 1 - np.exp(-mu) * (1 + mu)
            line += f" [>=2: MC {r2:.3e}±{e2:.1e} an {a2:.3e} ({(r2-a2)/e2:+.2f})]"
    print(line)
zs = np.array(zs); print(f"{len(zs)} B1 z-scores rms {np.sqrt(np.mean(zs**2)):.2f} max {np.abs(zs).max():.2f}")
