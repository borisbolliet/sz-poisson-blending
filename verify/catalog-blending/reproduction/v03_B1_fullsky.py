"""V03: independent B1 check: my own full-sky M>1e13 Poisson skies (own sampler, own sigma_Y), KD-tree pair
counts of single-cluster detections (q_obs > q_cut) vs Poisson formula with my own analytic Sigma_det;
own inflection radius s_infl and size-aware merge fraction."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr
from scipy.stats import poisson
import indep_model as im

QC = [4.0, 5.0, 6.0, 7.0]; THB = np.array([1.0, 2.0, 3.0, 5.0])
grid = im.hmf_grid()
ze, lme, dens = grid
cc = im.cell_counts(*grid)
zc = 0.5 * (ze[1:] + ze[:-1]); Mc = np.exp(0.5 * (lme[1:] + lme[:-1]))
Z, MM = np.meshgrid(zc, Mc, indexing="ij")
TH = im.theta500(MM, Z); Q0 = im.Ymed(MM, Z) / im.sigmaY_fast(TH)
xg, wg = np.polynomial.hermite_e.hermegauss(40); wg /= wg.sum()
AREA = 4 * np.pi / im.ARCMIN ** 2
# inflection radius of rho(th,th,s)
def s_infl(t):
    s = np.linspace(0.0, 10 * t + 5, 4001)
    r = im.rho(t, t, s)
    d2 = np.gradient(np.gradient(r, s), s)
    return s[np.nonzero(d2[5:] > 0)[0][0] + 5]
tt = np.geomspace(0.25, 60, 50)
SI = np.array([s_infl(t) for t in tt])
print("s_infl mine:", {t: round(float(np.exp(np.interp(np.log(t), np.log(tt), np.log(SI)))), 3) for t in (0.3, 1, 3, 10)})
sinf = lambda th: np.exp(np.interp(np.log(th), np.log(tt), np.log(SI)))
an = {}
for qc in QC:
    P = sum(w * ndtr(Q0 * np.exp(im.SIG_LNY * x) - qc) for x, w in zip(xg, wg))
    W = cc * P * AREA
    Nd = W.sum(); sig_deg = Nd / 41252.96
    Acap = 2 * np.pi * (1 - np.cos(THB * im.ARCMIN)) / (4 * np.pi) * Nd
    fk1 = 1 - np.exp(-Acap); fk2 = poisson.sf(1, Acap)
    # size-aware
    edges = np.geomspace(0.2, 80, 301); h, _ = np.histogram(TH.ravel(), edges, weights=W.ravel()); thc = np.sqrt(edges[1:] * edges[:-1])
    sA = sinf(thc); dm = sA[:, None] + sA[None, :]
    lam = (h[None, :] / (4 * np.pi) * 2 * np.pi * (1 - np.cos(dm * im.ARCMIN))).sum(1)
    fph = (h * (1 - np.exp(-lam))).sum() / h.sum()
    an[qc] = (Nd, fk1, fk2, fph)
    print(f"[analytic mine] q_cut={qc}: N_det={Nd:,.0f} Sigma_det={sig_deg:.4f}/deg^2 P(>=1)={np.array2string(fk1, precision=5)} P(>=2)={np.array2string(fk2, precision=3)} B1phys={fph:.5f}", flush=True)

NSKY = int(sys.argv[1]) if len(sys.argv) > 1 else 6
res = {qc: [] for qc in QC}
t0 = time.time()
cdf = np.cumsum(cc.ravel()); cdf /= cdf[-1]; nm = len(lme) - 1
for isky in range(NSKY):
    rng = np.random.default_rng(555_000 + isky)
    ntot = rng.poisson(cc.sum() * AREA)
    keep = []
    for a in range(0, ntot, 3_000_000):
        n = min(3_000_000, ntot - a)
        k = np.searchsorted(cdf, rng.random(n)); iz, imm = k // nm, k % nm
        z = ze[iz] + rng.random(n) * (ze[iz + 1] - ze[iz]); M = np.exp(lme[imm] + rng.random(n) * (lme[imm + 1] - lme[imm]))
        th = im.theta500(M, z); Y = im.Ymed(M, z) * np.exp(im.SIG_LNY * rng.standard_normal(n))
        qo = Y / im.sigmaY_fast(th) + rng.standard_normal(n)
        s = qo > 4
        ra = 360 * rng.random(n); dec = np.degrees(np.arcsin(2 * rng.random(n) - 1))
        keep.append(np.column_stack([ra[s], dec[s], th[s], qo[s]]))
    C = np.concatenate(keep)
    for qc in QC:
        c = C[C[:, 3] > qc]
        ra, dec = np.radians(c[:, 0]), np.radians(c[:, 1])
        v = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        tr = cKDTree(v)
        ch = lambda t: 2 * np.sin(0.5 * t * im.ARCMIN)
        cnts = np.array([tr.query_ball_point(v, ch(t), return_length=True) - 1 for t in THB])
        sA = sinf(c[:, 2])
        pr = tr.query_pairs(ch(2 * sA.max()), output_type="ndarray")
        d = 2 * np.arcsin(np.linalg.norm(v[pr[:, 0]] - v[pr[:, 1]], axis=1) / 2) / im.ARCMIN
        mg = d < sA[pr[:, 0]] + sA[pr[:, 1]]
        fl = np.zeros(len(c), bool); fl[pr[mg, 0]] = True; fl[pr[mg, 1]] = True
        res[qc].append([len(c)] + list((cnts >= 1).mean(1)) + list((cnts >= 2).mean(1)) + [fl.mean()])
    print(f"sky {isky} N={ntot} {time.time()-t0:.1f}s", flush=True)
print()
for qc in QC:
    r = np.array(res[qc]); m = r.mean(0); e = r.std(0, ddof=1) / np.sqrt(len(r))
    Nd, fk1, fk2, fph = an[qc]
    print(f"q_cut={qc}: N_det MC {m[0]:,.0f}±{e[0]:.0f} vs an {Nd:,.0f} ({(m[0]-Nd)/e[0]:+.2f}σ)")
    print("   P(>=1): " + "; ".join(f"{t}': {m[1+i]:.3e}±{e[1+i]:.1e}|{fk1[i]:.3e} ({(m[1+i]-fk1[i])/e[1+i]:+.1f})" for i, t in enumerate(THB)))
    print("   P(>=2): " + "; ".join(f"{t}': {m[5+i]:.2e}±{e[5+i]:.1e}|{fk2[i]:.2e}" for i, t in enumerate(THB)))
    print(f"   B1phys: {m[9]:.4e}±{e[9]:.1e} | {fph:.4e} ({(m[9]-fph)/e[9]:+.2f})")
