"""Naive vs blended dN/dz/dq_obs at the fiducial cosmology, and the blended convolution model vs the
full-sky catalogue Monte Carlo (data/blending/mc_sky1-8: exact neighbour sums over every M>1e13 halo,
true positions, filter at the true theta500, map mean removed with the ensemble-mean mu, dominance rule)."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
import glob, json
import numpy as np
from scipy.stats import chi2 as chi2d
import cclib as cc

ZE = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0])
QE = np.array([5, 6, 7, 8, 10, 12, 15, 20, 30, 50, np.inf])
K = np.load(f"{cc.OUT}/G_fid_b03rates.npz")
G = K["G"]
A = np.load(f"{cc.ROOT}/data/blending/analytic.npz")
ok = A["Pdom"][0] > 0
print(f"P_dom vs b03 where b03 computed it: max |diff| = {np.abs(K['Pdom'][ok] - A['Pdom'][0][ok]).max():.2e}")

pop = cc.population(cc.FID, zedges=ZE)
nz = len(ZE) - 1
Nn = cc.counts_gauss(pop, QE, nz)
Nb = cc.counts_blended(G, pop, QE, nz)
QE4 = np.array([4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, np.inf])
Nn4 = cc.counts_gauss(pop, QE4, nz); Nb4 = cc.counts_blended(G, pop, QE4, nz)
np.savez(f"{cc.OUT}/counts_fid.npz", ZE=ZE, QE=QE, Nn=Nn, Nb=Nb, QE4=QE4, Nn4=Nn4, Nb4=Nb4)
for qc, E_, n_, b_ in ((5, QE, Nn, Nb), (4, QE4, Nn4, Nb4)):
    print(f"q_obs>{qc}: N_naive {n_.sum():,.1f}  N_blended {b_.sum():,.1f}  ratio {b_.sum()/n_.sum():.5f}")
# b03 reference for q>5 total
print("(b03 analytic, primaries q_A>0.5: N_det_b/N_single = 0.984 at q_cut=5, 0.965 at q_cut=4; see b03_analytic.log)")

def table(M, rowlab, collab, fmt):
    s = "        " + " ".join(f"{c:>8s}" for c in collab) + "\n"
    for r, row in zip(rowlab, M):
        s += f"{r:>8s}" + " ".join(fmt(v) for v in row) + "\n"
    return s

zl = [f"{a:g}-{b:g}" for a, b in zip(ZE[:-1], ZE[1:])]
ql = [f"{a:g}-{b:g}" for a, b in zip(QE[:-1], QE[1:])]
print("\nfull-sky naive counts N_n[z, q]:")
print(table(Nn, zl, ql, lambda v: f"{v:8.1f}"))
print("blended/naive - 1 [%]:")
print(table(100 * (Nb / Nn - 1), zl, ql, lambda v: f"{v:8.2f}"))
for fs in (1.0, 0.4):
    P = cc.pull_sigma(Nb, Nn, fs)
    print(f"(N_b - N_n)/sqrt(N_n) for f_sky = {fs}:")
    print(table(P, zl, ql, lambda v: f"{v:8.2f}"))
    print(f"   sum of squared pulls (Delta chi2 naive vs blended, 120 bins) = {np.sum(P**2):.1f};"
          f" bins with |pull|>1: {(np.abs(P)>1).sum()}, >2: {(np.abs(P)>2).sum()}, >3: {(np.abs(P)>3).sum()}")
# marginals
print("q marginal: N_n, N_b, rel diff [%], full-sky pull, 40% pull")
for j in range(len(QE) - 1):
    n_, b_ = Nn[:, j].sum(), Nb[:, j].sum()
    print(f"  q {ql[j]:>8s}: {n_:9.1f} {b_:9.1f} {100*(b_/n_-1):+7.2f}%  {cc.pull_sigma(b_, n_, 1.0):+7.2f}  {cc.pull_sigma(b_, n_, 0.4):+7.2f}")
print("z marginal (q>5):")
for i in range(nz):
    n_, b_ = Nn[i].sum(), Nb[i].sum()
    print(f"  z {zl[i]:>9s}: {n_:9.1f} {b_:9.1f} {100*(b_/n_-1):+7.2f}%  {cc.pull_sigma(b_, n_, 1.0):+7.2f}  {cc.pull_sigma(b_, n_, 0.4):+7.2f}")

# ============================================================== catalogue MC comparison
print("\n=== blended convolution model vs catalogue MC (full skies, primaries q_A>0.5) ===")
pop05 = cc.population(cc.FID, zedges=ZE, qmin=0.5)
Nb05 = cc.counts_blended(G, pop05, QE, nz); Nn05 = cc.counts_gauss(pop05, QE, nz)
files = sorted(glob.glob(f"{cc.ROOT}/data/blending/mc_sky*.npz"), key=lambda f: int(f.split("sky")[1][:-4]))
lth_n = np.log(A["th"]); mu_n = A["mu_exact"].sum(1)
cq_b, cq_n, czq_b, czq_n = [], [], [], []
for f in files:
    s = int(f.split("sky")[1][:-4])
    d = np.load(f)
    th = d["p_theta500"].astype(np.float64); sig = cc.sigma_Y(th)
    q = d["p_q"].astype(np.float64); qo = d["p_q_obs"].astype(np.float64)
    dl = d["D"].sum(1) / sig; xm = d["xmax"].max(1) / sig
    mu = np.interp(np.log(th), lth_n, mu_n)
    qb = qo + dl - mu
    det = q > xm
    hb = np.histogram(qb[det], QE)[0]; hn = np.histogram(qo, QE)[0]
    cq_b.append(hb); cq_n.append(hn)
    if s <= 3:
        from szsim import sampler
        z = np.load(f"{cc.ROOT}/data/catalogs/cat_M1e13_seed{s}.npz", mmap_mode="r")["z"][d["p_idx"]].astype(np.float64)
        czq_b.append(np.histogram2d(z[det], qb[det], [ZE, QE])[0]); czq_n.append(np.histogram2d(z, qo, [ZE, QE])[0])
    print(f"  sky {s}: blended N(q>5) {hb.sum()}, naive {hn.sum()}")
cq_b = np.array(cq_b); cq_n = np.array(cq_n)
ns = len(files)
for lab, C, M in (("blended", cq_b, Nb05.sum(0)), ("naive  ", cq_n, Nn05.sum(0))):
    mean = C.mean(0); err = np.sqrt(M / ns)
    z_ = (mean - M) / err
    c2 = np.sum(z_ ** 2)
    print(f"  {lab}: q bins MC mean/model - 1 [%]: {np.array2string(100*(mean/M-1), precision=2)}")
    print(f"           pulls: {np.array2string(z_, precision=2)}  chi2 = {c2:.1f} / {len(M)} (p = {chi2d.sf(c2, len(M)):.3f});"
          f" between-sky var/Poisson (mean over bins) = {np.mean(C.var(0, ddof=1)/M):.2f}")
r_mc = cq_b.sum(1) / cq_n.sum(1)
print(f"  N_b/N_n (q>5): MC {r_mc.mean():.5f} +- {r_mc.std(ddof=1)/np.sqrt(ns):.5f} (sky scatter/sqrt(n)), model {Nb05.sum()/Nn05.sum():.5f}")
# z-q comparison with skies 1-3
czq_b = np.array(czq_b); czq_n = np.array(czq_n); n3 = len(czq_b)
for lab, C, M in (("blended", czq_b, Nb05), ("naive  ", czq_n, Nn05)):
    mean = C.mean(0); err = np.sqrt(M / n3); z_ = (mean - M) / err
    c2 = np.sum(z_ ** 2)
    print(f"  {lab} (z,q) 12x10 bins, {n3} skies: chi2 = {c2:.1f} / {M.size} (p = {chi2d.sf(c2, M.size):.3f}), max|pull| {np.abs(z_).max():.2f}")
    zm = C.sum(2).mean(0); Mz = M.sum(1); print(f"           z marginal MC/model - 1 [%]: {np.array2string(100*(zm/Mz-1), precision=2)}")
# the blended-minus-naive difference itself: MC vs model (paired; error from the MC paired difference scatter)
dmc = (czq_b - czq_n).mean(0); dmod = Nb05 - Nn05
print(f"  blended-naive difference summed in z, MC vs model per q bin: MC {np.array2string((cq_b-cq_n).mean(0), precision=0)}")
print(f"                                                         model {np.array2string(dmod.sum(0), precision=0)}")
json.dump(dict(Nb=Nb.tolist(), Nn=Nn.tolist()), open(f"{cc.OUT}/counts_fid.json", "w"))
