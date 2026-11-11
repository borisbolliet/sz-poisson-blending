"""Size of each ingredient of the blended count model (fiducial, q_obs > 5, 12 z x 10 q bins), and a paired
MC test of the blended-minus-naive difference.
 (i)   detections from primaries with noise-free q_A < 5 (boosted in; automatically in the convolution)
 (ii)  the dominance rule (lost/absorbed: counted once, under the brightest contributor) -> G without truncation
 (iii) noise-only peaks (map pure-noise false-detection rate, data/q2_maps/noise_confusion.json)
 (iv)  map-mean subtraction: raw-sky kernel G_raw(t) = G(t - mu)
 (v)   paired MC: per sky, (blended - naive) counts per q bin from the catalogue MC (8 full skies) vs model."""
import os, glob, json
os.environ.setdefault("NUMBA_NUM_THREADS", "3"); os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
from scipy.stats import chi2 as chi2d
import cclib as cc

C = np.load(f"{cc.OUT}/counts_fid.npz")
ZE, QE, Nb, Nn = C["ZE"], C["QE"], C["Nb"], C["Nn"]
nz = len(ZE) - 1
K = np.load(f"{cc.OUT}/G_fid_b03rates.npz")
G, Pdom, mu = K["G"], K["Pdom"], K["mu"]
pop = cc.population(cc.FID, zedges=ZE)


def dchi(N, ref, fs):
    P = cc.pull_sigma(N, ref, fs)
    return np.sum(P ** 2), np.abs(P).max()


def line(lab, N, ref=Nb):
    print(f"  {lab}: total {N.sum():,.1f} ({100*(N.sum()/ref.sum()-1):+.2f}% vs blended); q-marginal rel. [%] "
          f"{np.array2string(100*(N.sum(0)/ref.sum(0)-1), precision=2)}; Delta chi2 vs blended full sky {dchi(N, ref, 1.0)[0]:.1f}"
          f" (max|pull| {dchi(N, ref, 1.0)[1]:.2f}), 40% {dchi(N, ref, 0.4)[0]:.1f}")


print(f"blended N(q_obs>5) = {Nb.sum():,.1f}; naive = {Nn.sum():,.1f}")
# (i)
lo = dict(pop); lo["zb"] = np.where(pop["q"] < 5, pop["zb"], -1)
Nlo = cc.counts_blended(G, lo, QE, nz)
Nlo_n = cc.counts_gauss(lo, QE, nz)
print(f"(i) blended detections from primaries with noise-free q_A < 5: {Nlo.sum():,.1f} = {100*Nlo.sum()/Nb.sum():.2f}% "
      f"(naive model: {100*Nlo_n.sum()/Nn.sum():.2f}%); per q bin [%]: {np.array2string(100*Nlo.sum(0)/Nb.sum(0), precision=2)}")
lo2 = dict(pop); lo2["zb"] = np.where(pop["q"] < 3, pop["zb"], -1)
print(f"    from q_A < 3: blended {100*cc.counts_blended(G, lo2, QE, nz).sum()/Nb.sum():.2f}%, naive {100*cc.counts_gauss(lo2, QE, nz).sum()/Nn.sum():.2f}%")
# (ii) no dominance: untruncated F at every q node, P_dom = 1
kk = len(cc.QNODES) - 1
Gnd = np.repeat((G[:, kk] / Pdom[:, kk, None])[:, None, :], len(cc.QNODES), axis=1).astype(np.float32)
line("(ii) without the dominance rule (every halo counted, untruncated P(D))", cc.counts_blended(Gnd, pop, QE, nz))
print(f"     P_dom at q_A = 5 for theta = 0.52/2.3/10.1': {np.round(Pdom[[5,15,25], 49], 4)}")
# (iii) noise-only peaks
nc = json.load(open(f"{cc.ROOT}/data/q2_maps/noise_confusion.json"))["noise_false_rate_per_deg2"]["R2"]
print(f"(iii) map pure-noise false detections per deg^2 at q>4,5,6,7: {nc}")
f5 = nc[1] * cc.FULLSKY_DEG2
for fs in (1.0, 0.4):
    n56 = Nb[:, 0].sum() * fs
    print(f"     f_sky={fs}: {f5*fs:.1f} noise peaks with q>5 (~all in 5<q<6) vs {n56:,.0f} blended detections in 5<q<6: "
          f"+{100*f5/Nb[:,0].sum():.2f}%, {f5*fs/np.sqrt(n56):.2f} sigma (q marginal); no counterpart redshift")
# (iv) raw sky (mean not removed)
Graw = np.empty_like(G)
for i in range(G.shape[0]):
    for k in range(G.shape[1]):
        Graw[i, k] = np.interp(cc.TGRID - mu[i], cc.TGRID, G[i, k], left=0.0, right=float(G[i, k, -1]))
line("(iv) raw sky, map mean NOT removed", cc.counts_blended(Graw, pop, QE, nz))
line("     naive (for reference)", Nn)

# (v) paired MC test of the blended - naive difference
print("(v) paired catalogue-MC test of (blended - naive) per q bin, 8 full skies, primaries q_A > 0.5")
A = np.load(f"{cc.ROOT}/data/blending/analytic.npz")
lth_n = np.log(A["th"]); mu_n = A["mu_exact"].sum(1)
files = sorted(glob.glob(f"{cc.ROOT}/data/blending/mc_sky*.npz"), key=lambda f: int(f.split("sky")[1][:-4]))
diff = []
for f in files:
    d = np.load(f)
    th = d["p_theta500"].astype(np.float64); sig = cc.sigma_Y(th)
    q = d["p_q"].astype(np.float64); qo = d["p_q_obs"].astype(np.float64)
    qb = qo + d["D"].sum(1) / sig - np.interp(np.log(th), lth_n, mu_n)
    det = q > d["xmax"].max(1) / sig
    diff.append(np.histogram(qb[det], QE)[0] - np.histogram(qo, QE)[0])
diff = np.array(diff, float)
pop05 = cc.population(cc.FID, zedges=ZE, qmin=0.5)
dmod = (cc.counts_blended(G, pop05, QE, nz) - cc.counts_gauss(pop05, QE, nz)).sum(0)
m = diff.mean(0); e = diff.std(0, ddof=1) / np.sqrt(len(diff))
z_ = (m - dmod) / e
print(f"     MC mean      : {np.array2string(m, precision=1)}")
print(f"     model        : {np.array2string(dmod, precision=1)}")
print(f"     error of mean: {np.array2string(e, precision=1)}")
print(f"     pulls {np.array2string(z_, precision=2)}; chi2 = {np.sum(z_**2):.1f}/{len(z_)} (p = {chi2d.sf(np.sum(z_**2), len(z_)):.3f});"
      f" total: MC {diff.sum(1).mean():.1f} +- {diff.sum(1).std(ddof=1)/np.sqrt(len(diff)):.1f}, model {dmod.sum():.1f}")
