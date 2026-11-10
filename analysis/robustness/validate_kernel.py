"""Validate the neighbour-sum kernel:
(1) brute force (all neighbours, numpy, same rho table) vs kernel for 60 real candidates;
(2) Campbell's theorem: mean and variance of dY = sum_B rho_AB Y_B at random sky points
    (probe filters of fixed theta_A) vs the analytic shot-noise moments."""
import time, numpy as np
from szsim import sampler, signal as sg, cosmo_hmf as ch
import blendlib as bl

t0 = time.time()
model = sg.SZModel(); table = ch.default_table(); rt = bl.RhoTable(1.4)
cat = sampler.draw_catalog(1001, Mmin=1e13, model=model, fields=("ra", "dec", "z", "M500", "Y500", "theta500", "q"))
bl.add_distances(cat, model)
N = len(cat["ra"]); ids = np.arange(N)
print(f"drawn N={N} in {time.time()-t0:.1f}s")

# (1) brute force on 60 candidates with q>3
rng = np.random.default_rng(3)
cidx = rng.choice(np.nonzero(cat["q"] > 3)[0], 60, replace=False)
cand = {k: cat[k][cidx] for k in ("ra", "dec", "theta500", "chi", "Rcom")}
t = time.time()
res = bl.blend_sums(cand, cat, rt, ids[cidx], ids)
print(f"kernel on 60 candidates: {time.time()-t:.1f}s")
nx, ny, nz = [a.astype(np.float32) for a in bl.unitvec(cat["ra"], cat["dec"])]
lthB = np.log(cat["theta500"].astype(np.float64))
bf = np.empty(60)
for j, i in enumerate(cidx):
    # preselect with float32 (generous margin), then exact float64 geometry on the subset
    rsel = (5 * (cat["theta500"][i] + cat["theta500"].max()) + 4 * 1.4 + 5) / 60.0
    pre = np.nonzero(nx * nx[i] + ny * ny[i] + nz * nz[i] > np.cos(np.radians(rsel)) - 1e-6)[0]
    ax, ay, az = bl.unitvec(cat["ra"][pre].astype(np.float64), cat["dec"][pre].astype(np.float64))
    cx, cy, cz = bl.unitvec(float(cat["ra"][i]), float(cat["dec"][i]))
    chord = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2 + (az - cz) ** 2)
    sep = np.degrees(2 * np.arcsin(chord / 2)) * 60
    m = (sep < 5 * (cat["theta500"][i] + cat["theta500"][pre]) + 4 * 1.4) & (pre != i)
    r = np.empty(m.sum()); bl._rho_vec(rt.rho32, rt.lth, rt.sep, np.full(m.sum(), np.log(cat["theta500"][i])), lthB[pre][m], sep[m], 1.4, r)
    bf[j] = np.sum(r * cat["Y500"][pre][m])
rel = np.abs(res["sum"] / bf - 1)
print(f"brute force vs kernel (60 candidates): max rel diff = {rel.max():.2e}, median = {np.median(rel):.2e}; "
      f"median dY/sigma_A = {np.median(bf / model.sigma_Y(cat['theta500'][cidx])):.3f}")
del nx, ny, nz

# (2) Campbell at random probe points
thAs = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
from scipy.interpolate import RegularGridInterpolator
I2i = RegularGridInterpolator((rt.lth, rt.lth), np.log(rt.I2), bounds_error=False, fill_value=None)
nprobe = 400_000
mean_an, var_an = bl.campbell_moments(model, table, rt, thAs, Mmin=1e13)
ybar_an = bl.mean_ybar(model, table, 1e13)
ybar_sky = sg.RATIO_5R500 * cat["Y500"].astype(np.float64).sum() * bl.ARCMIN ** 2 / (4 * np.pi)
print(f"ybar: analytic {ybar_an:.5e}, this sky {ybar_sky:.5e} (ratio {ybar_sky/ybar_an:.4f}); "
      f"the probe mean must equal sigma_A^2 RATIO ybar_sky/N_w exactly (kernel test), and ybar_an in the ensemble")
for k, tA in enumerate(thAs):
    u = rng.random(nprobe); v = rng.random(nprobe)
    pr = dict(ra=360 * u, dec=np.degrees(np.arcsin(2 * v - 1)), theta500=np.full(nprobe, tA),
              chi=np.full(nprobe, 1e9), Rcom=np.zeros(nprobe))
    t = time.time()
    r = bl.blend_sums(pr, cat, rt, -np.ones(nprobe, dtype=np.int64), ids)["sum"]
    sA = model.sigma_Y(tA)
    # error on the mean and variance by jackknife over 100 blocks (heavy tails -> block resampling)
    blocks = np.array_split(r, 100)
    mj = np.array([np.mean(np.concatenate(blocks[:b] + blocks[b + 1:])) for b in range(100)])
    vj = np.array([np.var(np.concatenate(blocks[:b] + blocks[b + 1:])) for b in range(100)])
    em = np.sqrt(99 / 100 * np.sum((mj - mj.mean()) ** 2)); ev = np.sqrt(99 / 100 * np.sum((vj - vj.mean()) ** 2))
    msky = mean_an[k] * ybar_sky / ybar_an
    # sky-realised diagonal (shot-noise) variance: sum_B Y_B^2 int rho_AB^2 d^2sep / (4pi)
    I2B = np.exp(I2i(np.column_stack([np.full(N, np.log(tA)), lthB]))) / bl.ARCMIN ** 2
    vsky = np.sum(cat["Y500"].astype(np.float64) ** 2 * I2B) / (4 * np.pi / bl.ARCMIN ** 2)
    del I2B
    print(f"thetaA={tA:5.1f}' ({time.time()-t:.1f}s): <dq> MC = {r.mean()/sA:.4f} +- {em/sA:.4f}, sky-exact = {msky/sA:.4f} "
          f"({(r.mean()-msky)/em:+.2f} sigma), ensemble Campbell = {mean_an[k]/sA:.4f} | std(dq) MC = {np.sqrt(r.var())/sA:.4f}, Campbell = {np.sqrt(var_an[k])/sA:.4f}; "
          f"var ratio MC/Campbell = {r.var()/var_an[k]:.3f} +- {ev/var_an[k]:.3f} ({(r.var()-var_an[k])/ev:+.2f} sigma) | "
          f"var MC/sky-diagonal = {r.var()/vsky:.3f} ({(r.var()-vsky)/ev:+.2f} sigma) | median dq = {np.median(r)/sA:.4f}, P(dq>1) = {np.mean(r > sA):.4f}")
print(f"ybar(M>1e13) = {bl.mean_ybar(model, table, 1e13):.4e}, ybar(M>1e14) = {bl.mean_ybar(model, table, 1e14):.4e}")
import resource; print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB; total {time.time()-t0:.0f}s")
