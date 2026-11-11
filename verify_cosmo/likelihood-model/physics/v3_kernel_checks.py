"""V3: independent physics checks of the blended kernel ingredients (no rho table, no cclib/blendlib):
 (a) map-mean shift mu(theta_A) = sigma_A * RATIO^2 * ybar500 / N_w  (DC response of the matched filter to a
     uniform y; ybar500 from my own HMF integral incl. the lognormal mean factor), vs the kernel's mu.
 (b) P_dom(theta_A, q_A=5) = exp(-Lambda(x>=5)) and the mean / variance of the truncated neighbour sum,
     from szsim.cross_response (direct Hankel) and my own neighbour population, vs the G table."""
import numpy as np, time
import vlib as v
from szsim import signal as sg

t0 = time.time()
SZ = v.SZ
Nw = SZ.noise_y_arcmin ** 2                                   # arcmin^2 (y^2 arcmin^2)
pop = v.population(nd=v.nodes(nzper=6, nsub=6, nM=200, ngh=24))   # neighbours: M>1e13, 0.01<z<3
AREA = 4 * np.pi / v.ARC ** 2
ybar = np.sum(pop["w"] * pop["Y"]) / AREA
print(f"ybar500 (sum Y500 / area, M>1e13, 0.01<z<3) = {ybar:.4e}   (b03: {np.load(v.ROOT+'/data/blending/analytic.npz')['ybar'].sum():.4e})")
K = np.load(f"{v.ROOT}/data_cosmo/G_fid_b03rates.npz")
th = K["th"]; sA = SZ.sigma_Y(th)
mu_dc = sA * sg.RATIO_5R500 ** 2 * ybar / Nw
for i in (0, 10, 20, 26, 30, 39):
    print(f"  theta_A={th[i]:6.2f}': mu DC-identity {mu_dc[i]:.4f}  kernel mu {K['mu'][i]:.4f}  rel {K['mu'][i]/mu_dc[i]-1:+.2e}")
print(f"  max |rel| over 40 nodes: {np.max(np.abs(K['mu']/mu_dc-1)):.2e}")

# ---------------- (b) P_dom and truncated moments at q_A = 5
tg = K["tgrid"]; qn = K["qnodes"]; k5 = int(np.argmin(np.abs(qn - 5.0)))
thB_nodes = np.geomspace(0.03, 80, 36)
lthp = np.log(np.clip(pop["th"], thB_nodes[0], thB_nodes[-1]))
for iA in (5, 15, 25):
    thA = th[iA]; sigA = sA[iA]
    thr_grid = np.geomspace(1e-6, 100.0, 500)          # threshold on rho (rho_max can exceed 1: compact B in a wide filter)
    area = np.zeros((len(thB_nodes), len(thr_grid)))   # area where rho >= thr [arcmin^2]
    m1 = np.zeros_like(area); m2 = np.zeros_like(area) # int rho 1[rho<thr] d2s, int rho^2 1[rho<thr] d2s
    for j, tb in enumerate(thB_nodes):
        smax = 8 * (thA + tb) + 6.0
        s = np.linspace(0, smax, 1500); ds = s[1] - s[0]
        rho = SZ.cross_response(thA, tb, s, nl=12000)
        dA = 2 * np.pi * s * ds; dA[0] = np.pi * (ds / 2) ** 2; dA[-1] *= 0.5
        o = np.argsort(rho)[::-1]; r_s = rho[o]; a_s = dA[o]
        ca = np.cumsum(a_s); c1 = np.cumsum(a_s * r_s); c2 = np.cumsum(a_s * r_s ** 2)
        T1, T2 = c1[-1], c2[-1]
        n_ge = np.searchsorted(-r_s, -thr_grid, side="right")          # number of samples with rho >= thr
        area[j] = np.where(n_ge > 0, ca[np.maximum(n_ge - 1, 0)], 0.0)
        m1[j] = T1 - np.where(n_ge > 0, c1[np.maximum(n_ge - 1, 0)], 0.0)
        m2[j] = T2 - np.where(n_ge > 0, c2[np.maximum(n_ge - 1, 0)], 0.0)
    # each neighbour: threshold rho >= q_A sigma_A / Y_B
    thr = 5.0 * sigA / pop["Y"]
    def interp2(tab, extrap_low):
        ib = np.clip(np.searchsorted(np.log(thB_nodes), lthp) - 1, 0, len(thB_nodes) - 2)
        fb = (lthp - np.log(thB_nodes)[ib]) / np.diff(np.log(thB_nodes))[ib]
        lt = np.log(np.clip(thr, thr_grid[0], thr_grid[-1]))
        it = np.clip(np.searchsorted(np.log(thr_grid), lt) - 1, 0, len(thr_grid) - 2)
        ft = (lt - np.log(thr_grid)[it]) / np.diff(np.log(thr_grid))[it]
        val = 0
        for db, wb in ((0, 1 - fb), (1, fb)):
            for dt, wt in ((0, 1 - ft), (1, ft)):
                val = val + wb * wt * tab[ib + db, it + dt]
        return val
    A_ = interp2(area, 0); A_[thr > thr_grid[-1]] = 0.0
    M1 = interp2(m1, 0); M2 = interp2(m2, 0)
    # thresholds below grid (very bright neighbours): mean/var from rho<thr_grid[0] part only (negligible region)
    Lam = np.sum(pop["w"] * A_) / AREA
    x = pop["Y"] / sigA
    mean_tr = np.sum(pop["w"] * x * M1) / AREA
    var_tr = np.sum(pop["w"] * x ** 2 * M2) / AREA
    Pdom_mine = np.exp(-Lam)
    # kernel moments from G at (theta_A node, q=5 node)
    F = K["G"][iA, k5].astype(float); Pd = F[-1]
    pdf = np.diff(F) / Pd; tm = 0.5 * (tg[1:] + tg[:-1])
    mG = np.sum(pdf * tm); vG = np.sum(pdf * tm ** 2) - mG ** 2
    print(f"theta_A={thA:.2f}' q_A={qn[k5]:.2f}: P_dom mine {Pdom_mine:.4f} vs G {Pd:.4f} (c01 log value) | "
          f"E[y]=mean_tr-mu: mine {mean_tr-mu_dc[iA]:+.4f} vs G {mG:+.4f} | Var[y]=1+var_tr: mine {1+var_tr:.4f} vs G {vG:.4f}  ({time.time()-t0:.0f}s)")
