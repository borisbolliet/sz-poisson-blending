"""v06: the BUILD's catalogue-level P(D) (pdlib1 + blendlib, imported read-only, run exactly as in s06b/s06c)
evaluated at the 12 theta_A nodes of the map run (v05), for the comparison map vs catalogue.

Per run (baseline, goal, white, baseline_sub, goal_sub) and node:
  * the build's signed rate histogram (rate_signed), moments, Parseval extra variance (as s06b);
  * the UNTRUNCATED P(D) of delta - mu on the build's FFT grid, re-binned onto the map histogram bins (v05);
  * node probabilities on the build's q grid (QNODES) for q_cut = 4..7: WITH the dominance rule (pl.node_probs,
    as in the build's headline numbers) and WITHOUT it (same formulas, untruncated P(D); x >= 500 cap counted
    as detected and boosted) -- the map measures the latter.
  * primary populations H(theta node, q node): the build's pl.primary_population on these nodes, full z range and
    per z bin (own z-restricted copy, same algorithm) for the redshift tilt.
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, f"{ROOT}/analysis/realistic/step1")
import numpy as np                                     # noqa: E402
from scipy.special import ndtr                         # noqa: E402
import pdlib1 as pl                                    # noqa: E402
import blendlib as bl                                  # noqa: E402
from szsim import signal as sg, cosmo_hmf as ch        # noqa: E402
sys.path.insert(0, HERE)
import importlib.util                                  # noqa: E402
spec = importlib.util.spec_from_file_location("vhist", os.path.join(HERE, "vlib.py"))
# only the histogram edges are needed from vlib (avoid its thread settings: re-set after import)
VL = importlib.util.module_from_spec(spec); spec.loader.exec_module(VL)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

log = VL.Tee(os.path.join(HERE, "v06_pd_builder_nodes.log"))
QCUTS = [4.0, 5.0, 6.0, 7.0]
ZB = [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
t0 = time.time()
th_tab = np.load(f"{ROOT}/data_real/rho_mmf_baseline.npz")["th"]
NODES = th_tab[::2][6:40:3]
EDGES = VL.hist_edges()
RUNS = sys.argv[1:] or ["baseline", "goal", "white", "baseline_sub", "goal_sub"]
cv = np.load(f"{ROOT}/data_real/confusion_var.npz")
pop_all = bl.neighbour_population(zsub=1, msub=6, ngh=16)
log(f"[v06] nodes {np.round(NODES, 4)}; neighbour population {len(pop_all[0]):,} points; runs {RUNS}")


def primary_population_z(model, th_nodes, qmin, zlo, zhi, zsub=2, msub=2, ngh=24):
    """Copy of pdlib1.primary_population restricted to zlo <= z < zhi (trapezoid weights clipped to the range)."""
    T = ch.default_table()
    z = T.z[::zsub]; lnM = T.lnM[::msub]; d2N = T.d2N[::zsub, ::msub]
    wz = ch.HMFTable._weights(z, zlo, zhi)
    wM = np.full(len(lnM), lnM[1] - lnM[0]); wM[0] *= .5; wM[-1] *= .5
    x, wg = np.polynomial.hermite_e.hermegauss(ngh); wg /= wg.sum()
    NT = len(th_nodes); lth0 = np.log(th_nodes[0]); dlth = np.log(th_nodes[1] / th_nodes[0])
    H = np.zeros((NT, len(pl.QNODES)))
    M = np.exp(lnM)
    for iz, zz in enumerate(z):
        if wz[iz] == 0:
            continue
        th = model.theta500(M, zz)
        q0 = model.Y500_med(M, zz) / model.sigma_Y(th)
        q = (q0[:, None] * np.exp(model.sigma_lnY * x)[None, :]).ravel()
        w = ((d2N[iz] * wz[iz] * wM)[:, None] * wg[None, :]).ravel()
        tt = np.repeat(th, ngh)
        s = q > qmin
        q, w, tt = q[s], w[s], tt[s]
        fa = np.clip((np.log(tt) - lth0) / dlth, 0, NT - 1.000001)
        ia = fa.astype(int); wa = fa - ia
        k = np.clip(np.searchsorted(pl.QNODES, q) - 1, 0, len(pl.QNODES) - 2)
        j0 = np.clip(np.where(q - pl.QNODES[k] < pl.QNODES[k + 1] - q, k - 1, k), 0, len(pl.QNODES) - 3)
        x0, x1, x2 = pl.QNODES[j0], pl.QNODES[j0 + 1], pl.QNODES[j0 + 2]
        L = [(q - x1) * (q - x2) / ((x0 - x1) * (x0 - x2)), (q - x0) * (q - x2) / ((x1 - x0) * (x1 - x2)),
             (q - x0) * (q - x1) / ((x2 - x0) * (x2 - x1))]
        for da, fa_ in ((0, 1 - wa), (1, wa)):
            for db in range(3):
                np.add.at(H, (ia + da, j0 + db), w * fa_ * L[db])
    return H


def probs_from_pdf(p, g, qa, mu):
    """Same quantities as pl.node_probs for a pdf p on grid g, without dominance; written with masks
    (g > d  <->  index >= jidx(d) of the build's code), on the trimmed support p > 1e-18."""
    out = {}
    dg = g - mu
    pos = dg > 0
    for qc in QCUTS:
        phi = ndtr(qa + dg - qc)
        pg = p * phi
        phi0 = ndtr(qa - qc)
        Iall = pg.sum()
        out[qc] = dict(det=Iall, b3=max(pg[pos].sum() - phi0 * p[pos].sum(), 0.0),
                       lost=max((p[~pos] * (phi0 - phi[~pos])).sum(), 0.0),
                       b2=np.array([pg[dg > e * qa].sum() for e in pl.EPS]),
                       b2n=np.array([pg[dg <= -e * qa].sum() for e in pl.EPS]),
                       mean=(pg * dg).sum())
    return out


save = {"nodes": NODES, "qnodes": pl.QNODES, "eps": pl.EPS, "qcuts": np.array(QCUTS), "zb": np.array(ZB)}
pd_small = pl.PDs(N=2 ** 18, dd=0.001, x_small=0.002, off=60.0)
pd_big = pl.PDs(N=2 ** 19, dd=0.002, x_small=0.004, off=60.0)
pd_un = pl.PDs(N=2 ** 20, dd=0.001, x_small=0.002, off=60.0)
for run in RUNS:
    tr = time.time()
    mode = run.replace("_sub", "")
    if mode == "white":
        tab = bl.load_rho_table(); model = sg.SZModel()
    else:
        tab = bl.RhoTable(np.load(f"{ROOT}/data_real/rho_mmf_{mode}.npz")); model = pl.make_model(("so", mode, 100.0))
    pop = pop_all
    if run.endswith("_sub"):
        thB, YB, cB, wB = pop_all
        keep = YB / model.sigma_Y(thB) < 5.0
        pop = (thB[keep], YB[keep], cB[keep], wB[keep])
    vkey = f"{mode}_var_{'subthr' if run.endswith('_sub') else 'all'}"
    var_pars = np.exp(np.interp(np.log(NODES), np.log(cv["thA"]), np.log(cv[vkey])))
    sigA = model.sigma_Y(NODES)
    cl = (0, 1, 2)
    NA, nq, nE, nC = len(NODES), len(pl.QNODES), len(pl.EPS), len(QCUTS)
    PD = np.zeros((NA, VL.H_NB)); MU = np.zeros(NA); VAR = np.zeros(NA); EXTRA = np.zeros(NA); LCAP = np.zeros(NA)
    res = {k: np.zeros((2, nC, NA, nq)) for k in ("det", "b3", "lost", "mean")}
    res["b2"] = np.zeros((2, nC, NA, nq, nE)); res["b2n"] = np.zeros((2, nC, NA, nq, nE)); res["pdom"] = np.zeros((NA, nq))
    for i in range(NA):
        R = pl.rate_signed(NODES[i], sigA[i], pop, tab)
        mu, var = pl.moments(R, cl)
        extra = 0.0 if mode == "white" else max(var_pars[i] - var, 0.0)
        MU[i], VAR[i], EXTRA[i] = mu, var, extra
        W, X, X2, lam_cap = pl.signed_arrays(R, cl)
        LCAP[i] = lam_cap
        p = pd_un.pdf(W, X, X2, extra_var=extra)
        p = p / p.sum() * np.exp(-lam_cap)
        g = pd_un.grid - mu                                      # delta - mu
        b = np.clip(np.searchsorted(EDGES, g, side="right") - 1, 0, VL.H_NB - 1)
        PD[i] = np.bincount(b, weights=p, minlength=VL.H_NB)
        PD[i, -1] += 1 - np.exp(-lam_cap)                         # x >= 500 cap -> top bin
        # no dominance: untruncated pdf (same for every q_A); cap events are detected and boosted
        pu = pd_big.pdf(W, X, X2, extra_var=extra)
        pu = pu / pu.sum() * np.exp(-lam_cap)
        tr_ = pu > 1e-18
        pu_t, gu_t = pu[tr_], pd_big.grid[tr_]
        for k, qa in enumerate(pl.QNODES):
            o = probs_from_pdf(pu_t, gu_t, qa, mu)
            pc = 1 - np.exp(-lam_cap)
            for ic, qc in enumerate(QCUTS):
                r = o[qc]
                res["det"][0, ic, i, k] = r["det"] + pc
                res["b3"][0, ic, i, k] = r["b3"] + pc * (1 - ndtr(qa - qc))
                res["lost"][0, ic, i, k] = r["lost"]
                res["mean"][0, ic, i, k] = r["mean"] / qa
                res["b2"][0, ic, i, k] = r["b2"] + pc
                res["b2n"][0, ic, i, k] = r["b2n"]
            # with dominance (the build's headline convention), exactly pl.node_probs as in s06b
            pdm, o2 = pl.node_probs(R, cl, qa, QCUTS, mu, pd_small, pd_big, extra_var=extra)
            res["pdom"][i, k] = pdm
            for ic, qc in enumerate(QCUTS):
                r = o2[qc]
                res["det"][1, ic, i, k] = r["det"] * pdm
                res["b3"][1, ic, i, k] = r["b3"] * pdm
                res["lost"][1, ic, i, k] = r["lost"] * pdm
                res["mean"][1, ic, i, k] = r["mean"] * pdm / qa
                res["b2"][1, ic, i, k] = r["b2"] * pdm
                res["b2n"][1, ic, i, k] = r["b2n"] * pdm
        log(f"   [{run}] node {i} thA={NODES[i]:7.3f}': mu={mu:+.5f} var(hist)={var:.4f} var(Parseval)={var_pars[i]:.4f} "
            f"extra={extra:.5f} Lambda_cap={lam_cap:.1e} [{time.time()-tr:.0f}s]")
    H = pl.primary_population(model, NODES, 0.5)
    Hz = np.array([primary_population_z(model, NODES, 0.5, ZB[b], ZB[b + 1]) for b in range(len(ZB) - 1)])
    save.update({f"{run}_PD": PD, f"{run}_mu": MU, f"{run}_var": VAR, f"{run}_extra": EXTRA, f"{run}_lcap": LCAP,
                 f"{run}_sigA": sigA, f"{run}_H": H, f"{run}_Hz": Hz, f"{run}_varpars": var_pars,
                 **{f"{run}_{k}": v for k, v in res.items()}})
    np.savez(os.path.join(HERE, "out", "v06_pd_builder_nodes.npz"), **save)
    # quick headline at these nodes
    Phi = ndtr(pl.QNODES[None, :] - 5.0)
    Ns = (H * Phi).sum()
    for d, name in ((0, "no dominance"), (1, "with dominance")):
        Nd = (H * res["det"][d, 1]).sum()
        log(f"   [{run}] q>5 on these 12 nodes, {name}: N_single {Ns:,.0f} N_det {Nd:,.0f} ({Nd/Ns-1:+.4f}); "
            f"B2(0.1) {(H*res['b2'][d,1,:,:,1]).sum()/Nd:.4f}; B2(0.2) {(H*res['b2'][d,1,:,:,2]).sum()/Nd:.4f}; "
            f"B3 {(H*res['b3'][d,1]).sum()/Nd:.4f}; lost {(H*res['lost'][d,1]).sum()/Ns:.4f}; "
            f"mean bias {(H*res['mean'][d,1]).sum()/Nd:+.4f}")
    log(f"   [{run}] done {time.time()-tr:.0f}s (total {time.time()-t0:.0f}s)")
log(f"[v06] done {time.time()-t0:.0f}s")
