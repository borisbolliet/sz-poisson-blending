"""Step 3 helpers: tiles (truth, seeds), one finder run with the exact decomposition, and the per-tile driver.

Tile pid = stored Q2 patch pid (q2lib.load_patch; 10 x 10 deg, all halos M500 > 1e13, Y500 with scatter).  Its six
channel maps are   tSZ (painted, szsim.finder.paint_y)  +  CMB (seed SEED_CMB + pid)  +  noise (seed SEED_NOISE + pid;
the same unit draws for baseline and goal, only the white amplitudes differ).  All three components are kept apart,
so every filtered value splits exactly into tSZ + CMB + noise.
"""
import os, sys, time, resource
import numpy as np
import numba as nb

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/analysis/q2_maps")
import q2lib as L                                  # noqa: E402
from szsim import finder as F, instrument as ins   # noqa: E402

OUT = os.path.join(F.STEP3_DATA)
SEED_CMB, SEED_NOISE = 300_000, 400_000
SEED_INJ_CMB, SEED_INJ_NOISE = 500_000, 600_000
MODES = ("baseline", "goal")
K_TOP = 4                 # exact lattice values for the K largest contributors
Q_EXACT = 4.5             # ... of every final detection with q >= Q_EXACT
WIN = 6                   # isolated-search window half width [pixels] (3')
QT_FORCED = 2.0           # forced photometry at the true positions of halos with q_true > 2


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9      # macOS: bytes


_MMF = {}


def sigma_Y(mode, theta):
    """Step 1's MMF sigma_Y [arcmin^2] (szsim.instrument.MMFModel, l = 100-20000; spline as in steps 1-2)."""
    if mode not in _MMF:
        _MMF[mode] = ins.MMFModel(ins.Instrument.so_lat(mode, lmin=100.0, lmax=20000.0))
    return _MMF[mode].sigma_Y(theta)


def truth(pid):
    """Truth table of a tile: positions, z, M500, theta500, Y500 (with scatter), q_true for baseline and goal."""
    d = L.load_patch(pid)
    for m in MODES:
        d["q_" + m] = d["Y500"] / sigma_Y(m, d["theta500"])
    return d


@nb.njit(cache=True)
def topk_contrib(qy, qx, qj, own, hy, hx, hY, hlnth, Rn, Rf, lnth0, dln, rswitch, drn, rf_min, drf, box, rcut, topk,
                 out_idx, out_val):
    """Largest table contributions (excluding halo own[q]) within rcut of each query point at scale qj."""
    nth_nodes = Rn.shape[1]
    for d in range(qj.size):
        j = qj[d]
        tv = np.full(topk, -1e300)
        ti = np.full(topk, -1, np.int64)
        for h in range(hY.size):
            if h == own[d]:
                continue
            ddy = (hy[h] - qy[d] + 0.5 * box) % box - 0.5 * box
            if abs(ddy) > rcut:
                continue
            ddx = (hx[h] - qx[d] + 0.5 * box) % box - 0.5 * box
            if abs(ddx) > rcut:
                continue
            r = np.sqrt(ddy * ddy + ddx * ddx)
            if r > rcut:
                continue
            i0, w = F.theta_node(hlnth[h], lnth0, dln, nth_nodes)
            c = hY[h] * F._resp_eval(Rn, Rf, j, i0, w, r, rswitch, drn, rf_min, drf)
            if c > tv[topk - 1]:
                k = topk - 1
                while k > 0 and c > tv[k - 1]:
                    tv[k] = tv[k - 1]; ti[k] = ti[k - 1]
                    k -= 1
                tv[k] = c; ti[k] = h
        for k in range(topk):
            out_idx[d, k] = ti[k]
            out_val[d, k] = tv[k]


@nb.njit(cache=True)
def own_table(qy, qx, qj, own, hy, hx, hY, hlnth, Rn, Rf, lnth0, dln, rswitch, drn, rf_min, drf, box, out):
    nth_nodes = Rn.shape[1]
    for d in range(qj.size):
        h = own[d]
        ddy = (hy[h] - qy[d] + 0.5 * box) % box - 0.5 * box
        ddx = (hx[h] - qx[d] + 0.5 * box) % box - 0.5 * box
        r = np.sqrt(ddy * ddy + ddx * ddx)
        i0, w = F.theta_node(hlnth[h], lnth0, dln, nth_nodes)
        out[d] = hY[h] * F._resp_eval(Rn, Rf, qj[d], i0, w, r, rswitch, drn, rf_min, drf)


@nb.njit(cache=True)
def alone_window(cy, cx, h, hy, hx, hY, hlnth, Rn, Rf, lnth0, dln, rswitch, drn, rf_min, drf, box, pix, win, nmap,
                 sig, out_val, out_j, out_dy, out_dx):
    """Isolated search: max over a (2 win + 1)^2 pixel window centred on pixel (cy, cx) and over all scales of
    [Y_h R_j(|p - x_h|) + N_j(p)] / sigma_j, with N_j the filtered CMB + noise windows nmap[d, j, :, :]."""
    nth_nodes = Rn.shape[1]
    nsc = Rn.shape[0]
    for d in range(cy.size):
        hh = h[d]
        i0, w = F.theta_node(hlnth[hh], lnth0, dln, nth_nodes)
        best = -1e300; bj = -1; by = 0; bx = 0
        for j in range(nsc):
            for a in range(-win, win + 1):
                py = (cy[d] + a) * pix
                ddy = (py - hy[hh] + 0.5 * box) % box - 0.5 * box
                for b in range(-win, win + 1):
                    px = (cx[d] + b) * pix
                    ddx = (px - hx[hh] + 0.5 * box) % box - 0.5 * box
                    r = np.sqrt(ddy * ddy + ddx * ddx)
                    v = (hY[hh] * F._resp_eval(Rn, Rf, j, i0, w, r, rswitch, drn, rf_min, drf) + nmap[d, j, a + win, b + win]) / sig[j]
                    if v > best:
                        best = v; bj = j; by = a; bx = b
        out_val[d] = best; out_j[d] = bj; out_dy[d] = by; out_dx[d] = bx


def decompose(it, cxs, halos, basis, qmode, K=K_TOP, q_exact=Q_EXACT, forced=True):
    """Exact decomposition of the final detections of one finder run, the isolated search, and forced photometry at
    the true positions.  cxs: dict 'tsz', 'cmb', 'noise' of filt.CX(component channel rffts) for the final filter.
    Values in arcmin^2 (Y500 units)."""
    filt, det = it["filt"], it["det"]
    G = basis.G
    nth = len(filt.thetas)
    sig = filt.sigma_arcmin2
    t0 = time.time()
    Rn, Rf = F.response_tables(filt, basis)
    t_tab = time.time() - t0
    hy, hx, hY = halos["y"], halos["x"], halos["Y500"]
    hl = np.log(halos["theta500"])
    tab_args = (Rn, Rf, basis.lnth0, basis.dln, F.R_SWITCH, basis.rn[1] - basis.rn[0], F.R_FAR_MIN,
                basis.rf[1] - basis.rf[0], F.LBOX)
    nd = det["q"].size
    out = {}
    # ---- table contributions of every halo at every detection
    t0 = time.time()
    s_tab = np.zeros(nd); ti = np.full((nd, K), -1, np.int64); tv = np.zeros((nd, K)); nc = np.zeros(nd, np.int64)
    if nd:
        F.table_contributions(det["j"], det["y"], det["x"], hy, hx, hY, hl, *tab_args, K, s_tab, ti, tv, nc)
    t_con = time.time() - t0
    # ---- exact lattice values for the K largest, for q >= q_exact; re-rank
    t0 = time.time()
    ex = np.full((nd, K), np.nan)
    for d in np.flatnonzero(det["q"] >= q_exact):
        for k in range(K):
            h = ti[d, k]
            if h < 0:
                continue
            dy = (det["y"][d] - hy[h] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
            dx = (det["x"][d] - hx[h] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
            ex[d, k] = F.exact_contrib(filt, basis, det["j"][d], hY[h], halos["theta500"][h], dy, dx)
        o = np.argsort(-np.nan_to_num(ex[d], nan=-1e300))
        ex[d] = ex[d, o]; ti[d] = ti[d, o]; tv[d] = tv[d, o]
    t_ex = time.time() - t0
    # ---- component values at the detections; forced photometry at the true positions (halos with q_true > 2 and
    #      every detection primary): values at the nearest pixel and the scale nearest to theta500; CMB + noise windows
    t0 = time.time()
    cx_cn = cxs["cmb"] + cxs["noise"]
    S = {k: np.zeros(nd) for k in ("tsz", "cmb", "noise")}
    prim = ti[:, 0]
    fsel = np.flatnonzero(halos[qmode] > QT_FORCED) if forced else np.zeros(0, np.int64)
    fsel = np.union1d(fsel, prim[prim >= 0]).astype(np.int64)
    fiy = np.rint(hy[fsel] / F.PIX).astype(np.int64) % F.N
    fix = np.rint(hx[fsel] / F.PIX).astype(np.int64) % F.N
    lt = np.log(halos["theta500"][fsel])
    fj = np.abs(lt[:, None] - np.log(filt.thetas)[None, :]).argmin(1)
    fv = {k: np.zeros(fsel.size) for k in ("tsz", "cmb", "noise")}
    W2 = 2 * WIN + 1
    nwin = np.zeros((fsel.size, nth, W2, W2))
    ar = np.arange(-WIN, WIN + 1)
    for j in range(nth):
        mj = det["j"] == j
        fm = fj == j
        for k in ("tsz", "cmb", "noise"):
            m = filt.yhat_map(cxs[k], j) / F.AR ** 2
            S[k][mj] = m[det["iy"][mj], det["ix"][mj]]
            fv[k][fm] = m[fiy[fm], fix[fm]]
        mcn = filt.yhat_map(cx_cn, j) / F.AR ** 2
        nwin[:, j] = mcn[(fiy[:, None, None] + ar[None, :, None]) % F.N, (fix[:, None, None] + ar[None, None, :]) % F.N]
    t_maps = time.time() - t0
    # ---- isolated search around each forced halo: max over 13 x 13 pixels and all scales of (own + CMB + noise)/sigma
    t0 = time.time()
    qa = np.zeros(fsel.size); qa_j = np.zeros(fsel.size, np.int64); qa_dy = np.zeros(fsel.size, np.int64); qa_dx = np.zeros(fsel.size, np.int64)
    if fsel.size:
        alone_window(fiy, fix, fsel, hy, hx, hY, hl, *tab_args, F.PIX, WIN, nwin, sig, qa, qa_j, qa_dy, qa_dx)
    t_alone = time.time() - t0
    pos = np.searchsorted(fsel, np.maximum(prim, 0))
    d_qa = np.where(prim >= 0, qa[np.minimum(pos, max(fsel.size - 1, 0))], np.nan) if fsel.size else np.full(nd, np.nan)
    out.update(d_S_tot=det["q"] * sig[det["j"]], d_S_tsz=S["tsz"], d_S_cmb=S["cmb"], d_S_noise=S["noise"],
               d_tab_sum=s_tab, d_ncon=nc, d_top_idx=ti, d_top_tab=tv, d_top_ex=ex, d_q_alone=d_qa)
    # ---- forced photometry: own (table), largest neighbour contributions (table, within 60')
    t0 = time.time()
    own = np.zeros(fsel.size)
    qy, qx = fiy * F.PIX, fix * F.PIX
    own_table(qy, qx, fj, fsel, hy, hx, hY, hl, *tab_args, own)
    nb_i = np.full((fsel.size, 2), -1, np.int64); nb_v = np.zeros((fsel.size, 2))
    topk_contrib(qy, qx, fj, fsel, hy, hx, hY, hl, *tab_args, 60.0, 2, nb_i, nb_v)
    out.update(f_idx=fsel, f_j=fj, f_iy=fiy, f_ix=fix, f_S_tsz=fv["tsz"], f_S_cmb=fv["cmb"], f_S_noise=fv["noise"],
               f_own=own, f_nb_idx=nb_i, f_nb_val=nb_v, f_q_alone=qa, f_q_alone_j=qa_j, f_q_alone_dy=qa_dy,
               f_q_alone_dx=qa_dx)
    t_forced = time.time() - t0
    out["timing_decomp"] = np.array([t_tab, t_con, t_ex, t_maps, t_alone, t_forced])
    return out


def finder_outputs(its, prefix=""):
    """Compact per-iteration outputs of run_finder."""
    out = {}
    for k, it in enumerate(its):
        d = it["det"]
        out[f"{prefix}it{k}_sig"] = it["filt"].sigma_arcmin2
        out[f"{prefix}it{k}_fsky"] = it["fsky"]
        if it["Cbin"] is not None:
            out[f"{prefix}it{k}_C"] = it["Cbin"]
        for key in ("j", "iy", "ix", "q"):
            out[f"{prefix}it{k}_{key}"] = d[key]
    out[f"{prefix}n_it"] = len(its)
    return out
