"""x02 (indep lens), check (d): the exact decomposition of the build's detections, recomputed with own code.

Build tiles PIDS, both noise levels, the build's final filter (its stored it1_C, own MMF code).  For every final
detection with q > 5 (the detections the classes are built on):
  * c_p: the build's primary (d_top_idx[:, 0]) alone, filtered with own code at the detection's pixel and scale:
      - 'mix'  : with the build's 2-node template mixture (tests the build's numbers at machine precision);
      - 'own'  : with the template of the halo's own size (g_cub), i.e. painted independently of the build;
    by a direct lattice sum over all in-band modes (= paint alone + filter + read the pixel; shown literally below for
    a subset by painting the halo alone with the NUFFT painter and filtering the map);
  * delta = filtered tSZ of all other halos = filtered tSZ map (own painting) - c_p; for a subset the 'rest' map is
    painted literally without the primary and filtered;
  * the 2nd-4th contributors (build's d_top_idx[:, 1:]) by the same lattice sums;
  * own ranking: own continuous response tables for every halo within 60', own top 4 re-ranked by exact sums -> is the
    own primary the build's primary?
  * class labels (real c_p >= 2 sigma; Y boosted > 10% / > 20%) from own vs build numbers.
Also (D4): direction dependence of the binned-covariance filter's lattice response vs the analytic filter's.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vlib3 as V

PIDS = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 137, 290]
log = V.Tee(os.path.join(V.HERE, "x02_decomp_d.log"))
t00 = time.time()
G = V.grid()
P2 = V.Painter(G, eps=1e-9, scheme="mix2")
P4 = V.Painter(G, eps=1e-9, scheme="lag4")
HN = V.HankelNodes(G)
J0M = None
log(f"x02: check (d) on build tiles {PIDS}")


def own_ranking(filt, det_y, det_x, det_j, tr, K=4, rcut=60.0):
    """Own top-K contributors by the continuous tables, then exact lattice sums ('own' template)."""
    global J0M
    if J0M is None:
        from scipy.special import j0
        J0M = j0(np.outer(HN.l, V.R_NEAR * V.AR))
    lth = np.log(tr["theta500"])
    i0, _ = V.lagrange4(lth)
    nodes = np.arange(i0.min() - 2, i0.max() + 4)
    RT = V.ResponseTables(filt, HN, nodes, J0M)
    out_i = np.full((det_y.size, K), -1, np.int64)
    for d in range(det_y.size):
        dy = (tr["y"] - det_y[d] + 300) % 600 - 300
        dx = (tr["x"] - det_x[d] + 300) % 600 - 300
        r = np.hypot(dy, dx)
        cand = np.flatnonzero(r < rcut)
        v = np.array([tr["Y500"][h] * V.resp_eval(RT.R, int(det_j[d]), RT.n0, lth[h], r[h], RT.dr, np.log(V.TH0_NODE), V.DNODE)
                      for h in cand])
        out_i[d] = cand[np.argsort(-v)[:K]]
    return out_i


rows = []
for pid in PIDS:
    tr = V.build_truth(pid); bt = V.build_tile(pid)
    t0 = time.time()
    T2 = P2.T(tr["y"], tr["x"], tr["Y500"], tr["theta500"])
    T4 = P4.T(tr["y"], tr["x"], tr["Y500"], tr["theta500"])
    log(f"\n==================== tile {pid} (painted {time.time() - t0:.1f} s)")
    for mode in V.MODES:
        C1b = bt[f"{mode}_it1_C"] * V.RFFT_TO_PHYS
        F1 = V.MMF(G, C=C1b)
        EX2 = V.Exact(F1, P2)
        q = bt[f"{mode}_it1_q"]; j = bt[f"{mode}_it1_j"]; iy = bt[f"{mode}_it1_iy"]; ix = bt[f"{mode}_it1_ix"]
        sel = np.flatnonzero(q > 5)
        ti = bt[f"{mode}_d_top_idx"]; tex = bt[f"{mode}_d_top_ex"]; Stsz_b = bt[f"{mode}_d_S_tsz"]
        sd = F1.sigma_am[j]
        dyd = iy * V.PIX; dxd = ix * V.PIX
        # filtered tSZ maps (own painting, both templates) at the detections
        D2 = F1.combine(V.tsz_channels(T2, G)); D4 = F1.combine(V.tsz_channels(T4, G))
        S2 = np.zeros(q.size); S4 = np.zeros(q.size)
        for jj in range(15):
            m = j == jj
            if m.any():
                S2[m] = F1.ymap(D2, jj)[iy[m], ix[m]]
                S4[m] = F1.ymap(D4, jj)[iy[m], ix[m]]
        del D2, D4
        t1 = time.time()
        cmix = np.full((sel.size, 4), np.nan); cown = np.full((sel.size, 4), np.nan)
        for a, d in enumerate(sel):
            for k in range(4):
                h = ti[d, k]
                if h < 0 or not np.isfinite(tex[d, k]):
                    continue
                dy = (dyd[d] - tr["y"][h] + 300) % 600 - 300
                dx = (dxd[d] - tr["x"][h] + 300) % 600 - 300
                cmix[a, k] = EX2.contrib(int(j[d]), tr["Y500"][h], tr["theta500"][h], dy, dx, template="nodes")
                cown[a, k] = EX2.contrib(int(j[d]), tr["Y500"][h], tr["theta500"][h], dy, dx, template="own")
        t_ex = time.time() - t1
        cpb = tex[sel, 0]; s = sd[sel]
        dmix = np.abs(cmix[:, 0] - cpb) / s
        down = (cown[:, 0] - cpb) / s
        rel_own = np.abs(cown[:, 0] - cpb) / np.abs(cpb)
        delta_b = Stsz_b[sel] - cpb
        delta_mix = S2[sel] - cmix[:, 0]
        delta_own = S4[sel] - cown[:, 0]
        oth = np.isfinite(tex[sel, 1:]) & np.isfinite(cmix[:, 1:])
        dmix_o = (np.abs(cmix[:, 1:] - tex[sel, 1:]) / s[:, None])[oth]
        down_o = (np.abs(cown[:, 1:] - tex[sel, 1:]) / s[:, None])[oth]
        log(f"--- {mode}: {sel.size} final detections with q > 5 (exact sums {t_ex:.1f} s)")
        log(f"  primary alone, own lattice sum with the build's template mixture vs build c_p: max |d| = {dmix.max():.1e} sigma "
            f"(max relative {np.max(np.abs(cmix[:, 0] - cpb) / np.abs(cpb)):.1e})")
        log(f"  primary alone with the halo's own-size template vs build c_p: median |d| {np.median(np.abs(down)):.1e} sigma, max {np.abs(down).max():.1e} sigma;"
            f" relative median {np.median(rel_own):.1e}, max {rel_own.max():.1e}")
        log(f"  delta (all other halos) own - build: mixture template max |d| {np.abs(delta_mix - delta_b).max() / 1:.1e} arcmin^2 "
            f"= {np.max(np.abs(delta_mix - delta_b) / s):.1e} sigma; own templates median {np.median(np.abs(delta_own - delta_b) / s):.1e} sigma, max {np.max(np.abs(delta_own - delta_b) / s):.1e} sigma")
        log(f"  2nd-4th contributors ({oth.sum()} values): mixture max |d| {dmix_o.max():.1e} sigma; own template median {np.median(down_o):.1e}, max {down_o.max():.1e} sigma")
        # class labels
        real_b = cpb / s >= 2; real_o = cown[:, 0] / s >= 2
        b10_b = delta_b > 0.1 * cpb; b10_o = delta_own > 0.1 * cown[:, 0]
        b20_b = delta_b > 0.2 * cpb; b20_o = delta_own > 0.2 * cown[:, 0]
        log(f"  class flips build -> own numbers (own templates): real {np.sum(real_b != real_o)}, Y boosted > 10% {np.sum(b10_b != b10_o)}, > 20% {np.sum(b20_b != b20_o)}"
            f" (of {sel.size}; build real {real_b.sum()}, b10 {b10_b.sum()}, b20 {b20_b.sum()})")
        # own ranking for the q > 5 detections
        t1 = time.time()
        own_top = own_ranking(F1, dyd[sel], dxd[sel], j[sel], tr, K=4)
        # re-rank own top 4 by exact (own template) sums
        prim_own = np.zeros(sel.size, np.int64); cp_own2 = np.zeros(sel.size)
        for a, d in enumerate(sel):
            vals = []
            for h in own_top[a]:
                dy = (dyd[d] - tr["y"][h] + 300) % 600 - 300
                dx = (dxd[d] - tr["x"][h] + 300) % 600 - 300
                vals.append(EX2.contrib(int(j[d]), tr["Y500"][h], tr["theta500"][h], dy, dx, template="own"))
            prim_own[a] = own_top[a][int(np.argmax(vals))]; cp_own2[a] = max(vals)
        same = prim_own == ti[sel, 0]
        sets = np.array([set(own_top[a][:3]) == set(ti[d, :3]) for a, d in enumerate(sel)])
        log(f"  own ranking (continuous tables, all halos within 60', top 4 re-ranked by exact sums): same primary as the build for "
            f"{same.sum()} of {sel.size}; same top-3 set {sets.sum()} ({time.time() - t1:.1f} s)")
        if (~same).any():
            for a in np.flatnonzero(~same)[:5]:
                d = sel[a]
                log(f"     differs: q = {q[d]:.2f}; build primary {ti[d, 0]} c_p = {cpb[a] / s[a]:.3f} sigma, own primary {prim_own[a]} c = {cp_own2[a] / s[a]:.3f} sigma")
        rows.append(dict(n=sel.size, dmix=dmix.max(), down_med=np.median(np.abs(down)), down_max=np.abs(down).max(),
                         ddel_max=np.max(np.abs(delta_own - delta_b) / s), same=same.sum(), flips=int(np.sum(real_b != real_o) + np.sum(b10_b != b10_o))))
        # ---- literal: paint the primary alone and the rest separately, filter maps, read the pixel (subset)
        if mode == "baseline":
            t1 = time.time()
            sub = sel[np.argsort(-cpb)[:3]].tolist() + sel[np.argsort(-(delta_b / cpb))[:3]].tolist()
            lit = []
            for d in sub:
                h = ti[d, 0]
                Ta = P4.T([tr["y"][h]], [tr["x"][h]], [tr["Y500"][h]], [tr["theta500"][h]], exact=True)
                rest = np.ones(tr["y"].size, bool); rest[h] = False
                Tr = P4.T(tr["y"][rest], tr["x"][rest], tr["Y500"][rest], tr["theta500"][rest])
                va = F1.ymap(F1.combine(V.tsz_channels(Ta, G)), int(j[d]))[iy[d], ix[d]]
                vr = F1.ymap(F1.combine(V.tsz_channels(Tr, G)), int(j[d]))[iy[d], ix[d]]
                a = int(np.flatnonzero(sel == d)[0])
                lit.append((q[d], va / sd[d], cown[a, 0] / sd[d], cpb[a] / sd[d], vr / sd[d], delta_own[a] / sd[d], delta_b[a] / sd[d]))
            log(f"  literal maps for 6 detections (3 brightest primaries, 3 largest delta/c_p): q; c_p [sigma] map / lattice sum / build; delta [sigma] rest map / own / build  ({time.time() - t1:.0f} s)")
            for r_ in lit:
                log("     " + f"q {r_[0]:6.2f}; c_p {r_[1]:.6f} / {r_[2]:.6f} / {r_[3]:.6f}; delta {r_[4]:+.6f} / {r_[5]:+.6f} / {r_[6]:+.6f}")
    del T2, T4

# ---- D4: lattice response anisotropy
log("\n(D4) exact lattice response of a unit-Y500 halo (theta 1') to the scale-1.04' filter at distance r in 4 directions:"
    " max - min over directions, in units of the response at r = 0")
bt = V.build_tile(0)
for lab, F in (("analytic covariance", V.MMF(G, mode="baseline", analytic=True)),
               ("build tile 0 it1_C (binned)", V.MMF(G, C=bt["baseline_it1_C"] * V.RFFT_TO_PHYS))):
    EX = V.Exact(F, P4)
    r0 = EX.contrib(3, 1.0, 1.0, 0.0, 0.0, template="own")
    line = []
    for r in (0.3, 1.0, 2.5, 5.0, 10.0, 30.0):
        v = [EX.contrib(3, 1.0, 1.0, r * np.cos(a), r * np.sin(a), template="own") for a in (0.0, np.pi / 8, np.pi / 4, 0.3)]
        line.append(f"r {r}': {(max(v) - min(v)) / r0:.1e}")
    log(f"  {lab}: " + "; ".join(line))

tot = sum(r["n"] for r in rows)
log(f"\nsummary: {tot} detections q > 5 in {len(PIDS)} tiles x 2 levels; mixture-template c_p max |d| {max(r['dmix'] for r in rows):.1e} sigma; "
    f"own-template c_p median |d| {np.median([r['down_med'] for r in rows]):.1e}, max {max(r['down_max'] for r in rows):.1e} sigma; "
    f"delta max |d| {max(r['ddel_max'] for r in rows):.1e} sigma; same primary {sum(r['same'] for r in rows)}/{tot}; class flips {sum(r['flips'] for r in rows)}")
log(f"done [{time.time() - t00:.0f} s], peak RSS {V.peak_rss_gb():.2f} GB")
