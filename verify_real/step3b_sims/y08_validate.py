"""y08 (sims lens): is the own virtual-cluster kernel what the finder does, and what separates injection tiles from it?
Uses y03's own injections (production finder) and the own kernel (y05); and, for the kernel side of the build's V3,
step 3a's stored injection selection (data_real/step3/selection_arbiter.npz: t03 + a02 clusters, q_true, credited at
q > 5, mean r_inj) with the OWN kernel.
 (A) sparse injections (12 per set, >= 80' apart): painted end to end vs virtual copies (same Y, same sky), cluster by
     cluster; painted vs the kernel expectation (q' in each injection tile's own units), raw and weighted like the
     realistic halos; merges; sigma of sparse injection tiles vs the halo-free sky.
 (B) dense injections (t03/a02-like): sigma_dense / sigma_free; paired selection above q = 4 and 5: painted vs virtual
     through the halo-free filter at the same q' (the kernel's reference), virtual through the dense filter vs through
     the halo-free filter (normalisation only), painted vs virtual through the dense filter (painting); raw and weighted
     like the realistic halos.
 (C) step 3a's t03 + a02 injection selection vs the own kernel (q > 5), raw and weighted.
Errors: tile bootstrap (1000 resamples).  Usage: python y08_validate.py
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import alib as A
import vlib3b as V

log = V.Tee(os.path.join(A.HERE, "y08_validate.log"))
t00 = time.time()
i4 = int(np.argmin(np.abs(A.EDGES - 4.0))); i5 = int(np.argmin(np.abs(A.EDGES - 5.0)))
files = sorted(glob.glob(os.path.join(A.OUT, "inject", "i_*.npz")))
DS = (0.3, 0.6, 1.2, 2.4, 4.8, 9.6)
QB = np.arange(1.5, 9.01, 0.5)


def boot_ratio(num_t, den_t, B=1000, seed=3):
    """sum(num)/sum(den) - 1 and its tile-bootstrap error; num_t, den_t per tile."""
    num_t, den_t = np.asarray(num_t, float), np.asarray(den_t, float)
    W = A.boot_weights(num_t.size, B, seed)
    v = num_t.sum() / den_t.sum() - 1
    vb = (W @ num_t) / (W @ den_t) - 1
    return v, vb.std()


for mode in A.MODES:
    pre = mode + "_"
    kz = np.load(os.path.join(A.OUT, f"kernel_whitened_{mode}.npz"))
    K = kz["K"]
    # realistic-halo density in (theta node, q') for the weights
    recs = A.load_tiles("whitened", mode, keys=("th", "qp"), need_kernel=False)
    th_r = np.concatenate([r["th"] for r in recs]); qp_r = np.concatenate([r["qp"] for r in recs])
    nod_r = np.clip(np.rint((np.log(th_r) - A.LTK[0]) / A.DLT).astype(int), 0, 24)
    H_real = np.histogram2d(nod_r, qp_r, bins=[np.arange(26) - 0.5, QB])[0]
    log(f"\n=================== [{mode}] ===================")
    # ---------------- (A) sparse
    cols = {k: [] for k in ("th", "qinj", "qv", "q_none", "q_closest1", "q_fof10", "tile", "sig")}
    for fn in files:
        d = np.load(fn)
        if pre + "sp_th" not in d.files:
            continue
        n = d[pre + "sp_th"].size
        for k in ("th", "qinj", "qv", "q_none", "q_closest1", "q_fof10"):
            cols[k].append(d[pre + "sp_" + k])
        cols["tile"].append(np.full(n, int(d["tile"])))
        cols["sig"].append(d[pre + "sp_sigratio"][:, 0])
        d.close()
    sp = {k: np.concatenate(v) for k, v in cols.items()}
    nt = len(cols["tile"])
    tiles_sp = np.unique(sp["tile"])
    tix = np.searchsorted(tiles_sp, sp["tile"])
    pc = lambda x: np.bincount(tix, weights=x.astype(float), minlength=tiles_sp.size)
    nsp = sp["th"].size
    log(f"(A) sparse injections: {nsp:,} clusters on {tiles_sp.size} tiles; sigma_inj / sigma_free at 0.5' = "
        f"{sp['sig'].mean() - 1:+.4%} +- {sp['sig'].std() / np.sqrt(sp['sig'].size):.4%} (per set)")
    for e in (4.0, 5.0):
        a = sp["q_none"] > e; b = sp["qv"] > e
        log(f"    painted vs virtual copies (same Y, same sky), q > {e:.0f}: {a.sum():,} vs {b.sum():,} "
            f"(painted only {np.sum(a & ~b)}, virtual only {np.sum(b & ~a)}); ratio - 1 = "
            f"{a.sum() / b.sum() - 1:+.3%} +- {np.sqrt(np.sum(a ^ b)) / b.sum():.3%} (discordant pairs)")
    both = np.isfinite(sp["q_none"]) & np.isfinite(sp["qv"])
    for lo, hi in ((4, 6), (10, 40)):
        s = both & (sp["qv"] > lo) & (sp["qv"] < hi)
        dq = sp["q_none"][s] - sp["qv"][s]
        log(f"    dq = q_painted - q_virtual at q {lo}-{hi}: {dq.mean():+.4f} +- {dq.std():.4f} (n = {s.sum()})")
    Kh = A.K_at(K, sp["th"], sp["qinj"])
    nodes = np.clip(np.rint((np.log(sp["th"]) - A.LTK[0]) / A.DLT).astype(int), 0, 24)
    H_inj = np.histogram2d(nodes, sp["qinj"], bins=[np.arange(26) - 0.5, QB])[0]
    qb = np.clip(np.searchsorted(QB, sp["qinj"], side="right") - 1, 0, QB.size - 2)
    inr = (sp["qinj"] >= QB[0]) & (sp["qinj"] < QB[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        wcell = np.where(H_inj > 0, H_real / H_inj, 0.0)
    wgt = np.where(inr, wcell[nodes, qb], 0.0)
    for e, ie in ((4.0, i4), (5.0, i5)):
        v, err = boot_ratio(pc(sp["q_none"] > e), pc(Kh[:, ie]))
        vw, errw = boot_ratio(pc((sp["q_none"] > e) * wgt), pc(Kh[:, ie] * wgt))
        vv, errv = boot_ratio(pc(sp["qv"] > e), pc(Kh[:, ie]))
        log(f"    painted / kernel expectation - 1, q > {e:.0f}: {v:+.2%} +- {err:.2%} (raw); weighted like the realistic "
            f"halos (q' 1.5-9): {vw:+.2%} +- {errw:.2%}; virtual copies / kernel - 1: {vv:+.2%} +- {errv:.2%}")
    for m in ("closest1", "fof10"):
        ch = np.sum((sp[f"q_{m}"] > 5) != (sp["q_none"] > 5)); chq = np.sum(sp[f"q_{m}"] != sp["q_none"])
        log(f"    merge {m}: clusters whose q > 5 outcome changes {ch} of {nsp:,} (q_obs changes: {chq})")
    # ---------------- (B) dense
    dcols = {k: [] for k in ("th", "qd", "qpaint", "qvf", "qvd", "tile")}
    srat = []
    for fn in files:
        d = np.load(fn)
        if pre + "dn_qd" not in d.files:
            continue
        n = d[pre + "dn_qd"].size
        dcols["th"].append(np.full(n, float(d[pre + "dn_th"])))
        for k in ("qd", "qpaint", "qvf", "qvd"):
            dcols[k].append(d[pre + "dn_" + k])
        dcols["tile"].append(np.full(n, int(d["tile"])))
        srat.append(d[pre + "dn_rd"] / d[pre + "rf"])
        d.close()
    if dcols["th"]:
        dn = {k: np.concatenate(v) for k, v in dcols.items()}
        srat = np.array(srat)
        td = np.unique(dn["tile"]); tdx = np.searchsorted(td, dn["tile"])
        pcd = lambda x: np.bincount(tdx, weights=x.astype(float), minlength=td.size)
        log(f"(B) dense injections: {dn['th'].size:,} clusters on {td.size} tiles (64-225 per tile); sigma_dense / "
            f"sigma_free - 1 at 0.5' / 4.45' / 15': " + " / ".join(
                f"{srat[:, j].mean() - 1:+.3%} +- {srat[:, j].std() / np.sqrt(len(srat)):.3%}" for j in (0, 9, 14)))
        # weights like the realistic halos: realistic density in (nearest dense size, q') over the injection density
        lr = np.log(th_r)
        near = np.argmin(np.abs(lr[:, None] - np.log(np.array(DS))[None, :]), 1)
        Hr = np.histogram2d(near, qp_r, bins=[np.arange(7) - 0.5, QB])[0]
        di = np.argmin(np.abs(np.log(dn["th"])[:, None] - np.log(np.array(DS))[None, :]), 1)
        Hi = np.histogram2d(di, dn["qd"], bins=[np.arange(7) - 0.5, QB])[0]
        qbd = np.clip(np.searchsorted(QB, dn["qd"], side="right") - 1, 0, QB.size - 2)
        okd = (dn["qd"] >= QB[0]) & (dn["qd"] < QB[-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            wd = np.where(okd, np.where(Hi > 0, Hr / Hi, 0.0)[di, qbd], 0.0)
        Khd = A.K_at(K, dn["th"], dn["qd"])
        for e, ie in ((4.0, i4), (5.0, i5)):
            out = []
            for name, num, den in (("painted dense / virtual halo-free filter", dn["qpaint"] > e, dn["qvf"] > e),
                                   ("virtual dense filter / virtual halo-free filter", dn["qvd"] > e, dn["qvf"] > e),
                                   ("painted dense / virtual dense filter", dn["qpaint"] > e, dn["qvd"] > e),
                                   ("painted dense / own kernel", dn["qpaint"] > e, Khd[:, ie])):
                v, er = boot_ratio(pcd(num), pcd(den))
                vw, erw = boot_ratio(pcd(num * wd), pcd(den * wd))
                log(f"    q > {e:.0f}: {name:48s}: {v:+.2%} +- {er:.2%} (raw); weighted like the realistic halos "
                    f"{vw:+.2%} +- {erw:.2%}")
        for sz in DS:
            s = np.isclose(dn["th"], sz)
            if s.sum() == 0:
                continue
            a = dn["qpaint"][s] > 5; b = dn["qvf"][s] > 5
            log(f"    size {sz:4.1f}': {s.sum():5d} clusters; q > 5 painted {a.sum()} vs virtual halo-free {b.sum()} "
                f"({a.sum() / max(b.sum(), 1) - 1:+.2%})")
    # ---------------- (C) step 3a's injections with the own kernel
    sa = np.load(os.path.join(A.ROOT, "data_real", "step3", "selection_arbiter.npz"))
    rinj = sa[f"{mode}_rinj"]
    num_t, den_t, numw_t, denw_t = [], [], [], []
    allq, allK, alld, allw = [], [], [], []
    for sz in DS:
        q = sa[f"{mode}_{sz}_q"]; det = sa[f"{mode}_{sz}_det"]; tl = sa[f"{mode}_{sz}_tile"]
        qp = q / np.interp(np.log(sz), np.log(np.geomspace(0.5, 15.0, 15)), rinj)
        Kq = A.K_at(K, np.full(q.size, sz), qp)[:, i5]
        k = int(np.argmin(np.abs(np.log(np.array(DS)) - np.log(sz))))
        Hs = np.histogram(qp, bins=QB)[0].astype(float)
        lr = np.log(th_r)
        near = np.argmin(np.abs(lr[:, None] - np.log(np.array(DS))[None, :]), 1)
        Hr1 = np.histogram(qp_r[near == k], bins=QB)[0].astype(float)
        qb_ = np.clip(np.searchsorted(QB, qp, side="right") - 1, 0, QB.size - 2)
        w = np.where((qp >= QB[0]) & (qp < QB[-1]), np.where(Hs > 0, Hr1 / np.maximum(Hs, 1), 0.0)[qb_], 0.0)
        ut = np.unique(tl)
        for u in ut:
            s = tl == u
            num_t.append(det[s].sum()); den_t.append(Kq[s].sum())
            numw_t.append((det[s] * w[s]).sum()); denw_t.append((Kq[s] * w[s]).sum())
        v1, e1 = boot_ratio(np.array([det[tl == u].sum() for u in ut]), np.array([Kq[tl == u].sum() for u in ut]))
        log(f"(C) step 3a injections, size {sz}': {q.size:,} clusters on {ut.size} tiles; credited q > 5 / own kernel - 1 = "
            f"{v1:+.2%} +- {e1:.2%}")
    v, e = boot_ratio(num_t, den_t); vw, ew = boot_ratio(numw_t, denw_t)
    log(f"(C) step 3a t03 + a02 pooled, credited q > 5 / own kernel - 1: {v:+.2%} +- {e:.2%} (raw); weighted like the "
        f"realistic halos: {vw:+.2%} +- {ew:.2%}   [build with its kernel: -0.27 +- 0.12% / -0.30 +- 0.12%; weighted "
        f"-0.3 +- 0.5% / -0.9 +- 0.4%]")
log(f"\ny08 done in {time.time() - t00:.0f} s")
