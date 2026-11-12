"""x06 (indep lens), checks (b) and (e): science numbers of the own finder on own fresh skies (x04 outputs), with the
build's definitions (RESULTS.md 'Terms'; t05_analysis.py) re-implemented here, compared with the build's claims.

Selection of isolated clusters for (e): own injections (x05 -> out/selection_mine.npz: fit per size, own r_inj) and, as a
cross-check, the build's selection fit (data_real/step3/selection.npz) with the build's r_inj (its injection files).
Errors: binomial for fractions; tile bootstrap (2000) for count ratios; selection-model refits (40 bootstrap resamples
of the own injection tiles).  Step 2's normalisations (i), (ii), (iii) from data_real/step2 (inputs.npz, kernel_*_effiii)
and step 2's Delta/N from data_real/step2/truth_*.npz.
Usage: python x06_fresh_analysis.py [NT]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
import vlib3 as V

ODF = os.path.join(V.OUT, "fresh")
files = sorted(f for f in os.listdir(ODF) if f.startswith("fresh_"))
NT = int(sys.argv[1]) if len(sys.argv) > 1 else len(files)
files = files[:NT]
log = V.Tee(os.environ.get("X06LOG", os.path.join(V.HERE, "x06_fresh_analysis.log")))
t00 = time.time()
rng = np.random.default_rng(12345)
ZB = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
TH = V.THETAS
NBOOT = 2000
SELM = dict(np.load(os.environ.get("SELMINE", os.path.join(V.OUT, "selection_mine.npz"))))
SELB = dict(np.load(os.path.join(V.BUILD_DATA, "selection.npz")))
TH_INJ = SELM["theta_inj"]

# the build's claimed values (task text / analysis/realistic/step3/t05_analysis.log), baseline, goal
BUILD = {
    "r0_range": ((1.118, 1.176), (1.160, 1.242)), "r1_range": ((1.024, 1.051), (1.035, 1.069)),
    "frac_iii_before": ((0.86, 0.92), (0.84, 0.92)), "frac_i_after": ((0.55, 0.74), (0.67, 0.81)),
    "n_main_frac": None,
    "none": (1.75, 1.50), "frag": (5.65, 5.44), "b10": (10.62, 11.96), "b20": (5.90, 6.80), "needs": (6.04, 6.34),
    "needs_det": (7.07, 7.44), "lost": (6.22, 6.57), "low10": (1.44, 1.72), "merge": (0.45, 0.66), "wrong": (21.91, 20.63),
    "wrong_clean": (20.80, 19.31), "wrong_b10": (31.32, 30.40), "bias": (3.08, 3.53),
    "fp_b10": (8.28, 9.33), "fp_b20": (4.44, 5.16), "fp_boost": (6.00, 6.28), "fp_lost": (4.84, 5.11), "fp_change": (1.22, 1.26),
    "att_frac": (10.3, 11.6), "N1": (-3.39, -4.03), "N1_err_tiles": (0.19, 0.16), "N1_err_sel": (0.84, 0.95),
    "split1": (-0.19, -0.24), "split2": (-3.21, -3.79), "z23": (-12.3, -10.8), "closest10": (-10.02, -13.93), "fof10": (-10.37, -14.68),
    "closest1": (-3.50, -4.16), "closest2": (-3.53, -4.19), "table": (-4.08, -4.27), "it0_it1": ((54337, 69496), (77116, 103739)),
    "dom": ((95.25, 96.74, 89.55, 66.91, 27.87, 3.33), (94.63, 96.35, 87.51, 63.94, 27.18, 0.0)),
    "width_cn": ((0.957, 0.989), (0.941, 0.979)), "r_lit": ((1.50, 1.85), (1.64, 2.13)),
}


def p_sel(fits, q, th):
    P = np.array([ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)) for b, mu, ls in fits])
    x = np.clip(np.log(th), np.log(TH_INJ[0]), np.log(TH_INJ[-1]))
    i = np.clip(np.searchsorted(np.log(TH_INJ), x) - 1, 0, len(TH_INJ) - 2)
    w = (x - np.log(TH_INJ[i])) / np.log(TH_INJ[i + 1] / TH_INJ[i])
    n = np.arange(q.size)
    return (1 - w) * P[i, n] + w * P[i + 1, n]


def p_table(comp, q, th):
    qm = 0.5 * (SELM["qedges"][1:] + SELM["qedges"][:-1])
    Pq = np.array([np.interp(q, np.r_[qm, 9.75, 100.0], np.r_[c, 1.0, 1.0], left=0.0) for c in comp])
    x = np.clip(np.log(th), np.log(TH_INJ[0]), np.log(TH_INJ[-1]))
    i = np.clip(np.searchsorted(np.log(TH_INJ), x) - 1, 0, len(TH_INJ) - 2)
    w = (x - np.log(TH_INJ[i])) / np.log(TH_INJ[i + 1] / TH_INJ[i])
    n = np.arange(q.size)
    return (1 - w) * Pq[i, n] + w * Pq[i + 1, n]


def fit_sel(q, det):
    def nll(p):
        b, mu, ls = p
        pr = np.clip(ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)), 1e-12, 1 - 1e-12)
        return -np.sum(det * np.log(pr) + (1 - det) * np.log(1 - pr))
    return minimize(nll, [0.0, 0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000)).x


def rinj_build(mode):
    d = os.path.join(V.BUILD_DATA, "injections")
    a = [np.load(os.path.join(d, f))["sig"] for f in sorted(os.listdir(d)) if f.startswith(f"inj_{mode}_")]
    return np.mean(a, 0) / V.sigma_Y(mode, TH)


def tile_stats(fn, mode, rinj_m, rinj_b):
    r = dict(np.load(os.path.join(ODF, fn)))
    t = int(r["tile"])
    h = V.draw_tile_halos(7_000_000 + t)
    assert h["y"].size == int(r["nh"]) and abs(h["Y500"].sum() / float(r["Ysum"]) - 1) < 1e-12
    Y, th, z, hy, hx = h["Y500"], h["theta500"], h["z"], h["y"], h["x"]
    pre = mode + "_"
    sigY_s = V.sigma_Y(mode, TH)
    qt = Y / V.sigma_Y(mode, th)
    out = dict(r0=r[pre + "sig0"] / sigY_s, r1=r[pre + "sig1"] / sigY_s, r_cn=r[pre + "sig_cn"] / sigY_s,
               r_lit=r[pre + "sig_lit"] / sigY_s, r_anatsz=r[pre + "sig_anatsz"] / sigY_s,
               mask_cn=r[pre + "sig_cn_mask"] / r[pre + "sig_cn"], width=r[pre + "width_cn"], fsky1=float(r[pre + "fsky1"]))
    sig = r[pre + "sig1"]
    q, j, iy, ix = r[pre + "q"], r[pre + "j"], r[pre + "iy"], r[pre + "ix"]
    d5 = r[pre + "d5"]
    ti, ex = r[pre + "top_idx"], r[pre + "top_ex"]
    prim = ti[:, 0]; cp = ex[:, 0]
    c2 = np.where(np.isfinite(ex[:, 1]), ex[:, 1], 0.0); i2 = ti[:, 1]
    sd = sig[j[d5]]
    q5 = q[d5]
    delta = r[pre + "S_tsz"][d5] - cp
    nz = r[pre + "S_cn"][d5]
    real = cp / sd >= 2.0
    order = np.argsort(-q5)
    seen = set(); main = np.zeros(d5.size, bool)
    for a in order:
        if real[a] and prim[a] not in seen:
            main[a] = True; seen.add(int(prim[a]))
    frag = real & ~main
    credited = np.array(sorted(seen), np.int64)
    fidx, fqa = r[pre + "f_idx"], r[pre + "f_q_alone"]
    qa = fqa[np.searchsorted(fidx, prim)]
    merge = main & (i2 >= 0) & (qt[np.maximum(i2, 0)] > 5) & ~np.isin(i2, credited) & (c2 > 0.1 * cp)
    rth = TH[j[d5]] / th[prim]
    wrong = (rth > 2) | (rth < 0.5)
    out.update(n5=d5.size, n_none=int((~real).sum()), n_frag=int(frag.sum()), n_main=int(main.sum()),
               b10=int((main & (delta > 0.1 * cp)).sum()), b20=int((main & (delta > 0.2 * cp)).sum()),
               needs=int((main & (qa < 5)).sum()), needs_det=int((main & ((cp + nz) / sd < 5)).sum()),
               merge=int(merge.sum()), wrong=int((main & wrong).sum()),
               wrong_clean=int((main & (delta <= 0.1 * cp) & wrong).sum()), n_clean=int((main & (delta <= 0.1 * cp)).sum()),
               wrong_b10=int((main & (delta > 0.1 * cp) & wrong).sum()), bias=float(np.sum((delta / cp)[main])),
               low10=int((main & (delta < -0.1 * cp)).sum()))
    alone5 = fidx[fqa > 5]
    out.update(n_alone5=alone5.size, lost=int((~np.isin(alone5, credited)).sum()))
    # forced photometry (finder's filter, true positions, nearest pixel / scale, fdom = 1)
    fs = sig[r[pre + "f_j"]]
    own = r[pre + "f_own"] / fs
    fn_ = r[pre + "f_S_cn"] / fs
    fdl = (r[pre + "f_S_tsz"] - r[pre + "f_own"]) / fs
    xmx = r[pre + "f_nb_val"][:, 0] / fs
    single = own + fn_ > 5
    blend = (own + fn_ + fdl > 5) & (xmx < own)
    out.update(fp_single=int(single.sum()), fp_blend=int(blend.sum()), fp_b10=int((blend & (fdl > 0.1 * own)).sum()),
               fp_b20=int((blend & (fdl > 0.2 * own)).sum()), fp_boost=int((blend & ~single).sum()),
               fp_lost=int((single & ~blend).sum()), fp_bias=float(np.sum((fdl / own)[blend])))
    # attribution
    xidx, xm = r[pre + "x_idx"], r[pre + "x_max"]
    xof = lambda hh: xm[np.searchsorted(xidx, hh)]
    att = main & (c2 >= 0.1 * cp)
    A, B = prim[att], i2[att]
    out["att"] = dict(ratio=c2[att] / cp[att], qtA=qt[A], qtB=qt[B], xA=xof(A) if A.size else np.zeros(0),
                      xB=xof(B) if B.size else np.zeros(0),
                      dA=V.pdist(hy[A], hx[A], iy[d5][att] * V.PIX, ix[d5][att] * V.PIX),
                      dB=V.pdist(hy[B], hx[B], iy[d5][att] * V.PIX, ix[d5][att] * V.PIX))
    hq3 = np.flatnonzero(qt > 3)
    out["dom"] = dict(ratio=xof(hq3) / qt[hq3], cred=np.isin(hq3, credited), alone5=fqa[np.searchsorted(fidx, hq3)] > 5)
    # counts
    out["Ndet"] = np.histogram(z[credited], ZB)[0].astype(float)
    out["Nalone"] = np.histogram(z[alone5], ZB)[0].astype(float)
    rh = np.interp(np.log(th), np.log(TH), out["r1"])
    sigfin = rh * V.sigma_Y(mode, th)
    qp_m = Y / sigfin * np.interp(np.log(th), np.log(TH), rinj_m)
    qp_b = Y / sigfin * np.interp(np.log(th), np.log(TH), rinj_b)
    kk = qp_m > 1.4
    out["sel_q"], out["sel_th"], out["sel_z"] = qp_m[kk], th[kk], z[kk]
    out["Nexp"] = np.histogram(z[kk], ZB, weights=p_sel(SELM[mode + "_fit"], qp_m[kk], th[kk]))[0]
    kb = qp_b > 1.4
    out["Nexp_bsel"] = np.histogram(z[kb], ZB, weights=p_sel(SELB[mode + "_fit"], qp_b[kb], th[kb]))[0]
    out["Nexp_tab"] = np.histogram(z[kk], ZB, weights=p_table(SELM[mode + "_comp"], qp_m[kk], th[kk]))[0]
    # sigma_Y normalisation variant
    qsy = q5 * out["r1"][j[d5]]
    credY = np.unique(prim[(qsy > 5) & real])
    out["Ndet_sY"] = np.histogram(z[credY], ZB)[0].astype(float)
    qy_ = Y / V.sigma_Y(mode, th)
    ky = qy_ > 1.4
    out["Nexp_sY"] = np.histogram(z[ky], ZB, weights=p_sel(SELM[mode + "_fit"], qy_[ky], th[ky]))[0]
    # merges
    yd, xd = iy * V.PIX, ix * V.PIX
    for mn in V.MERGES:
        k = V.apply_merge(yd, xd, q, mn, 5.0)
        pos = np.searchsorted(d5, k)
        out["n5all_" + mn] = k.size
        kr = pos[real[pos]]
        out["Ndet_" + mn] = np.histogram(z[np.unique(prim[kr])], ZB)[0].astype(float)
        out["n5_" + mn] = kr.size
    out["n5_it0"] = int((r[pre + "it0_q"] > 5).sum())
    out["fp_dummy"] = 0
    out["timing"] = r[pre + "timing"]; out["t_total"] = float(r["t_total"]); out["rss"] = float(r["peak_rss_gb"])
    return out


def boot_ratio(num, den, nboot=NBOOT):
    tot = num.sum(0) / den.sum(0) - 1
    T = num.sum() / den.sum() - 1
    b = rng.integers(0, num.shape[0], (nboot, num.shape[0]))
    nb_ = num[b].sum(1); db = den[b].sum(1)
    return tot, (nb_ / db - 1).std(0), T, (nb_.sum(1) / db.sum(1) - 1).std()


def frac(a, b):
    p = a / b
    return 100 * p, 100 * np.sqrt(p * (1 - p) / b)


def cmp_line(lab, mine, e, build):
    """mine +- e (percent) against the build's value; z = difference / e (own error only: the build's are smaller)."""
    return f"  {lab:62s}: own {mine:7.2f} +- {e:.2f}   build {build:7.2f}   (own - build) / own error = {(mine - build) / e:+.1f}"


if __name__ == "__main__":
    log(f"x06: {len(files)} own fresh tiles per noise level ({100 * len(files)} deg^2), own finder")
    s2 = dict(np.load(os.path.join(V.ROOT, "data_real", "step2", "inputs.npz")))
    summary = {}
    for mi, mode in enumerate(V.MODES):
        rinj_m = SELM[mode + "_rinj"]; rinj_b = rinj_build(mode)
        T = [tile_stats(f, mode, rinj_m, rinj_b) for f in files]
        A = {k: np.array([t_[k] for t_ in T]) for k in T[0] if k not in ("att", "dom", "sel_q", "sel_th", "sel_z")}
        log(f"\n==================== {mode}: {len(T)} tiles; r_inj (own injections) {rinj_m.min():.4f}-{rinj_m.max():.4f}, build {rinj_b.min():.4f}-{rinj_b.max():.4f}")
        # ---------------- (4a)
        ri = np.interp(np.log(TH), np.log(s2[mode + "_th"]), s2[mode + "_r_i"])
        rii = np.interp(np.log(TH), np.log(s2[mode + "_th"]), s2[mode + "_r_ii"])
        k3 = np.load(os.path.join(V.ROOT, "data_real", "step2", f"kernel_{mode}_effiii.npz"))
        riii = np.interp(np.log(TH), np.log(k3["th"]), k3["r"])
        log("(4a) r(theta) = sigma_finder / sigma_Y, mean [tile-to-tile std]")
        log("  theta [']                           " + " ".join(f"{v:7.2f}" for v in TH))
        for lab, key in (("finder, before iteration", "r0"), ("finder, after iteration", "r1"),
                         ("CMB + noise only, estimated", "r_cn"), ("analytic CMB+noise + tile tSZ power", "r_anatsz"),
                         ("literal binary f_sky iteration", "r_lit")):
            log(f"  {lab:36s}" + " ".join(f"{v:7.4f}" for v in A[key].mean(0)))
            log(f"  {'   std':36s}" + " ".join(f"{v:7.4f}" for v in A[key].std(0)))
        log(f"  {'mask effect on CMB + noise alone':36s}" + " ".join(f"{v:7.4f}" for v in A["mask_cn"].mean(0)))
        log(f"  {'pixel std of the CMB+noise S/N maps':36s}" + " ".join(f"{v:7.4f}" for v in A["width"].mean(0)))
        log(f"  {'   compare 1 / r_after':36s}" + " ".join(f"{v:7.4f}" for v in 1 / A["r1"].mean(0)))
        r0m, r1m = A["r0"].mean(0), A["r1"].mean(0)
        fi = (r1m - 1) / (ri - 1); fiii0 = (r0m - 1) / (riii - 1); fii = (r1m - 1) / (rii - 1)
        log(f"  (r_after - 1) / (r_(i) - 1):    " + " ".join(f"{v:5.2f}" for v in fi))
        log(f"  (r_after - 1) / (r_(ii) - 1):   " + " ".join(f"{v:5.2f}" for v in fii))
        log(f"  (r_before - 1) / (r_(iii) - 1): " + " ".join(f"{v:5.2f}" for v in fiii0))
        rat_ana = A["r_anatsz"].mean(0) / r0m - 1
        log(f"  analytic + tile tSZ vs r_before: {100 * rat_ana.min():+.2f}% .. {100 * rat_ana.max():+.2f}%; masked fraction mean {100 * (1 - A['fsky1'].mean()):.1f}%")
        b_ = BUILD
        log(f"  SUMMARY r: before {r0m.min():.3f}-{r0m.max():.3f} (build {b_['r0_range'][mi][0]}-{b_['r0_range'][mi][1]}); "
            f"after {r1m.min():.3f}-{r1m.max():.3f} (build {b_['r1_range'][mi][0]}-{b_['r1_range'][mi][1]}); "
            f"fraction of (iii) excess before {fiii0.min():.2f}-{fiii0.max():.2f} (build {b_['frac_iii_before'][mi]}); "
            f"fraction of (i) excess after {fi.min():.2f}-{fi.max():.2f} (build {b_['frac_i_after'][mi]}); "
            f"literal binary {A['r_lit'].mean(0).min():.2f}-{A['r_lit'].mean(0).max():.2f} (build {b_['r_lit'][mi]}); "
            f"CMB+noise width {A['width'].mean(0).min():.3f}-{A['width'].mean(0).max():.3f} (build {b_['width_cn'][mi]})")
        log(f"  standard error of the tile mean r_after: {(A['r1'].std(0) / np.sqrt(len(T))).max():.4f}; r_before: {(A['r0'].std(0) / np.sqrt(len(T))).max():.4f}")
        n5 = A["n5"].sum(); nm = A["n_main"].sum()
        log(f"  q > 5 detections before -> after the iteration: {A['n5_it0'].sum()} -> {n5} ({100 * (n5 / A['n5_it0'].sum() - 1):+.1f}%; "
            f"build {b_['it0_it1'][mi][0]} -> {b_['it0_it1'][mi][1]} ({100 * (b_['it0_it1'][mi][1] / b_['it0_it1'][mi][0] - 1):+.1f}%))")
        # ---------------- (4b)
        log(f"(4b) final detections q > 5: {n5} ({n5 / len(T):.1f} per tile); main: {nm}")
        rows = [("no real cluster (c_p < 2 sigma), of all q > 5", A["n_none"].sum(), n5, "none"),
                ("fragments, of all q > 5", A["n_frag"].sum(), n5, "frag"),
                ("Y boosted > 10%, of main", A["b10"].sum(), nm, "b10"), ("Y boosted > 20%, of main", A["b20"].sum(), nm, "b20"),
                ("detected only thanks to neighbours (isolated search), of main", A["needs"].sum(), nm, "needs"),
                ("  same at the detection's own position and scale", A["needs_det"].sum(), nm, "needs_det"),
                ("lost (q_alone > 5, not credited), of q_alone > 5", A["lost"].sum(), A["n_alone5"].sum(), "lost"),
                ("Y lowered by > 10%, of main", A["low10"].sum(), nm, "low10"), ("merges, of main", A["merge"].sum(), nm, "merge"),
                ("wrong size, of main", A["wrong"].sum(), nm, "wrong"),
                ("wrong size among clean main", A["wrong_clean"].sum(), A["n_clean"].sum(), "wrong_clean"),
                ("wrong size among Y boosted > 10%", A["wrong_b10"].sum(), A["b10"].sum(), "wrong_b10")]
        for lab, a, b, key in rows:
            p, e = frac(a, b)
            log(cmp_line(lab, p, e, b_[key][mi]))
            summary[(mode, key)] = (p, e, b_[key][mi])
        log(f"  mean delta / c_p of main: own {100 * A['bias'].sum() / nm:+.2f}% (build {b_['bias'][mi]:+.2f}%)")
        nb_, ns_ = A["fp_blend"].sum(), A["fp_single"].sum()
        log(f"  forced photometry at the true positions (finder's filter, nearest scale, fdom = 1; q_true > 2): blended {nb_}, single {ns_}")
        for lab, a, b, key in (("  Y boosted > 10%", A["fp_b10"].sum(), nb_, "fp_b10"), ("  Y boosted > 20%", A["fp_b20"].sum(), nb_, "fp_b20"),
                               ("  boosted over the threshold", A["fp_boost"].sum(), nb_, "fp_boost"), ("  lost (of single)", A["fp_lost"].sum(), ns_, "fp_lost")):
            p, e = frac(a, b)
            log(cmp_line(lab, p, e, b_[key][mi]))
            summary[(mode, key)] = (p, e, b_[key][mi])
        tot, eb, Tt, eT = boot_ratio(A["fp_blend"][:, None].astype(float), A["fp_single"][:, None].astype(float))
        log(cmp_line("  change in detections (blended / single - 1)", 100 * Tt, 100 * eT, b_["fp_change"][mi]))
        log(f"  mean Y bias (forced): {100 * A['fp_bias'].sum() / nb_:+.2f}%")
        for key, kf in (("b10", "fp_b10"), ("lost", "fp_lost")):
            pass
        pb10, eb10 = frac(A["b10"].sum(), nm); pf10, ef10 = frac(A["fp_b10"].sum(), nb_)
        pl, el = frac(A["lost"].sum(), A["n_alone5"].sum()); pfl, efl = frac(A["fp_lost"].sum(), ns_)
        pn, en = frac(A["needs"].sum(), nm); pfb, efb = frac(A["fp_boost"].sum(), nb_)
        log(f"  search minus true position: Y > 10% {pb10 - pf10:+.2f} pp (build +2.3 / +2.6); lost {pl - pfl:+.2f} pp (build +1.4 / +1.5); "
            f"only thanks to neighbours {pn - pfb:+.2f} pp (build ~0)")
        # ---------------- (4c)
        att = {k: np.concatenate([t_["att"][k] for t_ in T]) for k in T[0]["att"]}
        rat = att["ratio"]
        p, e = frac(rat.size, nm)
        log(f"(4c) main detections with a second contributor >= 10% of the primary: {rat.size} ({p:.1f}% of main; build {b_['att_frac'][mi]}%)")
        for lo, hi in ((0.1, 0.25), (0.25, 0.5), (0.5, 1.01)):
            m = (rat >= lo) & (rat < hi)
            if not m.any():
                continue
            line = [f"c_B/c_A {lo}-{min(hi, 1)}: N = {m.sum()}; A has the larger true q in {100 * np.mean(att['qtA'][m] > att['qtB'][m]):.0f}%; "
                    f"closer to A in {100 * np.mean(att['dA'][m] < att['dB'][m]):.0f}%"]
            for fd in (0.5, 1.0, 2.0):
                ca = att["xA"][m] < fd * att["qtA"][m]; cb = att["xB"][m] < fd * att["qtB"][m]
                line.append(f"fdom {fd}: A only {100 * np.mean(ca & ~cb):.0f}%, B only {100 * np.mean(~ca & cb):.0f}%, both {100 * np.mean(ca & cb):.0f}%, neither {100 * np.mean(~ca & ~cb):.0f}%")
            log("  " + "\n     ".join(line))
        dom = {k: np.concatenate([t_["dom"][k] for t_ in T]) for k in T[0]["dom"]}
        log("  effective dominance: halos q_true > 3 with q_alone > 5, by x_max / q_true at the true position (step 1's filter); credited fraction [build]:")
        bd = b_["dom"][mi]
        for ib, (lo, hi) in enumerate(((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 4.0), (4.0, 1e9))):
            m = dom["alone5"] & (dom["ratio"] >= lo) & (dom["ratio"] < hi)
            if m.sum():
                p, e = frac(dom["cred"][m].sum(), m.sum())
                log(f"     x_max/q in [{lo}, {hi if hi < 1e8 else 'inf'}): N = {m.sum():6d}, credited {p:6.2f} +- {e:.2f}%" + (f"   [build {bd[ib]:.2f}%]" if ib < 6 else ""))
        # crossing point of 50%: logistic fit of credited vs x_max/q for x/q in 0.25-1.5
        m = dom["alone5"] & (dom["ratio"] >= 0.25) & (dom["ratio"] < 1.5)
        xx, yy = dom["ratio"][m], dom["cred"][m].astype(float)
        nll = lambda p: -np.sum(yy * np.log(np.clip(1 / (1 + np.exp((xx - p[0]) / np.exp(p[1]))), 1e-12, 1)) +
                                (1 - yy) * np.log(np.clip(1 - 1 / (1 + np.exp((xx - p[0]) / np.exp(p[1]))), 1e-12, 1)))
        pf = minimize(nll, [0.7, np.log(0.1)], method="Nelder-Mead").x
        bs = []
        for _ in range(200):
            ii = rng.integers(0, xx.size, xx.size)
            nll_b = lambda p: -np.sum(yy[ii] * np.log(np.clip(1 / (1 + np.exp((xx[ii] - p[0]) / np.exp(p[1]))), 1e-12, 1)) +
                                      (1 - yy[ii]) * np.log(np.clip(1 - 1 / (1 + np.exp((xx[ii] - p[0]) / np.exp(p[1]))), 1e-12, 1)))
            bs.append(minimize(nll_b, pf, method="Nelder-Mead").x[0])
        e50 = 0.5 * (np.percentile(bs, 84) - np.percentile(bs, 16))
        log(f"  50% crossing of the credited fraction (logistic fit, x_max/q 0.25-1.5, N = {m.sum()}): x_max/q = {pf[0]:.3f} +- {e50:.3f}"
            f" (bootstrap 16-84%; logistic width {np.exp(pf[1]):.3f})  (build: 'fdom ~ 0.7')")
        summary[(mode, "fdom50")] = (pf[0], e50, 0.7)
        # ---------------- (4d)
        log("(4d) count change at q > 5: credited halos vs the unit-noise selection of isolated clusters; tile bootstrap")
        log("  z bins: " + ", ".join(f"{a}-{b}" for a, b in zip(ZB[:-1], ZB[1:])))
        res4d = {}
        for lab, nk, ek in (("finder normalisation, OWN selection (own injections)", "Ndet", "Nexp"),
                            ("finder normalisation, BUILD's selection fit and r_inj", "Ndet", "Nexp_bsel"),
                            ("finder normalisation, own binned completeness table", "Ndet", "Nexp_tab"),
                            ("split 1: credited vs isolated search (neighbours, fixed noise)", "Ndet", "Nalone"),
                            ("split 2: isolated search vs own unit-noise selection", "Nalone", "Nexp")):
            tot, eb, Tt, eT = boot_ratio(A[nk], A[ek])
            res4d[lab] = (Tt, eT, tot, eb)
            log(f"  {lab}: N {A[nk].sum():.0f} vs {A[ek].sum():.1f}; Delta/N = {100 * Tt:+.2f} +- {100 * eT:.2f}%")
            log("     per z bin: " + ", ".join(f"{100 * a:+.2f}+-{100 * e:.2f}%" for a, e in zip(tot, eb)))
        # selection-model uncertainty: refits on bootstrap resamples of the own injection tiles
        sq = np.concatenate([t_["sel_q"] for t_ in T]); sth = np.concatenate([t_["sel_th"] for t_ in T]); sz = np.concatenate([t_["sel_z"] for t_ in T])
        Ndt = A["Ndet"].sum(0)
        bres = []
        for _ in range(40):
            fits = []
            for thv in TH_INJ:
                qi, qo, tl = SELM[f"{mode}_{thv}_q"], SELM[f"{mode}_{thv}_qobs"], SELM[f"{mode}_{thv}_tile"]
                ut = np.unique(tl); pick = rng.choice(ut, ut.size)
                ii = np.concatenate([np.flatnonzero(tl == u) for u in pick])
                fits.append(fit_sel(qi[ii], (qo[ii] > 5).astype(float)))
            bres.append(np.histogram(sz, ZB, weights=p_sel(np.array(fits), sq, sth))[0])
        bres = np.array(bres)
        esel_T = (Ndt.sum() / bres.sum(1) - 1).std(); esel = (Ndt[None, :] / bres - 1).std(0)
        Tt, eT, tot, eb = res4d["finder normalisation, OWN selection (own injections)"]
        log(f"  selection-model uncertainty (40 refits): {100 * esel_T:.2f}%; per z bin " + ", ".join(f"{100 * e:.2f}%" for e in esel))
        log(f"  RESULT N1: Delta/N = {100 * Tt:+.2f} +- {100 * eT:.2f} (tiles) +- {100 * esel_T:.2f}% (selection)   "
            f"[build {b_['N1'][mi]:+.2f} +- {b_['N1_err_tiles'][mi]:.2f} +- {b_['N1_err_sel'][mi]:.2f}%]; "
            f"difference / combined error = {(100 * Tt - b_['N1'][mi]) / np.sqrt((100 * eT) ** 2 + (100 * esel_T) ** 2 + b_['N1_err_tiles'][mi] ** 2 + b_['N1_err_sel'][mi] ** 2):+.1f}")
        log(f"  RESULT N3 (z 2-3): {100 * tot[-1]:+.1f} +- {100 * np.hypot(eb[-1], esel[-1]):.1f}%  [build {b_['z23'][mi]:+.1f}%]")
        s1 = res4d["split 1: credited vs isolated search (neighbours, fixed noise)"]
        s2_ = res4d["split 2: isolated search vs own unit-noise selection"]
        bsel = res4d["finder normalisation, BUILD's selection fit and r_inj"]
        log(f"  RESULT N2: split 1 {100 * s1[0]:+.2f} +- {100 * s1[1]:.2f}% [build {b_['split1'][mi]:+.2f}]; split 2 {100 * s2_[0]:+.2f} +- {100 * s2_[1]:.2f}% "
            f"(+- {100 * esel_T:.2f} selection) [build {b_['split2'][mi]:+.2f}]")
        log(f"  with the build's selection fit on the own skies: {100 * bsel[0]:+.2f} +- {100 * bsel[1]:.2f}%; own binned table: "
            f"{100 * res4d['finder normalisation, own binned completeness table'][0]:+.2f}% [build's table variant {b_['table'][mi]:+.2f}%]")
        summary[(mode, "N1")] = (100 * Tt, 100 * np.hypot(eT, esel_T), b_["N1"][mi])
        summary[(mode, "split1")] = (100 * s1[0], 100 * s1[1], b_["split1"][mi])
        summary[(mode, "split2")] = (100 * s2_[0], 100 * np.hypot(s2_[1], esel_T), b_["split2"][mi])
        for mn in V.MERGES:
            tot, eb, Tt, eT = boot_ratio(A["Ndet_" + mn], A["Nexp"])
            rem = 100 * (A["Ndet_" + mn].sum() / A["Ndet"].sum() - 1)
            bv = {"closest1": b_["closest1"][mi], "closest2": b_["closest2"][mi], "closest10": b_["closest10"][mi], "fof10": b_["fof10"][mi]}.get(mn, b_["N1"][mi])
            log(f"  merge {mn:9s}: credited {A['Ndet_' + mn].sum():.0f} ({rem:+.2f}% vs no merge); Delta/N = {100 * Tt:+.2f} +- {100 * eT:.2f}%  [build {bv:+.2f}%]"
                f"; catalogue entries q > 5 {A['n5all_' + mn].sum()}")
            summary[(mode, "merge_" + mn)] = rem
        for v in ("f1", "effi", "effii", "effiii"):
            d = np.load(os.path.join(V.ROOT, "data_real", "step2", f"truth_{mode}_{v}.npz"))
            log(f"  step 2 ({v}): Delta/N = {100 * d['D_q5'].sum() / d['Nn_q5'].sum():+.2f}%")
        tm = A["timing"]
        log(f"  compute: per tile and level find {tm[:, 0].mean():.1f} s, diagnostics {tm[:, 1].mean():.1f} s, decomposition {tm[:, 2].mean():.1f} s, "
            f"forced/isolated {tm[:, 3].mean():.1f} s; per tile (both levels, painting incl.) {A['t_total'].mean():.1f} s; worker peak RSS max {A['rss'].max():.2f} GB")
    log("\nSUMMARY own vs build (percent; z = (own - build) / own error):")
    for (mode, key), v in summary.items():
        if isinstance(v, tuple):
            log(f"  {mode:8s} {key:14s} own {v[0]:+8.3f} +- {v[1]:.3f}   build {v[2]:+8.3f}   z {(v[0] - v[2]) / v[1]:+.1f}")
        else:
            log(f"  {mode:8s} {key:14s} own {v:+8.3f}")
    log(f"\ndone [{time.time() - t00:.0f} s]")
