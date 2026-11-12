"""Step 3 / t05: science numbers on the realistic tiles (all halos M500 > 1e13), SO baseline and goal.

Inputs: data_real/step3/tiles/tile_PID.npz (t04), truth/truth_PID.npz, selection.npz and injections/ (t03), step 2's
noise-normalisation tables (data_real/step2/inputs.npz, kernel_*_effiii.npz) and truth vectors, step 1's rho tables
(data_real/rho_mmf_*.npz, read with analysis/blending/blendlib.rho_at).

Definitions (final catalogue = iteration 1 of the finder; q = q_obs of the finder; sigma = the finder's sigma at the
detection scale; c_p = exact lattice contribution of the largest contributor 'p' (the primary); delta = filtered tSZ
minus c_p (all other halos); n = filtered CMB + noise):
  no real cluster   c_p < 2 sigma;
  fragment          the primary is also the primary of a higher-q detection (q > 5);
  Y boosted > X%    delta > X% of c_p;
  only thanks to neighbours   q_alone < 5, q_alone = isolated search: max over 13 x 13 pixels around the primary's true
                    position and all scales of (primary alone + CMB + noise) / sigma (also given: (c_p + n)/sigma < 5
                    at the detection itself);
  merge             the second contributor has q_true > 5, gives > 10% of c_p and is not the primary of any q > 5
                    detection;
  wrong size        theta_det / theta500(primary) outside [1/2, 2];
  lost              q_alone > 5 but not the primary of any q > 5 detection (fraction of the halos with q_alone > 5).
Forced photometry (true position, nearest pixel and scale, the finder's filter) gives step 1's P(D) classes on the same
skies: q_own = own/sigma, n, delta; dominant if the largest single neighbour < q_own (fdom = 1).
Counts: halos credited with a q > 5 detection (each halo once, real clusters only) against
N_exp = sum over all halos of P_det(q', theta500), P_det = the finder's own selection of isolated clusters (t03 fit),
q' = Y500 / sigma_finder(theta500) of the tile (or Y500 / sigma_Y for the noise-only normalisation).
Arbiter fix (2026-11-12): a credited detection needs c_p >= 2 sigma, and c_p / sigma <= Y500 / sigma_finder(theta500),
so P_det = 0 exactly below q_inj = 2 r_inj.  The fitted form Phi((sqrt(q^2 + b) - 5 - mu) / s) keeps a floor there
(b < 0 clips q^2 + b at 0), which the ~70,000 faint halos per tile turned into 0.8 (baseline) / 2.2 (goal) spurious
expected detections per tile.  P_det is now set to 0 below the credit limit.  The nominal calibration is the arbiter's
pooled injection set (t03 + a02, data_real/step3/selection_arbiter.npz) when present; the build's t03 calibration,
with and without the cut, is printed for reference.
Usage: python t05_analysis.py [SUB (tiles)] [NT (324)]; outputs t05_SUB[_nNT]_MODE.npz.
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import numba as nb
from scipy.special import ndtr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                  # noqa: E402
sys.path.insert(0, f"{S.ROOT}/analysis/blending")
import blendlib as bl                              # noqa: E402
F = S.F
t0 = time.time()
SUB = sys.argv[1] if len(sys.argv) > 1 else "tiles"
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 324
TD = os.path.join(S.OUT, SUB)
ZB = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
NBOOT = 2000
rng = np.random.default_rng(11)


# ------------------------------------------------------------------ selection function of isolated clusters (t03)
SEL = dict(np.load(os.path.join(S.OUT, "selection.npz")))
TH_INJ = SEL["theta_inj"]
_SA = os.path.join(S.OUT, "selection_arbiter.npz")
SELA = dict(np.load(_SA)) if os.path.exists(_SA) else None
CAL = "arbiter (t03 + a02 pooled)" if SELA is not None else "build (t03)"


def fit_model(q, det, x0=(3.0, 0.0, 0.0)):
    """P(q_obs > 5 | q_inj) = Phi((sqrt(q^2 + b) - 5 - mu) / s), maximum likelihood (as in t03)."""
    from scipy.optimize import minimize

    def nll(p):
        b, mu, ls = p
        pr = np.clip(ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)), 1e-12, 1 - 1e-12)
        return -np.sum(det * np.log(pr) + (1 - det) * np.log(1 - pr))
    return minimize(nll, list(x0), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000)).x


def p_table(mode, q, th):
    """Alternative: the binned completeness of t03, linear in q and in ln theta (clamped)."""
    comp, qe = SEL[mode + "_comp"], SEL["qedges"]
    qm = 0.5 * (qe[1:] + qe[:-1])
    Pq = np.array([np.interp(q, np.r_[qm, 9.75, 100.0], np.r_[c, 1.0, 1.0], left=0.0) for c in comp])
    x = np.clip(np.log(th), np.log(TH_INJ[0]), np.log(TH_INJ[-1]))
    i = np.clip(np.searchsorted(np.log(TH_INJ), x) - 1, 0, len(TH_INJ) - 2)
    w = (x - np.log(TH_INJ[i])) / np.log(TH_INJ[i + 1] / TH_INJ[i])
    n = np.arange(q.size)
    return (1 - w) * Pq[i, n] + w * Pq[i + 1, n]


def p_sel(mode, q, th, fits=None, qmin=None):
    """P(q_obs > 5) of an isolated cluster with q_inj = q (Y / sigma_Y on noise-only tiles) and size th: the fit per
    theta_inj (nominal calibration unless fits are given), interpolated linearly in ln theta between the injected sizes
    (clamped outside); 0 below the credit limit qmin (= 2 r_inj) when given."""
    if fits is None:
        fits = SELA[mode + "_fit"] if SELA is not None else SEL[mode + "_fit"]
    P = np.array([ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)) for b, mu, ls in fits])   # (6, n)
    x = np.clip(np.log(th), np.log(TH_INJ[0]), np.log(TH_INJ[-1]))
    i = np.clip(np.searchsorted(np.log(TH_INJ), x) - 1, 0, len(TH_INJ) - 2)
    w = (x - np.log(TH_INJ[i])) / np.log(TH_INJ[i + 1] / TH_INJ[i])
    n = np.arange(q.size)
    out = (1 - w) * P[i, n] + w * P[i + 1, n]
    if qmin is not None:
        out = np.where(q < qmin, 0.0, out)
    return out


def p_table_arb(mode, q, th, qmin):
    """The pooled binned completeness (t03 + a02), linear in q between bin centres and in ln theta (clamped); 0 below
    qmin, the pooled fit above the last bin."""
    k, n, qe = SELA[mode + "_tab_k"], SELA[mode + "_tab_n"], SELA["qedges"]
    qc = 0.5 * (qe[1:] + qe[:-1])
    Pt = k / np.maximum(n, 1)
    hi = q > qc[-1]
    Pq = np.array([np.interp(q, qc, c) for c in Pt])
    x = np.clip(np.log(th), np.log(TH_INJ[0]), np.log(TH_INJ[-1]))
    i = np.clip(np.searchsorted(np.log(TH_INJ), x) - 1, 0, len(TH_INJ) - 2)
    w = (x - np.log(TH_INJ[i])) / np.log(TH_INJ[i + 1] / TH_INJ[i])
    nn = np.arange(q.size)
    out = (1 - w) * Pq[i, nn] + w * Pq[i + 1, nn]
    out[hi] = p_sel(mode, q[hi], th[hi])
    return np.where(q < qmin, 0.0, out)


def r_inj(mode):
    """sigma_finder / sigma_Y on the injection tiles, mean over tiles, at the 15 scales."""
    d = os.path.join(S.OUT, "injections")
    a = [np.load(os.path.join(d, f))["sig"] for f in sorted(os.listdir(d)) if f.startswith(f"inj_{mode}_")]
    return np.mean(a, 0) / S.sigma_Y(mode, F.THETAS15)


def interp_lnth(vals, th, nodes=F.THETAS15):
    return np.interp(np.log(th), np.log(nodes), vals)


# ------------------------------------------------------------------ step 1 analytic rho: dominance at true positions
@nb.njit(cache=True)
def true_pos_neighbours(idx, ys, order, hy, hx, hY, hth, sA, rho, lth0, dlth, RB, box, rcut, out_max, out_arg, out_sum):
    """For each halo idx[a]: largest single neighbour contribution x = Y_B rho(thA, thB, s) / sigma_Y(thA) within rcut
    (step 1's MMF at the true position and true size) and the sum of all of them.  ys = sorted y, order = argsort(y)."""
    n = ys.size
    for a in range(idx.size):
        h = idx[a]
        best = -1e300; barg = -1; ssum = 0.0
        for sh in (-box, 0.0, box):
            lo = np.searchsorted(ys, hy[h] - rcut + sh)
            hi = np.searchsorted(ys, hy[h] + rcut + sh)
            for k in range(lo, hi):
                b = order[k]
                if b == h:
                    continue
                dy = (hy[b] - hy[h] + 0.5 * box) % box - 0.5 * box
                dx = (hx[b] - hx[h] + 0.5 * box) % box - 0.5 * box
                if abs(dx) > rcut:
                    continue
                r = np.sqrt(dx * dx + dy * dy)
                if r > rcut:
                    continue
                x = hY[b] * bl.rho_at(rho, lth0, dlth, RB, hth[h], hth[b], r) / sA[a]
                ssum += x
                if x > best:
                    best = x; barg = b
        out_max[a] = best; out_arg[a] = barg; out_sum[a] = ssum


RHO = {}
for m in S.MODES:
    d = np.load(f"{S.ROOT}/data_real/rho_mmf_{m}.npz")
    RHO[m] = (d["rho"], float(np.log(d["th"][0])), float(np.log(d["th"][1] / d["th"][0])), float(d["R_BEAM"]))


# ------------------------------------------------------------------ per tile
def tile_stats(pid, mode, rinj, rinj_b):
    r = dict(np.load(os.path.join(TD, f"tile_{pid:03d}.npz")))
    t = dict(np.load(os.path.join(S.OUT, "truth", f"truth_{pid:03d}.npz")))
    th_s = r["thetas"]
    sigY_s = S.sigma_Y(mode, th_s)
    out = dict(r0=r[f"{mode}_it0_sig"] / sigY_s, r1=r[f"{mode}_it1_sig"] / sigY_s,
               r_cn=r[f"{mode}_sig_cn_est"] / sigY_s, r_lit=r[f"{mode}_sig_lit_it1"] / sigY_s,
               r_anatsz=r[f"{mode}_sig_ana_tsz"] / sigY_s,
               mask_cn=r[f"{mode}_sig_cn_mask"] / r[f"{mode}_sig_cn_est"] if f"{mode}_sig_cn_mask" in r else np.ones(th_s.size),
               fsky1=float(r[f"{mode}_it1_fsky"]))
    sig = r[f"{mode}_it1_sig"]
    j, q = r[f"{mode}_it1_j"], r[f"{mode}_it1_q"]
    iy, ix = r[f"{mode}_it1_iy"], r[f"{mode}_it1_ix"]
    ex = r[f"{mode}_d_top_ex"]; ti = r[f"{mode}_d_top_idx"]
    sd = sig[j]
    qt = t["q_" + mode].astype(float); z = t["z"].astype(float); th500 = t["theta500"].astype(float)
    Y = t["Y500"].astype(float)
    sel5 = q > 5
    prim = ti[:, 0]
    cp = ex[:, 0]
    c2 = np.where(np.isfinite(ex[:, 1]), ex[:, 1], 0.0)
    delta = r[f"{mode}_d_S_tsz"] - cp
    nz = r[f"{mode}_d_S_cmb"] + r[f"{mode}_d_S_noise"]
    real = cp / sd >= 2.0
    # fragments: per primary keep the highest-q detection among the real q > 5 ones
    d5 = np.flatnonzero(sel5 & real)
    o = d5[np.argsort(-q[d5])]
    seen = set(); main = np.zeros(q.size, bool)
    for dd in o:
        if prim[dd] not in seen:
            main[dd] = True; seen.add(prim[dd])
    frag = sel5 & real & ~main
    credited = np.array(sorted(seen), dtype=np.int64)            # halos credited with a q > 5 detection
    qa = r[f"{mode}_d_q_alone"]
    # merges
    i2 = ti[:, 1]
    merge = main & (i2 >= 0) & (qt[np.maximum(i2, 0)] > 5) & ~np.isin(i2, credited) & (c2 > 0.1 * cp)
    ratio_th = th_s[j] / th500[np.maximum(prim, 0)]
    out.update(n5=sel5.sum(), n_none=(sel5 & ~real).sum(), n_frag=frag.sum(), n_main=main.sum(),
               b10=(main & (delta > 0.1 * cp)).sum(), b20=(main & (delta > 0.2 * cp)).sum(),
               needs_alone=(main & (qa < 5)).sum(), needs_det=(main & ((cp + nz) / sd < 5)).sum(),
               merge=merge.sum(), wrong=(main & ((ratio_th > 2) | (ratio_th < 0.5))).sum(),
               wrong_clean=(main & (delta <= 0.1 * cp) & ((ratio_th > 2) | (ratio_th < 0.5))).sum(),
               n_clean=(main & (delta <= 0.1 * cp)).sum(),
               wrong_b10=(main & (delta > 0.1 * cp) & ((ratio_th > 2) | (ratio_th < 0.5))).sum(),
               mean_bias=np.sum((delta / cp)[main]), lowered10=(main & (delta < -0.1 * cp)).sum())
    # lost: forced halos with q_alone > 5 not credited
    fidx = r[f"{mode}_f_idx"]; fqa = r[f"{mode}_f_q_alone"]
    alone5 = fidx[fqa > 5]
    out.update(n_alone5=alone5.size, lost=(~np.isin(alone5, credited)).sum(),
               alone5_credited=np.isin(alone5, credited).sum())
    # forced photometry P(D)-like classes (finder's filter, true positions, nearest scale)
    fj = r[f"{mode}_f_j"]; fs = sig[fj]
    own = r[f"{mode}_f_own"] / fs
    fn = (r[f"{mode}_f_S_cmb"] + r[f"{mode}_f_S_noise"]) / fs
    fdl = (r[f"{mode}_f_S_tsz"] - r[f"{mode}_f_own"]) / fs
    xmx = r[f"{mode}_f_nb_val"][:, 0] / fs
    single = own + fn > 5
    dom = xmx < own
    blend = (own + fn + fdl > 5) & dom
    out.update(fp_single=single.sum(), fp_blend=blend.sum(), fp_b10=(blend & (fdl > 0.1 * own)).sum(),
               fp_b20=(blend & (fdl > 0.2 * own)).sum(), fp_boosted=(blend & ~single).sum(),
               fp_lost=(single & ~blend).sum(), fp_bias=np.sum((fdl / own)[blend]))
    # ---- attribution: detections where a second halo contributes a lot
    att = np.flatnonzero(main & (c2 >= 0.1 * cp))
    out["att_ratio"] = c2[att] / cp[att]
    out["att_q"] = q[att]
    A, B = prim[att], i2[att]
    out["att_qA"], out["att_qB"] = qt[A], qt[B]
    # step 1's dominance at the true positions: for the A, B of these detections and for every halo with q_true > 3
    hq3 = np.flatnonzero(qt > 3)
    idx = np.concatenate([A, B, hq3])
    rho, lth0, dlth, RB = RHO[mode]
    order = np.argsort(t["y"]).astype(np.int64)
    ys = t["y"][order].astype(float)
    sA = S.sigma_Y(mode, th500[idx])
    xm = np.zeros(idx.size); xa = np.zeros(idx.size, np.int64); xs = np.zeros(idx.size)
    true_pos_neighbours(idx, ys, order, t["y"].astype(float), t["x"].astype(float), Y, th500, sA, rho, lth0, dlth, RB,
                        F.LBOX, 60.0, xm, xa, xs)
    nA = A.size
    xq = xm[2 * nA:] / (Y[hq3] / sA[2 * nA:])
    fpos = np.searchsorted(fidx, hq3)
    out["dom_ratio"] = xq                                         # largest neighbour / own, step 1 filter, true position
    out["dom_cred"] = np.isin(hq3, credited)
    out["dom_alone5"] = fqa[np.minimum(fpos, fidx.size - 1)] > 5
    out["dom_qt"] = qt[hq3]
    if att.size:
        out["att_xA"], out["att_xB"] = xm[:nA], xm[nA:2 * nA]
        out["att_qtA"], out["att_qtB"] = Y[A] / sA[:nA], Y[B] / sA[nA:2 * nA]
        out["att_dA"] = np.hypot((t["y"][A] - iy[att] * F.PIX + 300) % 600 - 300, (t["x"][A] - ix[att] * F.PIX + 300) % 600 - 300)
        out["att_dB"] = np.hypot((t["y"][B] - iy[att] * F.PIX + 300) % 600 - 300, (t["x"][B] - ix[att] * F.PIX + 300) % 600 - 300)
    else:
        for k in ("att_xA", "att_xB", "att_qtA", "att_qtB", "att_dA", "att_dB"):
            out[k] = np.zeros(0)
    # ---- counts by redshift: credited (finder normalisation) and with sigma_Y re-normalisation
    zc = z[credited]
    out["Ndet"] = np.histogram(zc, ZB)[0].astype(float)
    r1 = out["r1"]
    rh = interp_lnth(r1, th500, th_s)
    sigfin_h = rh * S.sigma_Y(mode, th500)
    ri = interp_lnth(rinj, th500)
    qmin = 2.0 * ri                                                       # credit limit (c_p >= 2 sigma)
    pexp = p_sel(mode, Y / sigfin_h * ri, th500, qmin=qmin)
    out["Nexp"] = np.histogram(z, ZB, weights=pexp)[0]
    rib = interp_lnth(rinj_b, th500)                                      # the build's t03 calibration, for reference
    out["Nexp_b_nocut"] = np.histogram(z, ZB, weights=p_sel(mode, Y / sigfin_h * rib, th500, fits=SEL[mode + "_fit"]))[0]
    out["Nexp_b_cut"] = np.histogram(z, ZB, weights=p_sel(mode, Y / sigfin_h * rib, th500, fits=SEL[mode + "_fit"],
                                                          qmin=2.0 * rib))[0]
    out["Nalone"] = np.histogram(z[alone5], ZB)[0].astype(float)          # same filter and noise, no neighbours
    fqa_sY = fqa * out["r1"][r[f"{mode}_f_q_alone_j"]]                     # isolated search renormalised by sigma_Y
    out["Nalone_sY"] = np.histogram(z[fidx[fqa_sY > 5]], ZB)[0].astype(float)
    qprime = Y / sigfin_h * ri
    kk = qprime >= qmin
    out["sel_q"], out["sel_th"], out["sel_z"] = qprime[kk], th500[kk], z[kk]
    if SELA is not None:
        out["Nexp_tab"] = np.histogram(z[kk], ZB, weights=p_table_arb(mode, qprime[kk], th500[kk], qmin[kk]))[0]
    else:
        out["Nexp_tab"] = np.histogram(z[kk], ZB, weights=p_table(mode, qprime[kk], th500[kk]))[0]
    # sigma_Y normalisation: detections with q * r(theta_det) > 5 (credited, real), against P_sel(Y / sigma_Y)
    qsy = q * r1[j]
    dsy = np.flatnonzero((qsy > 5) & real)
    credY = np.unique(prim[dsy])
    out["Ndet_sY"] = np.histogram(z[credY], ZB)[0].astype(float)
    out["Nexp_sY"] = np.histogram(z, ZB, weights=p_sel(mode, Y / S.sigma_Y(mode, th500), th500, qmin=qmin))[0]
    # merge variants: distinct real primaries of the merged q > 5 catalogue
    det = dict(q=q, y=iy * F.PIX, x=ix * F.PIX)
    for mname in F.MERGES:
        k = F.apply_merge(det, mname, 5.0)
        out["n5all_" + mname] = k.size                                   # every entry, spurious ones included
        k = k[real[k]]
        out["Ndet_" + mname] = np.histogram(z[np.unique(prim[k])], ZB)[0].astype(float)
        out["n5_" + mname] = k.size
    # iteration 0 catalogue at q > 5 (no decomposition): number of detections
    out["n5_it0"] = (r[f"{mode}_it0_q"] > 5).sum()
    out["closure"] = (r[f"{mode}_d_S_tsz"] - r[f"{mode}_d_tab_sum"])[sel5] / sd[sel5]
    out["closure_rel"] = ((r[f"{mode}_d_S_tsz"] - r[f"{mode}_d_tab_sum"]) / r[f"{mode}_d_S_tot"])[sel5]
    out["comp_closure"] = np.abs(r[f"{mode}_d_S_tot"] - r[f"{mode}_d_S_tsz"] - r[f"{mode}_d_S_cmb"] - r[f"{mode}_d_S_noise"]).max() / np.abs(r[f"{mode}_d_S_tot"]).min()
    out["timing"] = r[f"{mode}_timing"]; out["t_total"] = float(r["t_total"])
    s5 = np.flatnonzero(sel5)
    out["det_arrays"] = dict(q=q[s5], none=~real[s5], frag=frag[s5], main=main[s5], b10=(main & (delta > 0.1 * cp))[s5],
                             b20=(main & (delta > 0.2 * cp))[s5], needs=(main & (qa < 5))[s5], merge=merge[s5],
                             wrong=(main & ((ratio_th > 2) | (ratio_th < 0.5)))[s5], z=z[np.maximum(prim, 0)][s5],
                             pid=np.full(s5.size, pid))
    return out


def boot_ratio(num, den, nboot=NBOOT):
    """num, den: (ntile, nbin). Ratio of sums - 1 per bin and total, with tile-bootstrap errors."""
    tot = num.sum(0) / den.sum(0) - 1
    T = num.sum() / den.sum() - 1
    b = rng.integers(0, num.shape[0], (nboot, num.shape[0]))
    nb_ = num[b].sum(1); db = den[b].sum(1)
    eb = (nb_ / db - 1).std(0)
    eT = (nb_.sum(1) / db.sum(1) - 1).std()
    return tot, eb, T, eT


def frac(a, b):
    return f"{100 * a / b:.2f} +- {100 * np.sqrt(a * (1 - a / b)) / b:.2f}%" if b else "n/a"


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(TD) if f.startswith("tile_"))
    pids = [int(f[5:8]) for f in files][:NT]
    TAG = SUB + ("" if len(pids) == len(files) else f"_n{len(pids)}")
    print(f"t05: {len(pids)} realistic tiles ({len(pids) * 100} deg^2 per noise level) from {TD}")
    print(f"selection calibration for N_exp: {CAL}; P_det = 0 below the credit limit q_inj = 2 r_inj")
    s2 = dict(np.load(f"{S.ROOT}/data_real/step2/inputs.npz"))
    for mode in S.MODES:
        rinj_b = r_inj(mode)
        rinj = SELA[mode + "_rinj"] if SELA is not None else rinj_b
        T = [tile_stats(p, mode, rinj, rinj_b) for p in pids]
        A = {k: np.array([t_[k] for t_ in T]) for k in T[0]
             if not k.startswith(("att_", "dom_", "sel_")) and k not in ("closure", "closure_rel", "det_arrays")}
        DA = {k: np.concatenate([t_["det_arrays"][k] for t_ in T]) for k in T[0]["det_arrays"]}
        np.savez(os.path.join(S.OUT, f"t05_{TAG}_{mode}.npz"), r0=A["r0"], r1=A["r1"], r_cn=A["r_cn"], r_lit=A["r_lit"],
                 r_anatsz=A["r_anatsz"], thetas=np.load(os.path.join(TD, files[0]))["thetas"],
                 **{"det_" + k: v for k, v in DA.items()})
        print(f"\n==================== {mode}: {len(pids)} tiles")
        # ---------------- 4a
        th = F.THETAS15
        ri = np.interp(np.log(th), np.log(s2[mode + "_th"]), s2[mode + "_r_i"])
        rii = np.interp(np.log(th), np.log(s2[mode + "_th"]), s2[mode + "_r_ii"])
        k3 = np.load(f"{S.ROOT}/data_real/step2/kernel_{mode}_effiii.npz")
        riii = np.interp(np.log(th), np.log(k3["th"]), k3["r"])
        print("(4a) noise normalisation r(theta) = sigma_finder / sigma_Y: mean [tile-to-tile std]")
        print("  theta [']                 " + " ".join(f"{v:7.2f}" for v in th))
        for lab, key in (("finder, before iteration", "r0"), ("finder, after iteration", "r1"),
                         ("CMB + noise only, estimated", "r_cn"), ("analytic CMB+noise + tile tSZ power", "r_anatsz"),
                         ("literal binary f_sky iteration", "r_lit")):
            print(f"  {lab:36s}" + " ".join(f"{v:7.4f}" for v in A[key].mean(0)))
            print(f"  {'   std':36s}" + " ".join(f"{v:7.4f}" for v in A[key].std(0)))
        print(f"  {'mask effect on CMB + noise alone':36s}" + " ".join(f"{v:7.4f}" for v in A["mask_cn"].mean(0)))
        print(f"  {'step 2 (i) sub-threshold confusion':36s}" + " ".join(f"{v:7.4f}" for v in ri))
        print(f"  {'step 2 (ii) robust width':36s}" + " ".join(f"{v:7.4f}" for v in rii))
        print(f"  {'step 2 (iii) plain std':36s}" + " ".join(f"{v:7.4f}" for v in riii))
        r1m = A["r1"].mean(0)
        for lab, rv in (("(i)", ri), ("(ii)", rii), ("(iii)", riii)):
            print(f"  (r_finder_after - 1) / (r_{lab} - 1): " + " ".join(f"{v:5.2f}" for v in (r1m - 1) / (rv - 1)))
        print(f"  (r_finder_before - 1) / (r_(iii) - 1): " + " ".join(f"{v:5.2f}" for v in (A['r0'].mean(0) - 1) / (riii - 1)))
        print(f"  masked fraction at the iteration: mean {100 * (1 - A['fsky1'].mean()):.1f}% [range {100 * (1 - A['fsky1'].max()):.1f}-{100 * (1 - A['fsky1'].min()):.1f}%]")
        G = F.grid()
        Ccn = F.analytic_bin_cov(mode, G)
        a_m = G.a_modes()
        widths = []
        for p in pids[:40]:
            C1 = np.load(os.path.join(TD, f"tile_{p:03d}.npz"))[f"{mode}_it1_C"]
            Ci = np.zeros_like(C1); Ci[G.bins_used] = np.linalg.inv(C1[G.bins_used])
            M = np.zeros_like(C1); M[G.bins_used] = np.einsum("bij,bjk,bkl->bil", Ci[G.bins_used], Ccn[G.bins_used], Ci[G.bins_used])
            aCa = np.einsum("mi,mij,mj->m", a_m, Ci[G.m_bin], a_m); aMa = np.einsum("mi,mij,mj->m", a_m, M[G.m_bin], a_m)
            w = []
            for th_ in F.THETAS15:
                tau2 = F.gt_lin(G.m_L * th_ * F.AR) ** 2
                w.append(np.sqrt(np.sum(G.m_w * tau2 * aMa) / np.sum(G.m_w * tau2 * aCa)))
            widths.append(w)
        widths = np.array(widths)
        print(f"  std of the CMB + noise part of the finder's S/N maps (after iteration; 40 tiles, analytic CMB + noise "
              f"through each tile's filter): " + " ".join(f"{v:.4f}" for v in widths.mean(0)))
        print(f"     compare 1 / r_after: " + " ".join(f"{v:.4f}" for v in 1 / A['r1'][:40].mean(0)))
        # ---------------- 4b
        n5 = A["n5"].sum(); nm = A["n_main"].sum()
        print(f"(4b) final detections q > 5: {n5} ({n5 / len(pids):.1f} per tile); real and not fragments ('main'): {nm}")
        print(f"  no real cluster (c_p < 2 sigma): {frac(A['n_none'].sum(), n5)} of all q > 5")
        print(f"  fragments (second detection of a halo): {frac(A['n_frag'].sum(), n5)} of all q > 5")
        for lab, key in (("Y boosted > 10% by neighbours", "b10"), ("Y boosted > 20%", "b20"),
                         ("detected only thanks to neighbours (isolated search)", "needs_alone"),
                         ("  same, at the detection's own position and size", "needs_det"),
                         ("Y lowered by > 10%", "lowered10"),
                         ("merge (second halo q_true > 5, > 10% of c_p, not detected itself)", "merge"),
                         ("wrong size (theta_det / theta500 outside 1/2-2)", "wrong")):
            print(f"  {lab:70s}: {frac(A[key].sum(), nm)} of main")
        print(f"  wrong size among clean main detections {frac(A['wrong_clean'].sum(), A['n_clean'].sum())}, among Y boosted > 10% "
              f"{frac(A['wrong_b10'].sum(), A['b10'].sum())}")
        print(f"  mean neighbour signal / own signal of main detections: {100 * A['mean_bias'].sum() / nm:+.2f}%")
        print(f"  lost: halos with q_alone > 5 not credited with a q > 5 detection: {frac(A['lost'].sum(), A['n_alone5'].sum())} "
              f"(of {A['n_alone5'].sum()})")
        nb_ = A["fp_blend"].sum(); ns = A["fp_single"].sum()
        print(f"  forced photometry at the true positions (finder's filter, nearest scale; dominance fdom = 1; halos q_true > 2): "
              f"blended detections {nb_}, single {ns} (change {100 * (nb_ / ns - 1):+.2f}%)")
        print(f"     Y boosted > 10% {frac(A['fp_b10'].sum(), nb_)}; > 20% {frac(A['fp_b20'].sum(), nb_)}; boosted over threshold "
              f"{frac(A['fp_boosted'].sum(), nb_)}; lost {frac(A['fp_lost'].sum(), ns)} (of single); mean bias {100 * A['fp_bias'].sum() / nb_:+.2f}%")
        cl = np.concatenate([t_["closure"] for t_ in T]); clr = np.concatenate([t_["closure_rel"] for t_ in T])
        print(f"  decomposition closure (q > 5): map value = tSZ + CMB + noise to {A['comp_closure'].max():.1e} (worst tile, relative); "
              f"tSZ - sum of all halos' table contributions: median |.| {np.median(np.abs(cl)):.1e} sigma ({np.median(np.abs(clr)):.1e} of the map value), "
              f"95% {np.percentile(np.abs(cl), 95):.1e} sigma, max {np.abs(cl).max():.1e} sigma")
        # ---------------- 4c
        rat = np.concatenate([t_["att_ratio"] for t_ in T])
        xA = np.concatenate([t_["att_xA"] for t_ in T]); xB = np.concatenate([t_["att_xB"] for t_ in T])
        qA = np.concatenate([t_["att_qtA"] for t_ in T]); qB = np.concatenate([t_["att_qtB"] for t_ in T])
        dA = np.concatenate([t_["att_dA"] for t_ in T]); dB = np.concatenate([t_["att_dB"] for t_ in T])
        print(f"(4c) attribution: main q > 5 detections whose second contributor B gives >= 10% of the primary A's signal: "
              f"{rat.size} ({100 * rat.size / nm:.1f}% of main)")
        for lo, hi in ((0.1, 0.25), (0.25, 0.5), (0.5, 1.01)):
            m = (rat >= lo) & (rat < hi)
            if not m.any():
                continue
            line = [f"c_B/c_A {lo}-{min(hi, 1)}: N = {m.sum()}; A has the larger true q (step 1 sigma_Y) in {100 * np.mean(qA[m] > qB[m]):.0f}%; "
                    f"detection closer to A than B in {100 * np.mean(dA[m] < dB[m]):.0f}%"]
            for fd in (0.5, 1.0, 2.0):
                ca = xA[m] < fd * qA[m]; cb = xB[m] < fd * qB[m]
                line.append(f"fdom {fd}: A only {100 * np.mean(ca & ~cb):.0f}%, B only {100 * np.mean(~ca & cb):.0f}%, "
                            f"both {100 * np.mean(ca & cb):.0f}%, neither {100 * np.mean(~ca & ~cb):.0f}%")
            print("  " + "\n     ".join(line))
        dr = np.concatenate([t_["dom_ratio"] for t_ in T]); dc = np.concatenate([t_["dom_cred"] for t_ in T])
        da = np.concatenate([t_["dom_alone5"] for t_ in T])
        print("  effective dominance rule: halos q_true > 3 that the isolated search would detect (q_alone > 5), binned in "
              "x_max / q_true at their true position (largest single neighbour, step 1's filter); fraction credited by the finder:")
        for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 4.0), (4.0, 1e9)):
            m = da & (dr >= lo) & (dr < hi)
            if m.sum():
                print(f"     x_max/q in [{lo}, {hi if hi < 1e8 else 'inf'}): N = {m.sum():6d}, credited {100 * dc[m].mean():6.2f}%")
        m = ~da & dc
        print(f"     credited although q_alone < 5 (boosted over the threshold): {m.sum()}; their x_max/q median {np.median(dr[m]) if m.any() else np.nan:.2f}")
        # ---------------- 4d
        print("(4d) count change at q > 5: credited halos vs the finder's own no-neighbour selection; tile bootstrap")
        print("  z bins: " + ", ".join(f"{a}-{b}" for a, b in zip(ZB[:-1], ZB[1:])))
        for lab, nk, ek in (("finder normalisation (q = Y/sigma_finder)", "Ndet", "Nexp"),
                            ("sigma_Y normalisation (q_obs r > 5; q = Y/sigma_Y)", "Ndet_sY", "Nexp_sY")):
            tot, eb, Tt, eT = boot_ratio(A[nk], A[ek])
            print(f"  {lab}: N_det {A[nk].sum():.0f}, N_exp {A[ek].sum():.1f}; Delta/N = {100 * Tt:+.2f} +- {100 * eT:.2f}%")
            print("     per z bin: " + ", ".join(f"{100 * a:+.2f}+-{100 * e:.2f}%" for a, e in zip(tot, eb)))
        for lab, ek in (("reference: the build's t03 fit as published (no credit limit)", "Nexp_b_nocut"),
                        ("reference: the build's t03 fit with the credit limit", "Nexp_b_cut")):
            tot, eb, Tt, eT = boot_ratio(A["Ndet"], A[ek])
            print(f"  {lab}: N_exp {A[ek].sum():.1f}; Delta/N = {100 * Tt:+.2f} +- {100 * eT:.2f}%; per z bin "
                  + ", ".join(f"{100 * a:+.2f}%" for a in tot))
        tot, eb, Tt, eT = boot_ratio(A["Ndet"], A["Nalone"])
        print(f"  split 1, neighbours at fixed filter and noise: credited vs isolated search (q_alone > 5): N_alone {A['Nalone'].sum():.0f}; "
              f"{100 * Tt:+.2f} +- {100 * eT:.2f}%; per z bin " + ", ".join(f"{100 * a:+.2f}+-{100 * e:.2f}%" for a, e in zip(tot, eb)))
        tot, eb, Tt, eT = boot_ratio(A["Nalone"], A["Nexp"])
        print(f"  split 2, normalisation convention: isolated search vs unit-noise selection: {100 * Tt:+.2f} +- {100 * eT:.2f}%; "
              f"per z bin " + ", ".join(f"{100 * a:+.2f}+-{100 * e:.2f}%" for a, e in zip(tot, eb)))
        tot, eb, Tt, eT = boot_ratio(A["Ndet_sY"], A["Nalone_sY"])
        print(f"  sigma_Y normalisation, split 1 (neighbours at fixed filter and noise): {100 * Tt:+.2f} +- {100 * eT:.2f}%; per z bin "
              + ", ".join(f"{100 * a:+.2f}+-{100 * e:.2f}%" for a, e in zip(tot, eb)))
        tot, eb, Tt, eT = boot_ratio(A["Nalone_sY"], A["Nexp_sY"])
        print(f"  sigma_Y normalisation, split 2 (isolated search vs unit-noise selection): {100 * Tt:+.2f} +- {100 * eT:.2f}%; per z bin "
              + ", ".join(f"{100 * a:+.2f}+-{100 * e:.2f}%" for a, e in zip(tot, eb)))
        # selection-model uncertainty: refit P_det on bootstrap resamples of the injection tiles (per theta_inj)
        sq = np.concatenate([t_["sel_q"] for t_ in T]); sth = np.concatenate([t_["sel_th"] for t_ in T])
        sz = np.concatenate([t_["sel_z"] for t_ in T])
        Ndt = A["Ndet"].sum(0)
        base = np.histogram(sz, ZB, weights=p_sel(mode, sq, sth))[0]
        NB_SEL = 40
        bres = []
        fit0 = SELA[mode + "_fit"] if SELA is not None else SEL[mode + "_fit"]
        for b_ in range(NB_SEL):
            fits = []
            for it_, thv in enumerate(TH_INJ):
                if SELA is not None:
                    qi, dt, tl = SELA[f"{mode}_{thv}_q"], SELA[f"{mode}_{thv}_det"], SELA[f"{mode}_{thv}_tile"]
                else:
                    qi, dt, tl = SEL[f"{mode}_{thv}_q"], SEL[f"{mode}_{thv}_qobs"] > 5, SEL[f"{mode}_{thv}_tile"]
                ut, inv = np.unique(tl, return_inverse=True)
                pick = rng.choice(ut.size, ut.size)
                wgt = np.bincount(pick, minlength=ut.size)[inv]               # tile multiplicities
                ii = np.repeat(np.arange(qi.size), wgt)
                fits.append(fit_model(qi[ii], dt[ii].astype(float), x0=fit0[it_]))
            bres.append(np.histogram(sz, ZB, weights=p_sel(mode, sq, sth, np.array(fits)))[0])
        bres = np.array(bres)
        esel = (Ndt[None, :] / bres - 1).std(0)
        esel_T = (Ndt.sum() / bres.sum(1) - 1).std()
        print(f"  selection-model uncertainty ({NB_SEL} refits on bootstrap resamples of the injection tiles): total "
              f"{100 * esel_T:.2f}%; per z bin " + ", ".join(f"{100 * e:.2f}%" for e in esel)
              + f"  (check: refit mean vs nominal N_exp {100 * (bres.sum(1).mean() / base.sum() - 1):+.2f}%)")
        tot, eb, Tt, eT = boot_ratio(A["Ndet"], A["Nexp"])
        print(f"  finder normalisation, tile bootstrap (+) selection model: total {100 * Tt:+.2f} +- {100 * np.hypot(eT, esel_T):.2f}%; per z bin "
              + ", ".join(f"{100 * a:+.2f}+-{100 * np.hypot(e, f_):.2f}%" for a, e, f_ in zip(tot, eb, esel)))
        tot, eb, Tt, eT = boot_ratio(A["Ndet"], A["Nexp_tab"])
        print(f"  same with the binned completeness table instead of the fit: N_exp {A['Nexp_tab'].sum():.1f}; Delta/N = "
              f"{100 * Tt:+.2f} +- {100 * eT:.2f}%; per z bin " + ", ".join(f"{100 * a:+.2f}%" for a in tot))
        for mname in F.MERGES:
            tot, eb, Tt, eT = boot_ratio(A["Ndet_" + mname], A["Nexp"])
            print(f"  merge {mname:9s}: N_det {A['Ndet_' + mname].sum():.0f}; Delta/N = {100 * Tt:+.2f} +- {100 * eT:.2f}%  "
                  f"(q > 5 catalogue entries {A['n5_' + mname].sum()})")
        print(f"  q > 5 detections before the iteration: {A['n5_it0'].sum()} vs after {A['n5'].sum()}")
        # a survey counts every entry of its catalogue and subtracts the false rate measured on noise-only tiles (t02)
        nd_ = os.path.join(S.OUT, "noise_only")
        if "szifi" in SUB:
            NOPRINT = True
            print("  (counting every catalogue entry minus the noise-only false rate: skipped; t02's false rates are the whitened finder's)")
        else:
            NOPRINT = False
        if not NOPRINT:
            NO = [dict(np.load(os.path.join(nd_, f))) for f in sorted(os.listdir(nd_))]
            print("  counting every q > 5 catalogue entry (spurious ones included) minus the noise-only false rate (t02, "
                  f"{len(NO)} tiles), against N_exp:")
            for mname in F.MERGES:
                nf = []
                for r_ in NO:
                    kk = f"{mode}_whitened_it{r_[f'{mode}_whitened_n_it'] - 1}"
                    dd = dict(q=r_[kk + "_q"], y=r_[kk + "_iy"] * F.PIX, x=r_[kk + "_ix"] * F.PIX)
                    nf.append(F.apply_merge(dd, mname, 5.0).size)
                nfal = np.mean(nf) * len(pids)
                nall = A["n5all_" + mname].sum()
                ne = A["Nexp"].sum()
                b = rng.integers(0, len(pids), (NBOOT, len(pids)))
                eb = ((A["n5all_" + mname][b].sum(1) - nfal) / A["Nexp"].sum(1)[b].sum(1) - 1).std()
                print(f"     {mname:9s}: entries {nall}, expected false {nfal:.1f}, (entries - false) / N_exp - 1 = "
                      f"{100 * ((nall - nfal) / ne - 1):+.2f} +- {100 * eb:.2f}% (credited only: {100 * (A['Ndet_' + mname].sum() / ne - 1):+.2f}%)")
        for v in ("f1", "effi", "effii", "effiii"):
            d = np.load(f"{S.ROOT}/data_real/step2/truth_{mode}_{v}.npz")
            print(f"  step 2 ({v}): Delta/N = {100 * d['D_q5'].sum() / d['Nn_q5'].sum():+.2f}%; per z bin "
                  + ", ".join(f"{100 * a:+.2f}%" for a in d["tilt"]))
        tm = A["timing"]
        print(f"  compute: per tile and noise level find {tm[:, 0].mean():.1f} s, diagnostics {tm[:, 1].mean():.1f} s, "
              f"decomposition {tm[:, 2].mean():.1f} s; per tile (both levels, painting incl.) {A['t_total'].mean():.1f} s")
    print(f"\ndone [{time.time() - t0:.0f} s]")
