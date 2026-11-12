"""x08 (indep lens), claim S1: 12 or 20 instead of 15 filter scales on the same own fresh skies (x04 / x04b outputs).
Compared: q > 5 detections, credited halos, class fractions, r after the iteration at 0.5, 1, 2, 4, 8, 15', and the count
change against the own selection.  Usage: python x08_scales.py NT
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import ndtr
import vlib3 as V

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
log = V.Tee(os.path.join(V.HERE, "x08_scales.log"))
SELM = dict(np.load(os.path.join(V.OUT, "selection_mine.npz")))
TH_INJ = SELM["theta_inj"]
TH6 = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 15.0])


def p_sel(fits, q, th):
    P = np.array([ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)) for b, mu, ls in fits])
    x = np.clip(np.log(th), np.log(TH_INJ[0]), np.log(TH_INJ[-1]))
    i = np.clip(np.searchsorted(np.log(TH_INJ), x) - 1, 0, len(TH_INJ) - 2)
    w = (x - np.log(TH_INJ[i])) / np.log(TH_INJ[i + 1] / TH_INJ[i])
    n = np.arange(q.size)
    return (1 - w) * P[i, n] + w * P[i + 1, n]


def stats(path, mode, ths, h):
    r = dict(np.load(path)); p = mode + "_"
    Y, th = h["Y500"], h["theta500"]
    qt = Y / V.sigma_Y(mode, th)
    sig = r[p + "sig1"]; q, j = r[p + "q"], r[p + "j"]
    r1 = sig / V.sigma_Y(mode, ths)
    d5 = r[p + "d5"]; ti, ex = r[p + "top_idx"], r[p + "top_ex"]
    prim, cp = ti[:, 0], ex[:, 0]
    c2 = np.where(np.isfinite(ex[:, 1]), ex[:, 1], 0.0); i2 = ti[:, 1]
    sd = sig[j[d5]]
    delta = r[p + "S_tsz"][d5] - cp
    real = cp / sd >= 2
    seen = set(); main = np.zeros(d5.size, bool)
    for a in np.argsort(-q[d5]):
        if real[a] and prim[a] not in seen:
            main[a] = True; seen.add(int(prim[a]))
    cred = np.array(sorted(seen), np.int64)
    fidx, fqa = r[p + "f_idx"], r[p + "f_q_alone"]
    qa = fqa[np.searchsorted(fidx, prim)]
    rth = ths[j[d5]] / th[prim]
    alone5 = fidx[fqa > 5]
    sigfin = np.interp(np.log(th), np.log(ths), r1) * V.sigma_Y(mode, th)
    qp = Y / sigfin * np.interp(np.log(th), np.log(V.THETAS), SELM[mode + "_rinj"])
    return dict(n5=d5.size, cred=cred.size, main=int(main.sum()), none=int((~real).sum()), frag=int((real & ~main).sum()),
                b10=int((main & (delta > 0.1 * cp)).sum()), b20=int((main & (delta > 0.2 * cp)).sum()),
                needs=int((main & (qa < 5)).sum()),
                merge=int((main & (i2 >= 0) & (qt[np.maximum(i2, 0)] > 5) & ~np.isin(i2, cred) & (c2 > 0.1 * cp)).sum()),
                wrong=int((main & ((rth > 2) | (rth < 0.5))).sum()), lost=int((~np.isin(alone5, cred)).sum()), nal=alone5.size,
                r6=np.interp(np.log(TH6), np.log(ths), r1), nexp=float(p_sel(SELM[mode + "_fit"], qp, th).sum()))


log(f"x08: S1 check, own fresh tiles 0..{NT - 1}, 12 / 15 / 20 scales")
for mode in V.MODES:
    res = {}
    for nsc in (12, 15, 20):
        ths = np.geomspace(0.5, 15.0, nsc)
        d = os.path.join(V.OUT, "fresh" if nsc == 15 else f"fresh_s{nsc}")
        S = []
        for t in range(NT):
            h = V.draw_tile_halos(7_000_000 + t)
            S.append(stats(os.path.join(d, f"fresh_{t:03d}.npz"), mode, ths, h))
        res[nsc] = {k: (np.sum([s[k] for s in S], 0) if k != "r6" else np.mean([s[k] for s in S], 0)) for k in S[0]}
    log(f"\n==================== {mode}")
    for nsc in (12, 15, 20):
        a = res[nsc]
        log(f"  {nsc} scales: q > 5 detections {a['n5']}, credited halos {a['cred']}, N_exp {a['nexp']:.1f}, Delta/N {100 * (a['cred'] / a['nexp'] - 1):+.2f}%")
        log(f"     of main: Y > 10% {100 * a['b10'] / a['main']:.2f}%, > 20% {100 * a['b20'] / a['main']:.2f}%, only thanks to neighbours {100 * a['needs'] / a['main']:.2f}%, "
            f"merges {100 * a['merge'] / a['main']:.2f}%, wrong size {100 * a['wrong'] / a['main']:.2f}%; of all: no real cluster {100 * a['none'] / a['n5']:.2f}%, "
            f"fragments {100 * a['frag'] / a['n5']:.2f}%; lost {100 * a['lost'] / a['nal']:.2f}%")
        log(f"     r after iteration at 0.5, 1, 2, 4, 8, 15': " + " ".join(f"{v:.4f}" for v in a["r6"]))
    for nsc in (12, 20):
        a, b = res[nsc], res[15]
        dfr = max(abs(100 * a[k] / a[n] - 100 * b[k] / b[n]) for k, n in (("b10", "main"), ("b20", "main"), ("needs", "main"), ("merge", "main"),
                                                                       ("wrong", "main"), ("none", "n5"), ("frag", "n5"), ("lost", "nal")))
        log(f"  {nsc} vs 15 scales: credited halos {100 * (a['cred'] / b['cred'] - 1):+.2f}%, q > 5 detections {100 * (a['n5'] / b['n5'] - 1):+.2f}%, "
            f"max class-fraction change {dfr:.2f} pp, max |r change| {np.abs(a['r6'] - b['r6']).max():.1e}, Delta/N change "
            f"{100 * (a['cred'] / a['nexp'] - b['cred'] / b['nexp']):+.2f} pp")
