"""x11 (indep lens): the count change at q > 5 against a criterion-matched unit-noise calibration (x10's virtual injections),
on the own fresh skies (x04) and on the build's skies (its stored arrays), and the same against the injection-based
selections (own x03, own second set x03b, the build's).

  P_alone(q, theta): fraction of virtual clusters on noise-only tiles with q_alone > 5 (isolated search), binned in q
  (width 0.25, 1.5-9) and ln theta (14 bins, 0.25'-12'), bilinear between bin centres (clamped in theta; 0 below
  q = 1.5, 1 above 9).  r_noise = sigma_final / sigma_Y on the noise-only tiles.
  N_exp_alone = sum over all halos of P_alone(q', theta500), q' = Y500 / sigma_finder(theta500) x r_noise(theta500)
  split 2' = N_alone / N_exp_alone - 1, N_alone = halos with q_alone > 5 on the realistic tile (isolated search);
  N1' = N_credited / N_exp_alone - 1.
Errors: tile bootstrap (2000) of the realistic tiles; calibration error from 40 bootstrap resamples of the noise tiles.
Usage: python x11_split2.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
import vlib3 as V

log = V.Tee(os.path.join(V.HERE, "x11_split2.log"))
t00 = time.time()
TH = V.THETAS
ZB = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
rng = np.random.default_rng(2024)
VD = os.path.join(V.OUT, "virt")
VF = [dict(np.load(os.path.join(VD, f))) for f in sorted(os.listdir(VD)) if f.endswith(".npz")]
QE = np.arange(1.5, 9.0 + 1e-9, 0.25)
LTE = np.linspace(np.log(0.25), np.log(12.0), 15)


def table(files, mode):
    q = np.concatenate([f["q_inj"] for f in files]).astype(float)
    lt = np.log(np.concatenate([f["theta"] for f in files]).astype(float))
    d = np.concatenate([f[mode + "_q_alone"] for f in files]) > 5
    iq = np.clip(np.digitize(q, QE) - 1, 0, QE.size - 2); it = np.clip(np.digitize(lt, LTE) - 1, 0, LTE.size - 2)
    n = np.zeros((LTE.size - 1, QE.size - 1)); k = np.zeros_like(n)
    np.add.at(n, (it, iq), 1); np.add.at(k, (it, iq), d)
    return k / np.maximum(n, 1), n


def p_alone(P, q, th):
    qc = 0.5 * (QE[1:] + QE[:-1]); tc = 0.5 * (LTE[1:] + LTE[:-1])
    x = np.clip(np.log(th), tc[0], tc[-1])
    i = np.clip(np.searchsorted(tc, x) - 1, 0, tc.size - 2)
    w = (x - tc[i]) / (tc[i + 1] - tc[i])
    out = np.zeros(q.size)
    for a in range(tc.size):
        m = (i == a) | (i + 1 == a)
        if not m.any():
            continue
        pa = np.interp(q[m], np.r_[1.5, qc, 9.0], np.r_[0.0, P[a], 1.0], left=0.0, right=1.0)
        out[m] += np.where(i[m] == a, 1 - w[m], w[m]) * pa
    return out


def p_fit(fits, thinj, q, th):
    P = np.array([ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)) for b, mu, ls in fits])
    x = np.clip(np.log(th), np.log(thinj[0]), np.log(thinj[-1]))
    i = np.clip(np.searchsorted(np.log(thinj), x) - 1, 0, len(thinj) - 2)
    w = (x - np.log(thinj[i])) / np.log(thinj[i + 1] / thinj[i])
    n = np.arange(q.size)
    return (1 - w) * P[i, n] + w * P[i + 1, n]


def fit_sel(q, det):
    def nll(p):
        b, mu, ls = p
        pr = np.clip(ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)), 1e-12, 1 - 1e-12)
        return -np.sum(det * np.log(pr) + (1 - det) * np.log(1 - pr))
    return minimize(nll, [0.0, 0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9, maxiter=8000)).x


def inj_selection(d, mode):
    """Fit per size from an own injection folder (x03 / x03b outputs) and its r_inj."""
    rows = {}; sig = []
    for f in sorted(os.listdir(d)):
        if not f.startswith(f"inj_{mode}_"):
            continue
        r = dict(np.load(os.path.join(d, f)))
        th = float(r["theta_inj"])
        rows.setdefault(th, ([], []))
        rows[th][0].append(r["q_inj"]); rows[th][1].append(r["qbest"])
        sig.append(r["sig"] / V.sigma_Y(mode, TH))
    thinj = np.array(sorted(rows))
    fits = np.array([fit_sel(np.concatenate(rows[t][0]), (np.concatenate(rows[t][1]) > 5).astype(float)) for t in thinj])
    return fits, thinj, np.mean(sig, 0), sum(len(rows[t][0]) for t in thinj)


def credited_fresh(r, p, qt):
    q, j = r[p + "q"], r[p + "j"]
    d5 = r[p + "d5"]; ti, ex = r[p + "top_idx"], r[p + "top_ex"]
    sd = r[p + "sig1"][j[d5]]
    real = ex[:, 0] / sd >= 2
    seen = set()
    for a in np.argsort(-q[d5]):
        if real[a] and ti[a, 0] not in seen:
            seen.add(int(ti[a, 0]))
    return np.array(sorted(seen), np.int64)


def credited_build(b, p):
    q, j = b[p + "it1_q"], b[p + "it1_j"]
    d5 = np.flatnonzero(q > 5)
    ti, ex = b[p + "d_top_idx"][d5], b[p + "d_top_ex"][d5]
    sd = b[p + "it1_sig"][j[d5]]
    real = ex[:, 0] / sd >= 2
    seen = set()
    for a in np.argsort(-q[d5]):
        if real[a] and ti[a, 0] not in seen:
            seen.add(int(ti[a, 0]))
    return np.array(sorted(seen), np.int64)


def boot(num, den, nb=2000):
    T = num.sum() / den.sum() - 1
    b = rng.integers(0, num.shape[0], (nb, num.shape[0]))
    return T, (num[b].sum(1).sum(1) / den[b].sum(1).sum(1) - 1).std(), num.sum(0) / den.sum(0) - 1


log(f"x11: count change vs a criterion-matched unit-noise calibration: {len(VF)} noise tiles x {VF[0]['q_inj'].size} virtual clusters")
SB = dict(np.load(os.path.join(V.BUILD_DATA, "selection.npz")))
for mode in V.MODES:
    P, n = table(VF, mode)
    rnoise = np.mean([f[mode + "_r_noise"] for f in VF], 0)
    log(f"\n==================== {mode}: r_noise {rnoise.min():.4f}-{rnoise.max():.4f}; P_alone at q = 4.5 / 5 / 5.5 / 6 for theta bins centred at "
        + ", ".join(f"{np.exp(0.5 * (LTE[a] + LTE[a + 1])):.2f}'" for a in (1, 4, 7, 10, 13)))
    qc = 0.5 * (QE[1:] + QE[:-1])
    for a in (1, 4, 7, 10, 13):
        log(f"   theta {np.exp(0.5 * (LTE[a] + LTE[a + 1])):5.2f}': " + " / ".join(f"{np.interp(qq, qc, P[a]):.3f}" for qq in (4.5, 5.0, 5.5, 6.0))
            + f"   (N per q bin ~{int(n[a].mean())})")
    # calibration bootstrap tables
    Pb = []
    for _ in range(40):
        pick = rng.integers(0, len(VF), len(VF))
        Pb.append(table([VF[i] for i in pick], mode)[0])
    sels = {}
    for lab, d in (("own injections (x03)", os.path.join(V.OUT, "inj")), ("own second injection set (x03b)", os.path.join(V.OUT, "inj2"))):
        if os.path.isdir(d) and len([f for f in os.listdir(d) if f.startswith(f"inj_{mode}_")]) >= 200:
            sels[lab] = inj_selection(d, mode)
    dB = os.path.join(V.BUILD_DATA, "injections")
    rinjB = np.mean([np.load(os.path.join(dB, f))["sig"] for f in sorted(os.listdir(dB)) if f.startswith(f"inj_{mode}_")], 0) / V.sigma_Y(mode, TH)
    sels["build's selection fit"] = (SB[mode + "_fit"], SB["theta_inj"], rinjB, None)
    if len(sels) == 3:
        f1, t1, r1_, n1 = sels["own injections (x03)"]; f2, t2, r2_, n2 = sels["own second injection set (x03b)"]
        # pooled own fit: refit on both sets together
        rows = {}
        for d in (os.path.join(V.OUT, "inj"), os.path.join(V.OUT, "inj2")):
            for f in sorted(os.listdir(d)):
                if f.startswith(f"inj_{mode}_"):
                    r = dict(np.load(os.path.join(d, f)))
                    rows.setdefault(float(r["theta_inj"]), ([], []))
                    rows[float(r["theta_inj"])][0].append(r["q_inj"]); rows[float(r["theta_inj"])][1].append(r["qbest"])
        thp = np.array(sorted(rows))
        fp = np.array([fit_sel(np.concatenate(rows[t][0]), (np.concatenate(rows[t][1]) > 5).astype(float)) for t in thp])
        sels["own injections, both sets pooled"] = (fp, thp, 0.5 * (r1_ + r2_), None)
    for sky in ("own fresh skies (x04)", "build's skies (stored arrays)"):
        Nd, Na, Ne, Nea, Ne_s = [], [], [], [], {k: [] for k in sels}
        Ne_b = [[] for _ in Pb]
        if sky.startswith("own"):
            files = sorted(f for f in os.listdir(os.path.join(V.OUT, "fresh")) if f.startswith("fresh_"))
            units = [("own", f) for f in files]
        else:
            units = [("build", pid) for pid in range(324)]
        for kind, u in units:
            p = mode + "_"
            if kind == "own":
                r = dict(np.load(os.path.join(V.OUT, "fresh", u)))
                h = V.draw_tile_halos(7_000_000 + int(r["tile"]))
                Y, th, z = h["Y500"], h["theta500"], h["z"]
                qt = Y / V.sigma_Y(mode, th)
                cred = credited_fresh(r, p, qt)
                fidx, fqa = r[p + "f_idx"], r[p + "f_q_alone"]
                sig1 = r[p + "sig1"]
            else:
                r = V.build_tile(u); h = V.build_truth(u)
                Y, th, z = h["Y500"], h["theta500"], h["z"]
                cred = credited_build(r, p)
                fidx, fqa = r[p + "f_idx"], r[p + "f_q_alone"]
                sig1 = r[p + "it1_sig"]
            r1 = sig1 / V.sigma_Y(mode, TH)
            sigfin = np.interp(np.log(th), np.log(TH), r1) * V.sigma_Y(mode, th)
            qp = Y / sigfin * np.interp(np.log(th), np.log(TH), rnoise)
            k = qp > 1.4
            Nd.append(np.histogram(z[cred], ZB)[0]); Na.append(np.histogram(z[fidx[fqa > 5]], ZB)[0])
            Ne.append(np.histogram(z[k], ZB, weights=p_alone(P, qp[k], th[k]))[0])
            for ib, Pbb in enumerate(Pb):
                Ne_b[ib].append(np.histogram(z[k], ZB, weights=p_alone(Pbb, qp[k], th[k]))[0].sum())
            for lab, (fits, thinj, rinj, _) in sels.items():
                qq = Y / sigfin * np.interp(np.log(th), np.log(TH), rinj)
                kk = qq > 1.4
                Ne_s[lab].append(np.histogram(z[kk], ZB, weights=p_fit(fits, thinj, qq[kk], th[kk]))[0])
        Nd, Na, Ne = np.array(Nd, float), np.array(Na, float), np.array(Ne)
        ecal = np.std([Na.sum() / np.sum(v) - 1 for v in Ne_b])
        T2, e2, z2 = boot(Na, Ne); T1, e1, z1 = boot(Nd, Ne)
        Ts, es, _ = boot(Nd, Na)
        log(f"  {sky} ({len(units)} tiles): credited {Nd.sum():.0f}, isolated search {Na.sum():.0f}, N_exp_alone {Ne.sum():.1f}")
        log(f"     split 1 (credited vs isolated search): {100 * Ts:+.2f} +- {100 * es:.2f}%")
        log(f"     split 2' (isolated search vs criterion-matched unit-noise calibration): {100 * T2:+.2f} +- {100 * e2:.2f} (tiles) +- {100 * ecal:.2f}% (calibration)"
            f"; z bins " + ", ".join(f"{100 * v:+.1f}" for v in z2))
        log(f"     N1' (credited vs criterion-matched calibration): {100 * T1:+.2f} +- {100 * e1:.2f} +- {100 * ecal:.2f}%; z bins " + ", ".join(f"{100 * v:+.1f}" for v in z1))
        for lab in sels:
            Nes = np.array(Ne_s[lab])
            Tq, eq, zq = boot(Nd, Nes); Tq2, eq2, _ = boot(Na, Nes)
            log(f"     with the {lab}: N1 {100 * Tq:+.2f} +- {100 * eq:.2f}%, split 2 {100 * Tq2:+.2f} +- {100 * eq2:.2f}% (tile errors only); z 2-3 {100 * zq[-1]:+.1f}%")
log(f"\ndone [{time.time() - t00:.0f} s]")
