"""x12 (indep lens): how much of the build's N_exp (claim N1, N3) comes from halos far below the threshold.

The build's nominal N_exp = sum over ALL halos of P_sel(q', theta500) with P_sel = Phi((sqrt(q'^2 + b) - 5 - mu) / s)
(t05_analysis.py).  For q'^2 < -b the fitted model has a floor Phi((-5 - mu) / s) ~ 1e-5 - 1e-4, while the injections
(q_inj >= 1.5) never detect a cluster below q_inj = 2 (t03_injections.log; own x05) and a credited detection needs a
filtered tSZ >= 2 sigma.  Each tile has ~70,000 halos, almost all with q' << 1, so the floor can add up.
Here, on the build's own skies (stored arrays, the build's selection fit and r_inj, exactly as in t05 / x07):
  N_exp from all halos vs from halos with q' > 1.4 / 2.0 / 2.5, the per-tile contribution of q' < 1.4, and Delta/N
  (claims N1, N3) with each choice; the floor value per injected size.
Usage: python x12_selection_tail.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import ndtr
import vlib3 as V

log = V.Tee(os.path.join(V.HERE, "x12_selection_tail.log"))
t00 = time.time()
TH = V.THETAS
ZB = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
SB = dict(np.load(os.path.join(V.BUILD_DATA, "selection.npz")))
TI = SB["theta_inj"]
rng = np.random.default_rng(5)


def p_sel(fits, q, th):
    P = np.array([ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)) for b, mu, ls in fits])
    x = np.clip(np.log(th), np.log(TI[0]), np.log(TI[-1]))
    i = np.clip(np.searchsorted(np.log(TI), x) - 1, 0, len(TI) - 2)
    w = (x - np.log(TI[i])) / np.log(TI[i + 1] / TI[i])
    n = np.arange(q.size)
    return (1 - w) * P[i, n] + w * P[i + 1, n]


log("x12: the build's N_exp and its far-sub-threshold part, build's skies (324 tiles), build's selection fit")
for mode in V.MODES:
    fits = SB[mode + "_fit"]
    log(f"\n==================== {mode}")
    log("  fitted floor Phi((-5 - mu) / s) per injected size: " + ", ".join(f"{t}': {ndtr((-5 - mu) / np.exp(ls)):.1e}" for t, (b, mu, ls) in zip(TI, fits)))
    for t in TI:
        q = SB[f"{mode}_{t}_q"]; qo = SB[f"{mode}_{t}_qobs"]
        m = q < 2.0
        log(f"  injections theta {t}': q_inj 1.5-2.0: {np.sum(qo[m] > 5)} of {m.sum()} detected; model mean P there {np.mean(p_sel(fits, q[m], np.full(m.sum(), t))):.1e}")
    dB = os.path.join(V.BUILD_DATA, "injections")
    rinj = np.mean([np.load(os.path.join(dB, f))["sig"] for f in sorted(os.listdir(dB)) if f.startswith(f"inj_{mode}_")], 0) / V.sigma_Y(mode, TH)
    Nd, Ne = [], {c: [] for c in (0.0, 1.4, 2.0, 2.5)}
    nlow = []
    for pid in range(324):
        r = V.build_tile(pid); h = V.build_truth(pid)
        p = mode + "_"
        q, j = r[p + "it1_q"], r[p + "it1_j"]
        d5 = np.flatnonzero(q > 5)
        ti, ex = r[p + "d_top_idx"][d5], r[p + "d_top_ex"][d5]
        real = ex[:, 0] / r[p + "it1_sig"][j[d5]] >= 2
        seen = set()
        for a in np.argsort(-q[d5]):
            if real[a] and ti[a, 0] not in seen:
                seen.add(int(ti[a, 0]))
        cred = np.array(sorted(seen), np.int64)
        Y, th, z = h["Y500"], h["theta500"], h["z"]
        sigfin = np.interp(np.log(th), np.log(TH), r[p + "it1_sig"] / V.sigma_Y(mode, TH)) * V.sigma_Y(mode, th)
        qp = Y / sigfin * np.interp(np.log(th), np.log(TH), rinj)
        P = p_sel(fits, qp, th)
        Nd.append(np.histogram(z[cred], ZB)[0])
        for c in Ne:
            Ne[c].append(np.histogram(z[qp > c], ZB, weights=P[qp > c])[0])
        nlow.append((P[qp < 1.4].sum(), (qp < 1.4).sum()))
    Nd = np.array(Nd, float)
    nlow = np.array(nlow)
    log(f"  per tile: halos with q' < 1.4: {nlow[:, 1].mean():.0f}; their summed P_sel {nlow[:, 0].mean():.2f} (vs credited {Nd.sum(1).mean():.1f} per tile)")
    for c in Ne:
        E = np.array(Ne[c])
        T = Nd.sum() / E.sum() - 1
        b = rng.integers(0, 324, (2000, 324))
        eT = (Nd[b].sum(1).sum(1) / E[b].sum(1).sum(1) - 1).std()
        zb = Nd.sum(0) / E.sum(0) - 1
        log(f"  N_exp from halos with q' > {c}: {E.sum():.1f}; Delta/N = {100 * T:+.2f} +- {100 * eT:.2f}%; z bins " + ", ".join(f"{100 * v:+.1f}" for v in zb)
            + ("   <- the build's nominal number (claims N1, N3)" if c == 0.0 else ""))
    E0, E14 = np.array(Ne[0.0]).sum(0), np.array(Ne[1.4]).sum(0)
    log("  share of N_exp from q' < 1.4, per z bin: " + ", ".join(f"{100 * (a - b_) / a:.2f}%" for a, b_ in zip(E0, E14)))
log(f"\ndone [{time.time() - t00:.0f} s]")

# ---------------------------------------------------------------- the same on the own fresh skies (x04)
log("\n-------- own fresh skies (x04, 100 tiles): N_exp with all halos vs q' > 1.4, build's selection fit (as t05) and own pooled fit")
from scipy.optimize import minimize


def fit_sel(q, det):
    def nll(p):
        b, mu, ls = p
        pr = np.clip(ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)), 1e-12, 1 - 1e-12)
        return -np.sum(det * np.log(pr) + (1 - det) * np.log(1 - pr))
    return minimize(nll, [0.0, 0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9, maxiter=8000)).x


FD = os.path.join(V.OUT, "fresh")
ffiles = sorted(f for f in os.listdir(FD) if f.startswith("fresh_"))
for mode in V.MODES:
    rows = {}; sig = []
    for d in (os.path.join(V.OUT, "inj"), os.path.join(V.OUT, "inj2")):
        for f in sorted(os.listdir(d)):
            if f.startswith(f"inj_{mode}_"):
                r = dict(np.load(os.path.join(d, f)))
                rows.setdefault(float(r["theta_inj"]), ([], []))
                rows[float(r["theta_inj"])][0].append(r["q_inj"]); rows[float(r["theta_inj"])][1].append(r["qbest"])
                sig.append(r["sig"] / V.sigma_Y(mode, TH))
    own_fits = np.array([fit_sel(np.concatenate(rows[t][0]), (np.concatenate(rows[t][1]) > 5).astype(float)) for t in TI])
    rinj_own = np.mean(sig, 0)
    dB = os.path.join(V.BUILD_DATA, "injections")
    rinj_b = np.mean([np.load(os.path.join(dB, f))["sig"] for f in sorted(os.listdir(dB)) if f.startswith(f"inj_{mode}_")], 0) / V.sigma_Y(mode, TH)
    Nd = []; E = {k: [] for k in ("b_all", "b_14", "o_all", "o_14")}
    for f in ffiles:
        r = dict(np.load(os.path.join(FD, f))); p = mode + "_"
        h = V.draw_tile_halos(7_000_000 + int(r["tile"]))
        Y, th, z = h["Y500"], h["theta500"], h["z"]
        q, j = r[p + "q"], r[p + "j"]
        d5 = r[p + "d5"]; ti, ex = r[p + "top_idx"], r[p + "top_ex"]
        real = ex[:, 0] / r[p + "sig1"][j[d5]] >= 2
        seen = set()
        for a in np.argsort(-q[d5]):
            if real[a] and ti[a, 0] not in seen:
                seen.add(int(ti[a, 0]))
        Nd.append(np.histogram(z[np.array(sorted(seen), np.int64)], ZB)[0])
        sigfin = np.interp(np.log(th), np.log(TH), r[p + "sig1"] / V.sigma_Y(mode, TH)) * V.sigma_Y(mode, th)
        for lab, fits, rinj in (("b", SB[mode + "_fit"], rinj_b), ("o", own_fits, rinj_own)):
            qp = Y / sigfin * np.interp(np.log(th), np.log(TH), rinj)
            P = p_sel(fits, qp, th)
            E[lab + "_all"].append(np.histogram(z, ZB, weights=P)[0])
            E[lab + "_14"].append(np.histogram(z[qp > 1.4], ZB, weights=P[qp > 1.4])[0])
    Nd = np.array(Nd, float)
    log(f"  {mode}: credited {Nd.sum():.0f}")
    for k, lab in (("b_all", "build's fit, all halos (t05 convention)"), ("b_14", "build's fit, q' > 1.4"),
                   ("o_all", "own pooled fit (x03 + x03b), all halos"), ("o_14", "own pooled fit (x03 + x03b), q' > 1.4")):
        Ek = np.array(E[k])
        T = Nd.sum() / Ek.sum() - 1
        b = rng.integers(0, Nd.shape[0], (2000, Nd.shape[0]))
        eT = (Nd[b].sum(1).sum(1) / Ek[b].sum(1).sum(1) - 1).std()
        zz = Nd.sum(0) / Ek.sum(0) - 1
        bz = (Nd[b].sum(1)[:, -1] / Ek[b].sum(1)[:, -1] - 1).std()
        log(f"     {lab:45s}: N_exp {Ek.sum():.1f}; Delta/N = {100 * T:+.2f} +- {100 * eT:.2f}%; z bins " + ", ".join(f"{100 * v:+.1f}" for v in zz)
            + f" (z 2-3 +- {100 * bz:.1f})")
log(f"done [{time.time() - t00:.0f} s]")
