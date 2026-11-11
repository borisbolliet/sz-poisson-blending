"""v11: population-level blending fractions from the MAPS with the build's dominance rule (v10), vs the build.

For each run and theta_A node, the dominance-aware map probabilities at q_A = Qd[i] use the lattice points with
x_max < q_A (classes 0..i): det, B2(eps) [(delta-mu)/q_A > eps among detections], B2neg, B3 (boosted over the
threshold), lost (of single-cluster detections), mean (delta/q_A of detections); the same without dominance
(all classes) as a cross-check of v05.  Primary populations: the build's H on the 12 nodes (v06) and the
realised primaries of the 400 v05 patches (v08).  Errors: jackknife over v10 blocks (P(D) noise) and over
v08 blocks (population noise), added in quadrature.  Also: Delta N/N in z bins and the explicit injections
vs the dominance-aware prediction.
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402
from scipy.special import ndtr                      # noqa: E402

log = V.Tee(os.path.join(V.HERE, "v11_compare_dom.log"))
t0 = time.time()
E = V.hist_edges()
XC = 0.5 * (E[1:] + E[:-1])
blocks = sorted(glob.glob(os.path.join(V.OUT, "v10_block*.npz")))
NB = len(blocks)
d0 = np.load(blocks[0])
nodes, RUNS, QD, CE = d0["nodes"], [str(r) for r in d0["runs"]], d0["QD"], d0["CE"]
XCC = 0.5 * (CE[1:] + CE[:-1])
NCL = QD.size + 1
B6 = np.load(os.path.join(V.OUT, "v06_pd_builder_nodes.npz"))
HR = np.load(os.path.join(V.OUT, "v08_Hreal.npz"))
QN, EPS, QCUTS, ZB = B6["qnodes"], B6["eps"], B6["qcuts"], B6["zb"]
NN, NQ, NC = len(nodes), len(QN), len(QCUTS)
iq_of = np.full(NQ, -1)
for k, q in enumerate(QN):
    m = np.nonzero(np.abs(QD - q) < 1e-9)[0]
    if m.size:
        iq_of[k] = m[0]
log(f"[v11] {NB} v10 blocks; {NN} nodes; {NCL} x_max classes; q nodes with dominance classes: {(iq_of >= 0).sum()}")


def load(run, j):
    H0c = np.zeros((NB, V.H_NB)); H0s = np.zeros(V.H_NB)
    Hc = np.zeros((NB, NCL, CE.size - 1)); Hs = np.zeros((NCL, CE.size - 1))
    for b, f in enumerate(blocks):
        with np.load(f) as d:
            H0c[b] = d[f"H0c_{run}_{j}"]; H0s += d[f"H0s_{run}_{j}"]
            Hc[b] = d[f"Hc_{run}_{j}"]; Hs += d[f"Hs_{run}_{j}"]
    return H0c, H0s, Hc, Hs


def node_probs(H0c, H0s, Hc, Hs, dom):
    """Per block: arrays (NB, NC, NQ[, nE]) of probabilities (already divided by the block's lattice count)."""
    c0 = H0c.sum(0)
    x0 = np.where(c0 > 0, H0s / np.maximum(c0, 1), XC)
    nz0 = c0 > 0
    A0, x0 = H0c[:, nz0], x0[nz0]
    Ntot = H0c.sum(1) + Hc.sum((1, 2))                                   # lattice points per block
    # cumulative class histograms 1..i (x_max < QD[i])
    Ccum = np.cumsum(Hc[:, 1:], axis=1)                                  # (NB, NCL-1, ncb): classes 1..i+1 -> index i
    Scum = np.cumsum(Hs[1:], axis=0)
    out = {k: np.zeros((NB, NC, NQ)) for k in ("det", "b3", "lost", "mean")}
    out["b2"] = np.zeros((NB, NC, NQ, len(EPS))); out["b2n"] = np.zeros_like(out["b2"])
    for k, q in enumerate(QN):
        if q < 0.4999:
            continue
        i = iq_of[k]
        if dom:
            # allowed classes: 0..i  (class c covers [QD[c-1], QD[c]); allowed iff QD[c] <= q)
            Cc = Ccum[:, i - 1] if i >= 1 else np.zeros((NB, CE.size - 1))
            Sc = Scum[i - 1] if i >= 1 else np.zeros(CE.size - 1)
        else:
            Cc = Ccum[:, -1]; Sc = Scum[-1]
        cc = Cc.sum(0)
        xc = np.where(cc > 0, Sc / np.maximum(cc, 1), XCC)
        nzc = cc > 0
        Ac, xc = Cc[:, nzc], xc[nzc]
        A = np.concatenate([A0, Ac], axis=1)
        x = np.concatenate([x0, xc])
        for ic, qc in enumerate(QCUTS):
            Phi = ndtr(q + x - qc)
            Phi0 = ndtr(q - qc)
            out["det"][:, ic, k] = A @ Phi
            out["b3"][:, ic, k] = A @ ((Phi - Phi0) * (x > 0))
            out["lost"][:, ic, k] = A @ ((Phi0 - Phi) * (x < 0))
            out["mean"][:, ic, k] = A @ (Phi * x) / q
            for ie, e in enumerate(EPS):
                out["b2"][:, ic, k, ie] = A @ (Phi * (x > e * q))
                out["b2n"][:, ic, k, ie] = A @ (Phi * (x < -e * q))
    for v in out.values():
        v /= Ntot.reshape((NB,) + (1,) * (v.ndim - 1))
    return out, Ntot


def jk(v):
    v = np.asarray(v); n = len(v)
    return np.sqrt((n - 1) / n * ((v - v.mean(0)) ** 2).sum(0))


def fr(P, ic, H):
    Phi = ndtr(QN[None, :] - QCUTS[ic])
    ns = (H * Phi).sum(); nd = (H * P["det"][ic]).sum()
    return dict(Ns=ns, Nd=nd, dN=nd / ns - 1, B2_10=(H * P["b2"][ic, :, :, 1]).sum() / nd,
                B2_20=(H * P["b2"][ic, :, :, 2]).sum() / nd, B3=(H * P["b3"][ic]).sum() / nd,
                lost=(H * P["lost"][ic]).sum() / ns, mb=(H * P["mean"][ic]).sum() / nd,
                B2n_10=(H * P["b2n"][ic, :, :, 1]).sum() / nd)


KS = ("dN", "B2_10", "B2_20", "B3", "lost", "mb", "B2n_10")
HEAD = {"baseline": dict(dN=0.0129, B2_10=0.0879, B2_20=0.0470, B3=0.0622, lost=0.0491, mb=0.0223, B2n_10=0.0206),
        "goal": dict(dN=0.0138, B2_10=0.1037, B2_20=0.0565, B3=0.0687, lost=0.0544, mb=0.0266, B2n_10=0.0277),
        "white": dict(dN=-0.0162, B2_10=0.1624, B2_20=0.1052, B3=0.0812, lost=0.0902, mb=0.0338, B2n_10=0.0571),
        "baseline_sub": dict(dN=0.0158, B2_10=0.0772, B2_20=0.0391, B3=0.0569, lost=0.0419, mb=0.0187, B2n_10=np.nan),
        "goal_sub": dict(dN=0.0178, B2_10=0.0894, B2_20=0.0459, B3=0.0619, lost=0.0451, mb=0.0218, B2n_10=np.nan)}
summary = {}
NODEP = {}
for run in RUNS:
    tr = time.time()
    m = run.replace("_sub", "")
    Pd, Pn = {}, {}
    for j in range(NN):
        H0c, H0s, Hc, Hs = load(run, j)
        for dom, PP in ((True, Pd), (False, Pn)):
            o, Ntot = node_probs(H0c, H0s, Hc, Hs, dom)
            for k, v in o.items():
                PP.setdefault(k, np.zeros((NB, NC, NN) + v.shape[2:]))
                PP[k][:, :, j] = v
    w = np.full(NB, 1.0 / NB)
    tot = {d: {k: v.mean(0) for k, v in PP.items()} for d, PP in (("dom", Pd), ("nodom", Pn))}
    loo = {d: [{k: (v.sum(0) - v[b]) / (NB - 1) for k, v in PP.items()} for b in range(NB)] for d, PP in (("dom", Pd), ("nodom", Pn))}
    NODEP.update({f"{run}_{d}_{k}": v for d in tot for k, v in tot[d].items()})
    Hb = B6[f"{run}_H"]
    Hr_b = HR[f"Hr_{m}"]                                   # (NBLK, nz, NN, NQ)
    Hr = Hr_b.sum((0, 1))
    Pb = {d: {k: B6[f"{run}_{k}"][d] for k in ("det", "b3", "lost", "mean", "b2", "b2n")} for d in (0, 1)}
    log(f"\n[{run}]  (fractions of detections; lost of single-cluster detections; mb = mean (delta-mu)/q_A of detections)")
    for ic, qc in enumerate(QCUTS):
        for label, H, Hblk in (("build H", Hb, None), ("realised", Hr, Hr_b)):
            f = fr(tot["dom"], ic, H)
            e_pd = {k: jk([fr(l, ic, H)[k] for l in loo["dom"]]) for k in KS}
            if Hblk is not None:
                nbk = Hblk.shape[0]
                e_pop = {k: jk([fr(tot["dom"], ic, H - Hblk[b].sum(0))[k] for b in range(nbk)]) for k in KS}
            else:
                e_pop = {k: 0.0 for k in KS}
            e = {k: np.hypot(e_pd[k], e_pop[k]) for k in KS}
            fb = fr(Pb[1], ic, H)
            fn = fr(tot["nodom"], ic, H)
            fbn = fr(Pb[0], ic, H)
            log(f"   q>{qc:.0f} [{label:8s}] MAP dom   : " + " ".join(f"{k} {f[k]:+.4f}+-{e[k]:.4f}" for k in KS)
                + f"   (N_single {f['Ns']:,.0f}, N_det {f['Nd']:,.0f})")
            log(f"          {'':10s} BUILD dom : " + " ".join(f"{k} {fb[k]:+.4f} ({(f[k]-fb[k])/e[k]:+.1f}s)" for k in KS))
            log(f"          {'':10s} MAP nodom : " + " ".join(f"{k} {fn[k]:+.4f}" for k in KS[:4]) + " | BUILD nodom: "
                + " ".join(f"{k} {fbn[k]:+.4f}" for k in KS[:4]))
            if qc == 5.0 and label == "build H":
                summary[run] = (f, e, fb)
                log(f"          {'':10s} BUILD headline (48 nodes, dom): " + " ".join(f"{k} {HEAD[run][k]:+.4f}" for k in KS))
            if qc == 5.0 and label == "realised":
                summary[run + "_real"] = (f, e, fb)
    if run in ("baseline", "goal", "white"):
        Hz = HR[f"Hr_{m}"].sum(0)                             # (nz, NN, NQ)
        Hzb = B6[f"{run}_Hz"]
        for label, HH in (("build H", Hzb), ("realised", Hz)):
            zm, ze, zb_ = [], [], []
            for b in range(len(ZB) - 1):
                zm.append(fr(tot["dom"], 1, HH[b])["dN"])
                ze.append(jk([fr(l, 1, HH[b])["dN"] for l in loo["dom"]]))
                zb_.append(fr(Pb[1], 1, HH[b])["dN"])
            log(f"   q>5 Delta N/N in z bins {list(ZB)} [{label}]:")
            log(f"        MAP dom   : " + " ".join(f"{a*100:+.2f}+-{e*100:.2f}%" for a, e in zip(zm, ze)))
            log(f"        BUILD dom : " + " ".join(f"{a*100:+.2f}%" for a in zb_))
    log(f"   [{time.time()-tr:.0f}s]")

# Gaussian-equivalent selection (claim gaussian_equivalent): sigma_Y -> sigma_Y sqrt(1 + v(theta)), v = OUR map variance
mom = np.load(os.path.join(V.OUT, "v05_moments.npz"))
M5 = mom["mom"]
RUN5 = [str(r) for r in mom["runs"]]
GE = {("baseline", "all"): {5: 85988, 4: 143875, 6: 57574, 7: 41081}, ("baseline", "sub"): {4: 135800, 5: 83170, 6: 56332, 7: 40424},
      ("goal", "all"): {4: 227707, 5: 137304, 6: 92901, 7: 67065}, ("goal", "sub"): {4: 209604, 5: 130832, 6: 90048, 7: 65553}}
log("\nGaussian-equivalent selection with OUR map confusion variance (v05 moments) vs the dominance-aware MAP P(D) counts:")
for run in ("baseline", "goal", "baseline_sub", "goal_sub"):
    k5 = RUN5.index(run)
    v = np.array([M5[:, k5 * NN + j, 1].sum() / M5[:, k5 * NN + j, 2].sum() for j in range(NN)])
    Hb = B6[f"{run}_H"]
    P_ = {kk: NODEP[f"{run}_dom_{kk}"] for kk in ("det",)}
    row = []
    for ic, qc in enumerate(QCUTS):
        Ng = (Hb * ndtr((QN[None, :] - qc) / np.sqrt(1 + v[:, None]))).sum()
        Nd = (Hb * P_["det"][ic]).sum()
        Ns = (Hb * ndtr(QN[None, :] - qc)).sum()
        ref = GE[(run.replace("_sub", ""), "sub" if run.endswith("_sub") else "all")][int(qc)]
        row.append(f"q>{qc:.0f}: N_gauss {Ng:,.0f} (build {ref:,}) N_det,map {Nd:,.0f} (N_det/N_gauss - 1 = {Nd/Ng-1:+.4f}; N_single {Ns:,.0f})")
    log(f"   [{run}] v(theta nodes) = {np.array2string(v, precision=3)}")
    for r_ in row:
        log(f"      {r_}")

# explicit injections vs the dominance-aware map / build prediction at the same node and q_A
inj = mom["inj"]
log("\nExplicit injections (v05; painted into halo sky + CMB + noise; detection Y_hat/sigma > 5; no dominance applied to the"
    " injected objects) vs predictions at the same (theta_A, q_A):")
for st, run in ((0, "baseline"), (1, "goal")):
    for a, tA in ((0, 1.0069), (1, 4.025)):
        j = int(np.argmin(np.abs(nodes - tA)))
        for qa in (5.0, 7.0):
            k = int(np.argmin(np.abs(QN - qa)))
            s = (inj[:, 1] == st) & (inj[:, 2] == a) & (np.abs(inj[:, 3] - qa) < 1e-9)
            r = inj[s]
            Y, tot_, iso, sg_ = r[:, 4], r[:, 5], r[:, 6], r[:, 9]
            boost = (tot_ - iso) / Y
            det = tot_ / sg_ > 5.0
            deti = iso / sg_ > 5.0
            nd = det.sum()
            vals = dict(B2_10=(det & (boost > 0.1)).sum() / nd, B2_20=(det & (boost > 0.2)).sum() / nd,
                        B3=(det & ~deti).sum() / nd, lost=(deti & ~det).sum() / max(deti.sum(), 1), mb=boost[det].mean(),
                        mball=boost.mean())
            err = dict(B2_10=np.sqrt(vals["B2_10"] * (1 - vals["B2_10"]) / nd), B2_20=np.sqrt(vals["B2_20"] * (1 - vals["B2_20"]) / nd),
                       B3=np.sqrt(vals["B3"] * (1 - vals["B3"]) / nd), lost=np.sqrt(vals["lost"] * (1 - vals["lost"]) / max(deti.sum(), 1)),
                       mb=boost[det].std() / np.sqrt(nd), mball=boost.std() / np.sqrt(len(boost)))
            preds = {}
            for name, src in (("map dom", "dom"), ("map nodom", "nodom")):
                P_ = {kk: NODEP[f"{run}_{src}_{kk}"] for kk in ("det", "b2", "b3", "lost", "mean")}
                dt = P_["det"][1, j, k]
                preds[name] = dict(B2_10=P_["b2"][1, j, k, 1] / dt, B2_20=P_["b2"][1, j, k, 2] / dt, B3=P_["b3"][1, j, k] / dt,
                                   lost=P_["lost"][1, j, k] / ndtr(qa - 5), mb=P_["mean"][1, j, k] / dt, mball=0.0)
            for name, d in (("build nodom", 0), ("build dom", 1)):
                dt = B6[f"{run}_det"][d, 1, j, k]
                preds[name] = dict(B2_10=B6[f"{run}_b2"][d, 1, j, k, 1] / dt, B2_20=B6[f"{run}_b2"][d, 1, j, k, 2] / dt,
                                   B3=B6[f"{run}_b3"][d, 1, j, k] / dt, lost=B6[f"{run}_lost"][d, 1, j, k] / ndtr(qa - 5),
                                   mb=B6[f"{run}_mean"][d, 1, j, k] / dt, mball=0.0)
            log(f"   [{run}] theta_A={nodes[j]:.3f}' q_A={qa}: N={len(r)}, detected {nd}")
            for kk in ("B2_10", "B2_20", "B3", "lost", "mb", "mball"):
                log(f"      {kk:6s}: injected {vals[kk]:+.4f} +- {err[kk]:.4f} | " + " | ".join(
                    f"{nm} {pv[kk]:+.4f} ({(vals[kk]-pv[kk])/err[kk]:+.1f}s)" for nm, pv in preds.items()))
np.savez(os.path.join(V.OUT, "v11_nodeprobs.npz"), **NODEP)
log(f"[v11] done {time.time()-t0:.0f}s")
