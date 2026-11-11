"""v07: map-level halo skies (v05) vs the build's catalogue-level P(D) (v06 at the same nodes; build logs).

(A) random-point distribution of delta = Y_hat_A(neighbours)/sigma_A at theta_A = 1.007, 4.025, 16.09' (baseline,
    goal): moments, quantiles, tail probabilities: maps (jackknife over 20 blocks of 20 patches) vs the build's FFT
    P(D) (v06) and vs the numbers printed in the build's s06d log.  Plus variance (claim confusion_var) and the
    robust (16-84%) half-width of n + delta (claim robust_width) at all 12 nodes.
(B) per-object fractions at fixed (theta_A, q_A), q_cut = 5: from the map P(D) vs the build (no dominance /
    with dominance); and the EXPLICIT injection counts (v05: clusters painted into halo sky + CMB + noise).
(C) population-level fractions (claims go_nogo_fractions, subthreshold, detections net change): map P(D) at 12 nodes
    integrated over the build's primary population H on the same nodes, vs the build's P(D) on the same nodes
    (no dominance / dominance) and vs the build's headline (48 nodes, dominance).
(D) redshift tilt Delta N/N in z bins (claim ztilt).
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402
from scipy.special import ndtr                      # noqa: E402
from scipy.optimize import brentq                   # noqa: E402

log = V.Tee(os.path.join(V.HERE, "v07_compare.log"))
t0 = time.time()
E = V.hist_edges()
XC = 0.5 * (E[1:] + E[:-1])
blocks = sorted(glob.glob(os.path.join(V.OUT, "v05_block*.npz")))
NB = len(blocks)
mom = np.load(os.path.join(V.OUT, "v05_moments.npz"))
nodes = mom["nodes"]
RUNS = [str(r) for r in mom["runs"]]
B6 = np.load(os.path.join(V.OUT, "v06_pd_builder_nodes.npz"))
QN, EPS, QCUTS, ZB = B6["qnodes"], B6["eps"], B6["qcuts"], B6["zb"]
assert np.allclose(B6["nodes"], nodes)
npatch = mom["pinfo"].shape[0]
log(f"[v07] {NB} histogram blocks, {npatch} patches ({npatch} x 100 deg^2), nodes {np.round(nodes, 4)}")
pinfo = mom["pinfo"]
log(f"   halos per patch {pinfo[:,1].mean():,.1f} (expected 71,014.3); unpainted theta500 > 60' halos: {int(pinfo[:,2].sum())} "
    f"in {npatch} patches; mean y over patches {pinfo[:,3].mean():.4e} +- {pinfo[:,3].std()/np.sqrt(npatch):.1e} "
    f"(ensemble {float(mom['ybar']):.4e})")


def load_hist(run, j):
    C = np.zeros((NB, V.H_NB)); S1 = np.zeros(V.H_NB)
    for b, f in enumerate(blocks):
        with np.load(f) as d:
            C[b] = d[f"cnt_{run}_{j}"]; S1 += d[f"s1_{run}_{j}"]
    return C, S1


def xrep(C, S1):
    c = C.sum(0)
    return np.where(c > 0, S1 / np.maximum(c, 1), XC)


def jk(vals_loo):
    """Jackknife error from leave-one-out estimates."""
    v = np.asarray(vals_loo)
    n = len(v)
    return np.sqrt((n - 1) / n * ((v - v.mean(0)) ** 2).sum(0))


def dist_stats(p, x):
    """p: probabilities per bin (sum 1), x: representative values.  Returns dict of statistics."""
    out = {}
    out["mean"] = float((p * x).sum())
    cdf = np.cumsum(p)
    for ql in (0.02275, 0.15866, 0.5, 0.84134, 0.97725):
        k = int(np.searchsorted(cdf, ql))
        lo = cdf[k - 1] if k > 0 else 0.0
        fr = (ql - lo) / max(p[k], 1e-300)
        out[f"q{ql*100:.2f}"] = float(E[k] + fr * (E[k + 1] - E[k]))
    for t in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
        out[f"P>{t}"] = float(p[x > t].sum())
    for t in (0.1, 0.2, 0.5, 1.0):
        out[f"P<-{t}"] = float(p[x < -t].sum())
    # robust half-width of n + delta (n ~ N(0,1)), on the non-empty bins
    nz = p > 0
    pp, xx = p[nz], x[nz]
    F = lambda t: float((pp * ndtr(t - xx)).sum())
    t16 = brentq(lambda t: F(t) - 0.15866, -10, 10)
    t84 = brentq(lambda t: F(t) - 0.84134, -10, 10)
    out["rw"] = 0.5 * (t84 - t16)
    return out


# ---------------------------------------------------------------- (A) random-point distribution
S06D = {  # the build's s06d log: analytic (FFT) values at theta_A = 1.007, 4.025, 16.09'
    "baseline": {1.0069: dict(var=0.33481, **{"q2.27": -0.7863, "q15.87": -0.2554, "q50.00": -0.0349, "q84.13": 0.2051, "q97.72": 1.0230},
                              **{"P>0.1": .25705, "P>0.2": .16209, "P>0.5": .06086, "P>1.0": .02353, "P>2.0": .00788, "P>5.0": .00141,
                                 "P<-0.1": .36429, "P<-0.2": .21201, "P<-0.5": .05508, "P<-1.0": .01359}),
                 4.025: dict(var=0.41601, **{"q2.27": -0.8103, "q15.87": -0.2794, "q50.00": -0.0498, "q84.13": 0.2125, "q97.72": 1.1598},
                             **{"P>0.1": .25208, "P>0.2": .16628, "P>0.5": .06829, "P>1.0": .02808, "P>2.0": .00985, "P>5.0": .00187,
                                "P<-0.1": .39913, "P<-0.2": .23848, "P<-0.5": .06071, "P<-1.0": .01424}),
                 16.0897: dict(var=0.30642, **{"q2.27": -0.4944, "q15.87": -0.2245, "q50.00": -0.0785, "q84.13": 0.1481, "q97.72": 1.0302},
                               **{"P>0.1": .19432, "P>0.2": .13035, "P>0.5": .05629, "P>1.0": .02371, "P>2.0": .00834, "P>5.0": .00156,
                                  "P<-0.1": .43906, "P<-0.2": .19650, "P<-0.5": .02204, "P<-1.0": .00265})},
    "goal": {1.0069: dict(var=0.47991, **{"q2.27": -0.9457, "q15.87": -0.3029, "q50.00": -0.0405, "q84.13": 0.2430, "q97.72": 1.2210},
                          **{"P>0.1": .27769, "P>0.2": .18556, "P>0.5": .07532, "P>1.0": .03036, "P>2.0": .01060, "P>5.0": .00203,
                             "P<-0.1": .39291, "P<-0.2": .24993, "P<-0.5": .07529, "P<-1.0": .02026}),
             4.025: dict(var=0.59615, **{"q2.27": -0.9846, "q15.87": -0.3319, "q50.00": -0.0566, "q84.13": 0.2537, "q97.72": 1.3821},
                         **{"P>0.1": .27252, "P>0.2": .18909, "P>0.5": .08322, "P>1.0": .03568, "P>2.0": .01305, "P>5.0": .00265,
                            "P<-0.1": .42563, "P<-0.2": .27890, "P<-0.5": .08442, "P<-1.0": .02201}),
             16.0897: dict(var=0.42977, **{"q2.27": -0.6068, "q15.87": -0.2651, "q50.00": -0.0894, "q84.13": 0.1761, "q97.72": 1.2175},
                           **{"P>0.1": .20958, "P>0.2": .14652, "P>0.5": .06740, "P>1.0": .02969, "P>2.0": .01091, "P>5.0": .00218,
                              "P<-0.1": .47454, "P<-0.2": .25204, "P<-0.5": .03786, "P<-1.0": .00547})}}
M = mom["mom"]                                       # (npatch, nrun*nnode, 5): sum x, sum x^2, n, min, max
nn = len(nodes)


def var_patch(run, j):
    k = RUNS.index(run) * nn + j
    s1, s2, n = M[:, k, 0], M[:, k, 1], M[:, k, 2]
    tot = s2.sum() / n.sum() - (s1.sum() / n.sum()) ** 2
    # jackknife over the same 20 blocks
    bs = npatch // NB
    loo = []
    for b in range(NB):
        msk = np.ones(npatch, bool); msk[b * bs:(b + 1) * bs] = False
        loo.append(s2[msk].sum() / n[msk].sum() - (s1[msk].sum() / n[msk].sum()) ** 2)
    return tot, jk(loo), s1.sum() / n.sum()


log("\n(A) random-point P(D) of delta: maps (+- jackknife) | build FFT at the same node (v06) | build s06d log (analytic)")
keys = ["mean", "q2.27", "q15.87", "q50.00", "q84.13", "q97.72", "P>0.1", "P>0.2", "P>0.5", "P>1.0", "P>2.0", "P>5.0",
        "P<-0.1", "P<-0.2", "P<-0.5", "P<-1.0", "rw"]
pulls = []
HCACHE = {}
for run in ("baseline", "goal"):
    for tA in (1.0069, 4.025, 16.0897):
        j = int(np.argmin(np.abs(nodes - tA)))
        C, S1 = load_hist(run, j)
        HCACHE[(run, j)] = (C, S1)
        x = xrep(C, S1)
        ctot = C.sum(0)
        st = dist_stats(ctot / ctot.sum(), x)
        loo = [dist_stats((ctot - C[b]) / (ctot - C[b]).sum(), x) for b in range(NB)]
        pb = B6[f"{run}_PD"][j]
        sb = dist_stats(pb / pb.sum(), XC)
        var, evar, _ = var_patch(run, j)
        log(f"   [{run}] theta_A = {nodes[j]:.4f}':  var map {var:.4f} +- {evar:.4f} | Parseval (build) {B6[f'{run}_varpars'][j]:.4f} "
            f"| s06d analytic {S06D[run][tA]['var']:.4f}")
        for k in keys:
            e = jk([l[k] for l in loo])
            ref = S06D[run][tA].get(k.replace("q2.27", "q2.27"), None)
            pull = (st[k] - sb[k]) / e if e > 0 else 0.0
            if k not in ("mean", "rw"):
                pulls.append(pull)
            log(f"      {k:7s}: map {st[k]:+.5f} +- {e:.5f} | build FFT {sb[k]:+.5f} (pull {pull:+.2f})"
                + (f" | s06d {ref:+.5f}" if ref is not None else ""))
pulls = np.array(pulls)
log(f"   pulls map vs build FFT over {pulls.size} quantities: rms {np.sqrt((pulls**2).mean()):.2f}, max |pull| {np.abs(pulls).max():.2f}, "
    f"{(np.abs(pulls) < 2).sum()} within 2 sigma")

log("\n(A2) all 12 nodes: variance (maps vs Parseval) and robust half-width of n + delta (maps vs build FFT)")
for run in RUNS:
    rows = []
    for j in range(nn):
        C, S1 = HCACHE.get((run, j)) or load_hist(run, j)
        x = xrep(C, S1)
        ctot = C.sum(0)
        rw = dist_stats(ctot / ctot.sum(), x)["rw"]
        pb = B6[f"{run}_PD"][j]
        rwb = dist_stats(pb / pb.sum(), XC)["rw"]
        var, evar, mean = var_patch(run, j)
        rows.append((nodes[j], var, evar, B6[f"{run}_varpars"][j] if run != "white" else B6[f"{run}_var"][j], rw, rwb, mean))
    log(f"   [{run}] theta_A   : " + " ".join(f"{r[0]:8.3f}" for r in rows))
    log(f"   [{run}] var map   : " + " ".join(f"{r[1]:8.4f}" for r in rows))
    log(f"   [{run}]  +- (jk)  : " + " ".join(f"{r[2]:8.4f}" for r in rows))
    log(f"   [{run}] var build : " + " ".join(f"{r[3]:8.4f}" for r in rows) + ("  (Parseval)" if run != "white" else "  (rate hist)"))
    log(f"   [{run}] rw map    : " + " ".join(f"{r[4]:8.4f}" for r in rows))
    log(f"   [{run}] rw build  : " + " ".join(f"{r[5]:8.4f}" for r in rows))
    log(f"   [{run}] mean map  : " + " ".join(f"{r[6]:+8.4f}" for r in rows))
log(f"   [{time.time()-t0:.0f}s]")


# ---------------------------------------------------------------- node probabilities from map histograms
def map_node_probs(C, x, qs, qcs):
    """Per block (rows of C) un-normalised sums for det, b2(eps), b2n(eps), b3, lost, mean; returns dict of arrays
    (nblock, nqc, nq[, neps]) of SUMS over pixels (normalise with the block counts)."""
    nz = C.sum(0) > 0
    Cn, xn = C[:, nz], x[nz]
    out = {k: np.zeros((C.shape[0], len(qcs), len(qs))) for k in ("det", "b3", "lost", "mean")}
    out["b2"] = np.zeros((C.shape[0], len(qcs), len(qs), len(EPS)))
    out["b2n"] = np.zeros_like(out["b2"])
    pos, neg = xn > 0, xn < 0
    for ic, qc in enumerate(qcs):
        Phi = ndtr(qs[:, None] + xn[None, :] - qc)                    # (nq, nb)
        Phi0 = ndtr(qs - qc)[:, None]
        out["det"][:, ic] = Cn @ Phi.T
        out["b3"][:, ic] = Cn @ ((Phi - Phi0) * pos[None, :]).T
        out["lost"][:, ic] = Cn @ ((Phi0 - Phi) * neg[None, :]).T
        out["mean"][:, ic] = Cn @ (Phi * xn[None, :]).T / qs[None, :]
        for ie, e in enumerate(EPS):
            out["b2"][:, ic, :, ie] = Cn @ (Phi * (xn[None, :] > e * qs[:, None])).T
            out["b2n"][:, ic, :, ie] = Cn @ (Phi * (xn[None, :] < -e * qs[:, None])).T
    return out, C.sum(1)


log("\n(B) per-object fractions at fixed (theta_A, q_A), q_cut = 5 [fractions of detections; lost of single-detectable]")
QB = np.array([5.0, 6.0, 7.0, 10.0])
inj = mom["inj"]
for run in ("baseline", "goal"):
    for tA in (1.0069, 4.025):
        j = int(np.argmin(np.abs(nodes - tA)))
        C, S1 = HCACHE[(run, j)]
        x = xrep(C, S1)
        P_, Nb = map_node_probs(C, x, QB, [5.0])
        tot = {k: v.sum(0) / Nb.sum() for k, v in P_.items()}
        loo = [{k: (v.sum(0) - v[b]) / (Nb.sum() - Nb[b]) for k, v in P_.items()} for b in range(NB)]
        for iq, qa in enumerate(QB):
            kq = int(np.argmin(np.abs(QN - qa)))
            def fr(dct, d=None):
                if d is None:
                    det = dct["det"][0, iq]
                    return dict(B2_10=dct["b2"][0, iq, 1] / det, B2_20=dct["b2"][0, iq, 2] / det, B3=dct["b3"][0, iq] / det,
                                lost=dct["lost"][0, iq] / ndtr(qa - 5.0), mb=dct["mean"][0, iq] / det, det=det)
                det = B6[f"{run}_det"][d, 1, j, kq]
                return dict(B2_10=B6[f"{run}_b2"][d, 1, j, kq, 1] / det, B2_20=B6[f"{run}_b2"][d, 1, j, kq, 2] / det,
                            B3=B6[f"{run}_b3"][d, 1, j, kq] / det, lost=B6[f"{run}_lost"][d, 1, j, kq] / ndtr(qa - 5.0),
                            mb=B6[f"{run}_mean"][d, 1, j, kq] / det, det=det)
            fm = fr(tot); fl = [fr(l) for l in loo]
            f0, f1 = fr(None, 0), fr(None, 1)
            line = "  ".join(f"{k} {fm[k]:.4f}+-{jk([l[k] for l in fl]):.4f} (build {f0[k]:.4f} / dom {f1[k]:.4f})"
                             for k in ("B2_10", "B2_20", "B3", "lost", "mb"))
            log(f"   [{run}] thA={nodes[j]:.3f}' qA={qa:4.1f}: P(det) map {fm['det']:.4f} build {f0['det']:.4f}; {line}")
        # explicit injection rows for this setup/theta
        a = 0 if tA < 2 else 1
        for qa in (5.0, 7.0):
            s = (inj[:, 1] == (0 if run == "baseline" else 1)) & (inj[:, 2] == a) & (np.abs(inj[:, 3] - qa) < 1e-9)
            r = inj[s]
            Y, tot_, iso, nl, dsk, sg_ = r[:, 4], r[:, 5], r[:, 6], r[:, 7], r[:, 8], r[:, 9]
            boost = (tot_ - iso) / Y
            lin = np.abs(boost - dsk * sg_ / Y).max()
            det = tot_ / sg_ > 5.0
            deti = iso / sg_ > 5.0
            n, nd = len(r), det.sum()
            b2 = (det & (boost > 0.1)).sum() / nd
            b22 = (det & (boost > 0.2)).sum() / nd
            b3 = (det & ~deti & (boost > 0)).sum() / nd
            lost = (deti & ~det & (boost < 0)).sum() / max(deti.sum(), 1)
            mbd = boost[det].mean()
            eb = np.sqrt(b2 * (1 - b2) / nd)
            kq = int(np.argmin(np.abs(QN - qa)))
            det0 = B6[f"{run}_det"][0, 1, j, kq]
            pred = B6[f"{run}_b2"][0, 1, j, kq, 1] / det0
            iq = int(np.argmin(np.abs(QB - qa)))
            det_m = tot["det"][0, iq]
            log(f"   [{run}] INJECTED thA={nodes[j]:.3f}' qA={qa}: N={n} detected {nd} ({nd/n:.3f}; map P(D) {det_m:.3f}); "
                f"Y boost>10% {b2:.4f}+-{eb:.4f} (map P(D) {tot['b2'][0,iq,1]/det_m:.4f}, build {pred:.4f}); boost>20% {b22:.4f} "
                f"(map {tot['b2'][0,iq,2]/det_m:.4f}); boosted over q=5 {b3:.4f} (map {tot['b3'][0,iq]/det_m:.4f}); "
                f"lost {lost:.4f} (map {tot['lost'][0,iq]/ndtr(qa-5):.4f}); mean boost of detected {mbd:+.4f} "
                f"(map {tot['mean'][0,iq]/det_m:+.4f}); mean boost of all {boost.mean():+.4f}+-{boost.std()/np.sqrt(n):.4f}; "
                f"linearity max|boost - delta_sky sigma/Y| = {lin:.1e}; self-response |Y_nl/Y - 1| max {np.abs(nl/Y-1).max():.1e}")
log(f"   [{time.time()-t0:.0f}s]")

# ---------------------------------------------------------------- (C) population-level fractions
log("\n(C) population-level fractions (full sky, primaries q_A > 0.5, the build's H on the 12 nodes)")
HEAD = {  # build headline (s06b/s06c logs, 48 nodes, dominance): q>5 rows of RESULTS.md / s08
    "baseline": dict(B2_10=0.0879, B2_20=0.0470, B3=0.0622, lost=0.0491, mb=0.0223, dN=0.0129),
    "goal": dict(B2_10=0.1037, B2_20=0.0565, B3=0.0687, lost=0.0544, mb=0.0266, dN=0.0138),
    "white": dict(B2_10=0.1624, B2_20=0.1052, B3=0.0812, lost=0.0902, mb=0.0338, dN=-0.0162),
    "baseline_sub": dict(B2_10=0.0772, B2_20=0.0391, B3=0.0569, lost=0.0419, mb=0.0187, dN=0.0158),
    "goal_sub": dict(B2_10=0.0894, B2_20=0.0459, B3=0.0619, lost=0.0451, mb=0.0218, dN=0.0178)}
popres = {}
NODEPROBS = {"nodes": nodes, "qnodes": QN, "qcuts": QCUTS, "eps": EPS}
for run in RUNS:
    H = B6[f"{run}_H"]
    Hz = B6[f"{run}_Hz"]
    Pm = {}
    Nb = None
    for j in range(nn):
        C, S1 = HCACHE.get((run, j)) or load_hist(run, j)
        x = xrep(C, S1)
        P_, Nb_ = map_node_probs(C, x, QN, QCUTS)
        for k, v in P_.items():
            Pm.setdefault(k, np.zeros((NB,) + v.shape[1:2] + (nn,) + v.shape[2:]))
            Pm[k][:, :, j] = v / Nb_[:, None, None] if v.ndim == 3 else v / Nb_[:, None, None, None]
        Nb = Nb_
    # combine blocks: total = weighted mean over blocks (equal pixel counts per block)
    w = Nb / Nb.sum()

    def fracs(P, ic, Hm):
        Phi = ndtr(QN[None, :] - QCUTS[ic])
        Ns = (Hm * Phi).sum()
        Nd = (Hm * P["det"][ic]).sum()
        return dict(Ns=Ns, Nd=Nd, dN=Nd / Ns - 1, B2_10=(Hm * P["b2"][ic, :, :, 1]).sum() / Nd,
                    B2_20=(Hm * P["b2"][ic, :, :, 2]).sum() / Nd, B3=(Hm * P["b3"][ic]).sum() / Nd,
                    lost=(Hm * P["lost"][ic]).sum() / Ns, mb=(Hm * P["mean"][ic]).sum() / Nd,
                    B2n_10=(Hm * P["b2n"][ic, :, :, 1]).sum() / Nd)
    Ptot = {k: np.tensordot(w, v, axes=(0, 0)) for k, v in Pm.items()}
    NODEPROBS.update({f"{run}_{k}": v for k, v in Ptot.items()})
    NODEPROBS.update({f"{run}_blk_{k}": v for k, v in Pm.items() if k in ("det", "b2", "b3", "lost", "mean")})
    Ploo = [{k: (np.tensordot(w, v, axes=(0, 0)) - w[b] * v[b]) / (1 - w[b]) for k, v in Pm.items()} for b in range(NB)]
    Pb = {d: {k: B6[f"{run}_{k}"][d] for k in ("det", "b3", "lost", "mean", "b2", "b2n")} for d in (0, 1)}
    log(f"   [{run}]")
    for ic, qc in enumerate(QCUTS):
        fm = fracs(Ptot, ic, H)
        fl = [fracs(l, ic, H) for l in Ploo]
        f0 = fracs(Pb[0], ic, H)
        f1 = fracs(Pb[1], ic, H)
        ks = ("dN", "B2_10", "B2_20", "B3", "lost", "mb", "B2n_10")
        log(f"      q>{qc:.0f}: N_single {fm['Ns']:,.0f}; MAP(no dom): " + " ".join(f"{k} {fm[k]:+.4f}+-{jk([l[k] for l in fl]):.4f}" for k in ks))
        log(f"            BUILD same nodes, no dom : " + " ".join(f"{k} {f0[k]:+.4f}" for k in ks))
        log(f"            BUILD same nodes, dom    : " + " ".join(f"{k} {f1[k]:+.4f}" for k in ks))
        if qc == 5.0:
            log(f"            BUILD headline (48 nodes, dom): " + " ".join(f"{k} {HEAD[run][k]:+.4f}" for k in HEAD[run]))
            popres[run] = (fm, {k: jk([l[k] for l in fl]) for k in ks}, f0, f1)
    # (D) redshift tilt at q > 5
    if run in ("baseline", "goal", "white"):
        zm, ze, z0, z1 = [], [], [], []
        for b in range(len(ZB) - 1):
            fm = fracs(Ptot, 1, Hz[b]); fl = [fracs(l, 1, Hz[b]) for l in Ploo]
            zm.append(fm["dN"]); ze.append(jk([l["dN"] for l in fl]))
            z0.append(fracs(Pb[0], 1, Hz[b])["dN"]); z1.append(fracs(Pb[1], 1, Hz[b])["dN"])
        log(f"      (D) q>5 Delta N/N in z bins {ZB}:")
        log(f"          map (no dom)        : " + " ".join(f"{a*100:+.2f}+-{e*100:.2f}%" for a, e in zip(zm, ze)))
        log(f"          build same nodes, no dom: " + " ".join(f"{a*100:+.2f}%" for a in z0))
        log(f"          build same nodes, dom   : " + " ".join(f"{a*100:+.2f}%" for a in z1))
    log(f"   [{time.time()-t0:.0f}s]")
np.savez(os.path.join(V.OUT, "v07_summary.npz"), popres=np.array([str(popres)]))
np.savez(os.path.join(V.OUT, "v07_nodeprobs.npz"), **NODEPROBS)
log(f"[v07] done {time.time()-t0:.0f}s")
