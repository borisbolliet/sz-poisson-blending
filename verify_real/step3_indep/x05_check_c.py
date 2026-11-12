"""x05 (indep lens), check (c) analysis: own noise-only tiles and own injections (x03 outputs) against the build.

(1) spectra of the noise-only tiles (iteration 0) vs the input model (own analytic binned covariance): ratio, pulls;
(2) sigma(theta) / analytic sigma (own lattice sum, exact per-mode covariance): iteration 0, iteration 1 whitened,
    iteration 1 literal binary f_sky;
(3) false detections per 100 deg^2 of the final catalogue, q > 4 / 4.5 / 5, merges none / closest 1, 2, 10' / fof 10';
(4) injections: completeness P(credited q_obs > 5) per q_inj bin and size, q_obs - q_inj, size and position recovery,
    fragments; the same statistics recomputed with the same code from the build's stored injection files
    (data_real/step3/injections) for a like-for-like comparison; the selection fit
    P = Phi((sqrt(q^2 + b) - 5 - mu) / s) per size (own maximum likelihood), saved to out/selection_mine.npz.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
import vlib3 as V

log = V.Tee(os.path.join(V.HERE, "x05_check_c.log"))
t00 = time.time()
G = V.grid()
ODN = os.path.join(V.OUT, "noise"); ODI = os.path.join(V.OUT, "inj")
BINJ = os.path.join(V.BUILD_DATA, "injections")
TH_INJ = np.array([0.3, 0.6, 1.2, 2.4, 4.8, 9.6])
QE = np.array([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5])

# ------------------------------------------------------------------ noise-only
R = [dict(np.load(os.path.join(ODN, f))) for f in sorted(os.listdir(ODN)) if f.endswith(".npz")]
NT = len(R); area = 100.0 * NT
log(f"x05: (c) checks.  Noise-only: {NT} own tiles per level ({area:.0f} deg^2)")
nind = G.cnt / 2.0
lc = (np.arange(G.nbin) + 0.5) * V.BW
for mode in V.MODES:
    Ca = G.analytic_cov_binned(mode)
    C0 = np.array([r[f"{mode}_C0"] for r in R])
    log(f"\n=========== {mode}")
    log("(1) estimated spectra (iteration 0) / input model, mean over tiles [pull mean, pull rms], per l range; cross spectra pulls")
    for lo, hi in ((100, 500), (500, 2000), (2000, 5000), (5000, 10000), (10000, 20000)):
        b = np.flatnonzero((lc >= lo) & (lc < hi)); b = b[np.isin(b, G.bins_used)]
        line = []
        for i in range(6):
            rat = C0[:, b, i, i] / Ca[b, i, i]
            pull = (C0[:, b, i, i] - Ca[b, i, i]) / (Ca[b, i, i] / np.sqrt(nind[b]))
            line.append(f"{V.FREQ[i]:.0f}: {rat.mean():.4f} [{pull.mean():+.2f}, {pull.std():.2f}]")
        cp = []
        for i in range(6):
            for j in range(i + 1, 6):
                v = (Ca[b, i, i] * Ca[b, j, j] + Ca[b, i, j] ** 2) / (2 * nind[b])
                cp.append((C0[:, b, i, j] - Ca[b, i, j]) / np.sqrt(v))
        cp = np.array(cp)
        log(f"  l {lo}-{hi} ({b.size} bins): " + "; ".join(line) + f" | cross: pull mean {cp.mean():+.2f}, rms {cp.std():.2f}")
    sa = V.MMF(G, mode=mode, analytic=True).sigma_am
    log("(2) sigma / analytic sigma: mean [tile-to-tile std] at theta = " + ", ".join(f"{t:.2f}'" for t in V.THETAS[::2]))
    for lab, key in (("iteration 0", "sig0"), ("iteration 1, whitened (finder)", "whitened_sig1"),
                     ("iteration 1, literal binary f_sky", "binary_sig1")):
        a = np.array([r[f"{mode}_{key}"] / sa for r in R])
        log(f"  {lab:34s}: " + ", ".join(f"{m:.4f} [{s:.4f}]" for m, s in zip(a.mean(0)[::2], a.std(0)[::2]))
            + f"   (range of the mean over theta {a.mean(0).min():.4f}-{a.mean(0).max():.4f})")
    log("(3) false detections per 100 deg^2 (final catalogue), q > 4 / 4.5 / 5 (Poisson errors):")
    for meth in ("whitened", "binary"):
        for mn in V.MERGES:
            ns = []
            for qc in (4.0, 4.5, 5.0):
                n = 0
                for r in R:
                    k = f"{mode}_{meth}_it1"
                    n += V.apply_merge(r[k + "_iy"] * V.PIX, r[k + "_ix"] * V.PIX, r[k + "_q"], mn, qc).size
                ns.append(n)
            log(f"  {meth:8s} {mn:9s}: " + " / ".join(f"{100 * n / area:.3f} +- {100 * np.sqrt(max(n, 1)) / area:.3f}" for n in ns) + f"   (counts {ns})")
    n0 = [int(sum((r[f"{mode}_it0_q"] > qc).sum() for r in R)) for qc in (4.0, 4.5, 5.0)]
    log("  iteration 0 (no merge): " + " / ".join(f"{100 * n / area:.3f} +- {100 * np.sqrt(max(n, 1)) / area:.3f}" for n in n0) + f"   (counts {n0})")


# ------------------------------------------------------------------ injections
def per_cluster_build(r):
    """The same per-cluster reduction applied to the build's stored injection files."""
    sig = r["sig"]
    cred = r["d_S_tsz"] / sig[r["d_j"]] >= 2.0
    n = r["q_inj"].size
    qb = np.full(n, -np.inf); jb = np.full(n, -1); db = np.full(n, np.nan); n5 = np.zeros(n, int)
    for d in np.flatnonzero(cred):
        h = r["d_near"][d]
        if r["d_q"][d] > 5:
            n5[h] += 1
        if r["d_q"][d] > qb[h]:
            qb[h] = r["d_q"][d]; jb[h] = r["d_j"][d]; db[h] = r["d_dist"][d]
    return qb, jb, db, n5, int(((~cred) & (r["d_q"] > 5)).sum())


def load_inj(src, mode):
    rows = {th: dict(q=[], qobs=[], j=[], dist=[], n5=[], fq=[], tile=[]) for th in TH_INJ}
    sig = []; nsp = 0; nt = 0
    d = ODI if src == "own" else BINJ
    for f in sorted(os.listdir(d)):
        if not f.startswith(f"inj_{mode}_"):
            continue
        r = dict(np.load(os.path.join(d, f)))
        th = float(r["theta_inj"])
        if src == "own":
            qb, jb, db, n5, sp = r["qbest"], r["jbest"], r["dbest"], r["n5"], int(r["n_spur5"])
        else:
            qb, jb, db, n5, sp = per_cluster_build(r)
        R_ = rows[th]
        R_["q"].append(r["q_inj"]); R_["qobs"].append(qb); R_["j"].append(jb); R_["dist"].append(db); R_["n5"].append(n5)
        R_["fq"].append(r["f_q"]); R_["tile"].append(np.full(r["q_inj"].size, int(r["k"])))
        sig.append(r["sig"] / V.sigma_Y(mode, V.THETAS)); nsp += sp; nt += 1
    for th in TH_INJ:
        rows[th] = {k: np.concatenate(v) for k, v in rows[th].items()}
    return rows, np.array(sig), nsp, nt


def fit_sel(q, det):
    def nll(p):
        b, mu, ls = p
        pr = np.clip(ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)), 1e-12, 1 - 1e-12)
        return -np.sum(det * np.log(pr) + (1 - det) * np.log(1 - pr))
    return minimize(nll, [0.0, 0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9, maxiter=8000)).x


sel_out = dict(theta_inj=TH_INJ, qedges=QE)
for mode in V.MODES:
    log(f"\n=========== injections, {mode}")
    own, sig_o, nsp_o, nt_o = load_inj("own", mode)
    bld, sig_b, nsp_b, nt_b = load_inj("build", mode)
    log(f"  own: {nt_o} tiles, {sum(own[t]['q'].size for t in TH_INJ)} clusters; build: {nt_b} tiles, {sum(bld[t]['q'].size for t in TH_INJ)} clusters")
    log(f"  sigma_final / sigma_Y on injection tiles: own mean {sig_o.mean():.4f} (tile means {sig_o.mean(1).min():.4f}-{sig_o.mean(1).max():.4f}); "
        f"build {sig_b.mean():.4f}")
    log(f"  uncredited q > 5 detections (noise peaks) per 100 deg^2: own {100 * nsp_o / (100 * nt_o):.3f} ({nsp_o}), build {100 * nsp_b / (100 * nt_b):.3f} ({nsp_b})")
    sel_out[mode + "_rinj"] = sig_o.mean(0)
    log("  completeness P(q_obs > 5) per q_inj bin: own / build [difference / combined binomial error], per size; models Phi(q-5), Phi(sqrt(q^2+3)-5)")
    comp = np.zeros((6, QE.size - 1)); pulls = []
    for b in range(QE.size - 1):
        qa_, qb_ = QE[b], QE[b + 1]
        line = []
        for it_, th in enumerate(TH_INJ):
            mo = (own[th]["q"] >= qa_) & (own[th]["q"] < qb_); mb = (bld[th]["q"] >= qa_) & (bld[th]["q"] < qb_)
            po = np.mean(own[th]["qobs"][mo] > 5); pb = np.mean(bld[th]["qobs"][mb] > 5)
            e = np.sqrt(po * (1 - po) / max(mo.sum(), 1) + pb * (1 - pb) / max(mb.sum(), 1))
            comp[it_, b] = po
            z = (po - pb) / e if e > 0 else 0.0
            if 0.02 < pb < 0.98:
                pulls.append(z)
            line.append(f"{po:.3f}/{pb:.3f}[{z:+.1f}]")
        qs = np.linspace(qa_, qb_, 50)
        log(f"  {qa_:3.1f}-{qb_:3.1f}: " + " ".join(line) + f"   {np.mean(ndtr(qs - 5)):.3f} {np.mean(ndtr(np.sqrt(qs ** 2 + 3) - 5)):.3f}")
    pulls = np.array(pulls)
    log(f"  pulls over the {pulls.size} bins with 0.02 < P < 0.98: mean {pulls.mean():+.2f}, rms {pulls.std():.2f}, max |.| {np.abs(pulls).max():.1f}")
    for it_, th in enumerate(TH_INJ):
        mo = (own[th]["q"] >= 5) & (own[th]["q"] < 5.5); mb = (bld[th]["q"] >= 5) & (bld[th]["q"] < 5.5)
        if it_ == 0:
            log("  q_inj 5-5.5 completeness by size, own vs build: ", )
        log(f"     theta {th}': own {np.mean(own[th]['qobs'][mo] > 5):.3f} +- {np.sqrt(np.mean(own[th]['qobs'][mo] > 5) * (1 - np.mean(own[th]['qobs'][mo] > 5)) / mo.sum()):.3f}, "
            f"build {np.mean(bld[th]['qobs'][mb] > 5):.3f}")
    log("  credited q_obs > 5, q_inj in [6, 9.5): mean [std] of q_obs - q_inj, own vs build; forced (true pixel, nearest scale) q - q_inj")
    for th in TH_INJ:
        out = []
        for lab, Rr in (("own", own), ("build", bld)):
            m = (Rr[th]["q"] >= 6) & (Rr[th]["q"] < 9.5) & (Rr[th]["qobs"] > 5)
            dq = Rr[th]["qobs"][m] - Rr[th]["q"][m]
            fm = (Rr[th]["q"] >= 6) & (Rr[th]["q"] < 9.5)
            fq = Rr[th]["fq"][fm] - Rr[th]["q"][fm]
            out.append(f"{lab} {dq.mean():+.3f} [{dq.std():.3f}] (N {m.sum()}), forced {fq.mean():+.3f} [{fq.std():.3f}]")
        log(f"    theta {th}': " + "; ".join(out))
    log("  size / position recovery (credited q_obs > 5): theta ratio median; factor > 2 off; offset median [84%]; extra credited q > 5 per detected cluster")
    for th in TH_INJ:
        for lo, hi in ((5, 7), (7, 10), (10, 31)):
            out = []
            for lab, Rr in (("own", own), ("build", bld)):
                m = (Rr[th]["qobs"] > 5) & (Rr[th]["q"] >= lo) & (Rr[th]["q"] < hi)
                rt = V.THETAS[Rr[th]["j"][m]] / th
                out.append(f"{lab} {np.median(rt):.2f}, {np.mean((rt > 2) | (rt < 0.5)):.3f}, {np.median(Rr[th]['dist'][m]):.2f}' [{np.percentile(Rr[th]['dist'][m], 84):.2f}'], "
                           f"{np.mean(Rr[th]['n5'][m] - 1):.3f}")
            log(f"    theta {th:4}', q_inj {lo}-{hi}: " + " | ".join(out))
    fits = []
    log("  selection fit P = Phi((sqrt(q^2 + b) - 5 - mu) / s), own injections (build's fit in brackets):")
    bf = np.load(os.path.join(V.BUILD_DATA, "selection.npz"))[mode + "_fit"]
    for it_, th in enumerate(TH_INJ):
        p = fit_sel(own[th]["q"], (own[th]["qobs"] > 5).astype(float))
        fits.append(p)
        log(f"    theta {th}': b = {p[0]:+.2f}, mu = {p[1]:+.3f}, s = {np.exp(p[2]):.3f}   [build: {bf[it_][0]:+.2f}, {bf[it_][1]:+.3f}, {np.exp(bf[it_][2]):.3f}]"
            f"   P at q = 4.5 / 5 / 5.5 / 6: " + "/".join(f"{ndtr((np.sqrt(qq * qq + p[0]) - 5 - p[1]) / np.exp(p[2])):.3f}" for qq in (4.5, 5, 5.5, 6))
            + "  [" + "/".join(f"{ndtr((np.sqrt(qq * qq + bf[it_][0]) - 5 - bf[it_][1]) / np.exp(bf[it_][2])):.3f}" for qq in (4.5, 5, 5.5, 6)) + "]")
    sel_out[mode + "_fit"] = np.array(fits)
    sel_out[mode + "_comp"] = comp
    for th in TH_INJ:
        for kk in ("q", "qobs", "tile"):
            sel_out[f"{mode}_{th}_{kk}"] = own[th][kk]
np.savez(os.path.join(V.OUT, "selection_mine.npz"), **sel_out)
log(f"\ndone [{time.time() - t00:.0f} s]")
