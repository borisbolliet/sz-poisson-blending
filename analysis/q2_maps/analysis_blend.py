"""Map-level blending statistics (Q2) from the 108 patches, with patch-bootstrap errors, and comparison with
catalog-level / analytic Poisson estimates.

Definitions (detection d at pixel p, best scale k, Q = (S+N)/sigma_k; S = noise-free filtered signal incl. l=0 removal,
 c_i = Y_i R_k(|x_i-p|; theta_i), primary P = argmax c_i):
  f_cont     = (S - c_P)/S                net fraction of the noise-free recovered Y not from the primary
  f_gross    = sum_{i!=P} c_i / sum_i c_i (no mean removal; all other halos counted positively)
  f_obs      = (S - c_P)/(S + N)         fraction of the measured Y_hat
  boosted    : (c_P + N)/sigma_k < q_cut  (detection needs the other halos)
  no-signal  : c_P/sigma_k < 2  -> 'noise' if N > S - c_P else 'confusion'
  duplicate  : primary already primary of a brighter accepted detection (fragment)
  multiple(r): >=2 halos with q_true > q_cut within r of p (q_true = catalogue mean SNR at own theta500)
Halos: detectable if q_iso >= q_cut (would-be SNR alone, at nearest pixel, max over scales).
  detected : primary of an accepted detection with Q >= q_cut
  merged   : own candidate (primary = this halo) with Q >= q_cut existed but was removed by the merging rule
  unresolved: no own candidate >= q_cut, but an accepted detection with another primary lies within 3'
  faded    : the rest
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ndtr
import q2lib as L

FIG = f"{L.ROOT}/figures/q2_maps"
PIDS = L.pids_from_env(); NP = len(PIDS); AREA = L.LBOX ** 2; AREA_DEG2 = AREA / 3600
QC = [4, 5, 6, 7]
RULES = {"R1": (1.0, 0.0), "R2": (2.0, 0.0), "R3": (2.0, 1.0)}     # r_merge = max(r0, a*theta_k)
RULE_DESC = {"R1": "1'", "R2": "2' (baseline)", "R3": "max(2', theta_k)"}
VARS = ["V13", "V3e13", "V14"]
tab = L.build_tables()
MODE = os.environ.get("DETMODE", "maxQ")
SUF = "" if MODE == "maxQ" else "_ps"
print("DETECTION MODE:", MODE, "(maxQ = peaks of max-over-scales SNR map; ps = peaks of each scale map, merged across scales)")


def merge(xy, th, Q, rule):
    o_ = np.argsort(-Q); acc = []; rej_by = np.full(Q.size, -1)
    for j in o_:
        ok = True
        for a in acc:
            dx = (xy[j] - xy[a] + L.LBOX / 2) % L.LBOX - L.LBOX / 2
            rm = rule(th[a])
            if dx[0] ** 2 + dx[1] ** 2 < rm * rm:
                ok = False; rej_by[j] = a; break
        if ok:
            acc.append(j)
    return np.array(acc, int), rej_by


METRICS = ["n_det", "prim_1", "prim_eq_near", "cont10", "cont20", "gross10", "gross20", "obs10", "obs20", "neg10",
           "nc10", "multi1", "multi2", "multi3", "multiP2", "multiP2_den", "boost", "dup", "nosig_noise", "nosig_conf",
           "boost_real", "clean", "n_real", "cont10r", "cont20r", "neg10r", "nc10r", "boostr", "spur1", "faint12",
           "mP1", "mP1d", "mP3", "mP3d", "faded_lowpos", "single_expect14",
           "h_detectable", "h_detected", "h_merged", "h_unres", "h_faded", "h_gain",
           "cat_cont10", "cat_cont20", "cat_den", "cat_boost", "cat_pos_det", "single_expect"]

acc_all = {(v, r): np.zeros((NP, len(QC), len(METRICS))) for v in VARS for r in RULES}
det_rows = []        # per-detection rows for V13/R2 (for the analytic B2 comparison)
halo_q = []          # q_true of all halos with q_true >= 1 (for densities)
pop = []             # (theta, Y) of all halos for the contaminant population
sig = None
for ip_, p in enumerate(PIDS):
    o = np.load(f"{L.DATA}/patch_out/patch_{p:03d}.npz"); sig = o["sig"]
    dc = dict(np.load(f"{L.DATA}/decomp{SUF}/decomp_{p:03d}.npz"))
    d = L.load_patch(p)
    halo_q.append(d["q"][d["q"] >= 1.0])
    pop.append(np.stack([d["theta500"], d["Y500"]], 1).astype(np.float32))
    hs = dc["h_idx"]; qiso = dc["h_qiso"]; qtrue = d["q"][hs]
    for v in VARS:
        rec, near = dc[f"cand_{v}"], dc[f"near_{v}"]
        iy, ix, k = rec[:, 0].astype(int), rec[:, 1].astype(int), rec[:, 2].astype(int)
        Q, S, N, ip, cp = rec[:, 3], rec[:, 4], rec[:, 5], rec[:, 6].astype(int), rec[:, 7]
        Ssum, nc10, dp, yk = rec[:, 10], rec[:, 11], rec[:, 12], rec[:, 13]
        xy = np.stack([(ix + 0.5) * L.PIX, (iy + 0.5) * L.PIX], 1); th = L.THETA_K[k]; sk = sig[k]
        f_cont = (S - cp) / S; f_gross = (Ssum + yk - cp) / (Ssum + yk); f_obs = (S - cp) / (S + N)
        dnear = near[:, :, 0]; qnear = near[:, :, 1]
        # nearest q>=3 halo within 1' is the primary?
        for r, rule in RULES.items():
            acc = L.merge_greedy(xy[:, 0].copy(), xy[:, 1].copy(), th.astype(np.float64), Q.copy(), rule[0], rule[1], L.LBOX)
            A = acc_all[(v, r)]
            for iq, qc in enumerate(QC):
                a = acc[Q[acc] >= qc]
                a = a[np.argsort(-Q[a])]
                seen = set(); dup = np.zeros(a.size, bool)
                for t, j in enumerate(a):
                    if ip[j] in seen:
                        dup[t] = True
                    seen.add(ip[j])
                aa = a
                M = dict.fromkeys(METRICS, 0.0)
                M["n_det"] = aa.size
                M["prim_1"] = (dp[aa] < 1.0).sum()
                M["cont10"] = (f_cont[aa] > 0.1).sum(); M["cont20"] = (f_cont[aa] > 0.2).sum()
                M["neg10"] = (f_cont[aa] < -0.1).sum()
                M["gross10"] = (f_gross[aa] > 0.1).sum(); M["gross20"] = (f_gross[aa] > 0.2).sum()
                M["obs10"] = (f_obs[aa] > 0.1).sum(); M["obs20"] = (f_obs[aa] > 0.2).sum()
                M["nc10"] = (nc10[aa] >= 1).sum()
                for rr, key in [(1.0, "multi1"), (2.0, "multi2"), (3.0, "multi3")]:
                    M[key] = (((dnear[aa] < rr) & (qnear[aa] > qc)).sum(1) >= 2).sum()
                # Slivnyak test: detections whose primary has q_true>qc and lies within 2'
                okP = (ip[aa] >= 0) & (d["q"][np.maximum(ip[aa], 0)] > qc) & (dp[aa] < 2.0)
                M["multiP2_den"] = okP.sum(); M["multiP2"] = (okP & (((dnear[aa] < 2.0) & (qnear[aa] > qc)).sum(1) >= 2)).sum()
                for rr, kn, kd in [(1.0, "mP1", "mP1d"), (3.0, "mP3", "mP3d")]:
                    okr = (ip[aa] >= 0) & (d["q"][np.maximum(ip[aa], 0)] > qc) & (dp[aa] < rr)
                    M[kd] = okr.sum(); M[kn] = (okr & (((dnear[aa] < rr) & (qnear[aa] > qc)).sum(1) >= 2)).sum()
                qa = (cp[aa] + N[aa]) / sk[aa]
                M["boost"] = (qa < qc).sum()
                nosig = cp[aa] / sk[aa] < 2.0
                M["dup"] = dup.sum()
                M["nosig_noise"] = (nosig & ~dup & (N[aa] > S[aa] - cp[aa])).sum()
                M["nosig_conf"] = (nosig & ~dup & (N[aa] <= S[aa] - cp[aa])).sum()
                M["boost_real"] = (~nosig & ~dup & (qa < qc)).sum()
                M["clean"] = (~nosig & ~dup & (qa >= qc)).sum()
                real = ~nosig & ~dup
                M["n_real"] = real.sum()
                M["cont10r"] = (real & (f_cont[aa] > 0.1)).sum(); M["cont20r"] = (real & (f_cont[aa] > 0.2)).sum()
                M["neg10r"] = (real & (f_cont[aa] < -0.1)).sum(); M["nc10r"] = (real & (nc10[aa] >= 1)).sum()
                M["boostr"] = (real & (qa < qc)).sum()
                M["spur1"] = (~dup & (cp[aa] / sk[aa] < 1.0)).sum()
                M["faint12"] = (~dup & (cp[aa] / sk[aa] >= 1.0) & nosig).sum()
                # halo side
                prim_det = set(ip[aa].tolist())
                own_cand = set(ip[(Q >= qc)].tolist())
                dt = qiso >= qc
                M["h_detectable"] = dt.sum()
                det_flag = np.array([h in prim_det for h in hs])
                cand_flag = np.array([h in own_cand for h in hs])
                M["h_detected"] = (dt & det_flag).sum()
                M["h_merged"] = (dt & ~det_flag & cand_flag).sum()
                lost_nc = np.flatnonzero(dt & ~det_flag & ~cand_flag)
                qp_l = dc[f"h_qpos_{v}"][lost_nc]
                # absorbed: the map at the halo is above threshold but the peak belongs to another (brighter) halo
                M["h_unres"] = (qp_l >= qc).sum()
                # faded: net contamination (incl. l=0 removal) pushes the forced-photometry SNR below q_cut
                M["h_faded"] = (qp_l < qc).sum()
                n3 = 0
                for h in lost_nc:
                    dx = (xy[aa] - np.array([d["x"][hs[h]], d["y"][hs[h]]]) + L.LBOX / 2) % L.LBOX - L.LBOX / 2
                    if aa.size and (np.hypot(dx[:, 0], dx[:, 1]) < 3.0).any():
                        n3 += 1
                M["faded_lowpos"] = n3
                M["h_gain"] = (~dt & det_flag).sum()
                if v == "V13":
                    # catalog-level forced photometry at the true position and own scale (no peak finding):
                    # contamination X_i/Y_i R(delta) and boosting, for halos with q_true > qc
                    s_ = qtrue > qc
                    fc = dc["h_Xnet"][s_] / dc["h_own"][s_]
                    M["cat_den"] = s_.sum(); M["cat_cont10"] = (fc > 0.1).sum(); M["cat_cont20"] = (fc > 0.2).sum()
                    qpos = dc["h_qpos_V13"]
                    M["cat_pos_det"] = (qpos >= qc).sum(); M["cat_boost"] = ((qpos >= qc) & (qiso < qc)).sum()
                    M["single_expect"] = ndtr(d["q"] - qc).sum()      # sum_i P(q_true_i + N(0,1) > qc)
                    M["single_expect14"] = ndtr(d["q"][d["M500"] >= 1e14] - qc).sum()
                A[ip_, iq] = [M[m] for m in METRICS]
                if v == "V13" and r == "R2" and qc == 4:
                    for t_, j in enumerate(aa):
                        det_rows.append([p, k[j], Q[j], S[j], N[j], cp[j], sk[j], dup[t_]])
    if p % 20 == 0:
        print("processed patch", p, flush=True)

det_rows = np.array(det_rows)
halo_q = np.concatenate(halo_q)
rng = np.random.default_rng(42)
BOOT = rng.integers(0, NP, (1000, NP))
mi = {m: i for i, m in enumerate(METRICS)}


def ratio(A, num, den, iq):
    """Ratio of patch sums with bootstrap error over patches."""
    a = A[:, iq, mi[num]]; b = A[:, iq, mi[den]] if isinstance(den, str) else None
    val = a.sum() / (b.sum() if b is not None else den)
    bs = a[BOOT].sum(1) / (b[BOOT].sum(1) if b is not None else den)
    return val, bs.std(ddof=1)


res = {}
print("\n================ detection counts per deg^2 (map) vs single-cluster expectation")
fsky = AREA_DEG2 / (4 * np.pi * (180 / np.pi) ** 2)
bs = json.load(open(f"{L.ROOT}/data/build_summary.json"))
for iq, qc in enumerate(QC):
    A = acc_all[("V13", "R2")]
    ndet = {v: ratio(acc_all[(v, "R2")], "n_det", NP * AREA_DEG2, iq) for v in VARS}
    se = ratio(A, "single_expect", NP * AREA_DEG2, iq)
    an13 = bs["Ndet_expected_floor1e13"][str(qc)] / (4 * np.pi * (180 / np.pi) ** 2)
    an14 = bs["Ndet_expected_floor1e14"][str(qc)] / (4 * np.pi * (180 / np.pi) ** 2)
    nf = json.load(open(f"{L.DATA}/noise_confusion.json"))["noise_false_rate_per_deg2"]["R2"][iq]
    print(f"q>{qc}: V13 {ndet['V13'][0]:.4f}+-{ndet['V13'][1]:.4f}  V3e13 {ndet['V3e13'][0]:.4f}+-{ndet['V3e13'][1]:.4f}"
          f"  V14 {ndet['V14'][0]:.4f}+-{ndet['V14'][1]:.4f} /deg^2 | single-cluster sum_i Phi(q_i-qc) {se[0]:.4f}+-{se[1]:.4f}"
          f" | analytic full-sky: floor1e13 {an13:.4f}, floor1e14 {an14:.4f} | noise-only {nf:.4f}"
          f" | V13/single {ndet['V13'][0]/se[0]:.4f}")
    res[f"counts_q{qc}"] = dict(V13=ndet["V13"], V3e13=ndet["V3e13"], V14=ndet["V14"], single=se, an13=an13, an14=an14, noise=nf)

def pr(lbl, v, r, num, den, scale=100):
    out = []
    for iq, qc in enumerate(QC):
        val, err = ratio(acc_all[(v, r)], num, den, iq)
        out.append((val, err))
    print(f"  {lbl:58s} " + "  ".join(f"q>{qc}: {scale*a:6.2f}+-{scale*b:4.2f}" for qc, (a, b) in zip(QC, out)))
    return out

for v in VARS:
    for r in RULES:
        print(f"\n================ variant {v} (halos painted: {'>=1e13' if v=='V13' else '>=3e13' if v=='V3e13' else '>=1e14'}), merge rule {r} = {RULE_DESC[r]}  [percent]")
        R = {}
        R["N_det_per_patch"] = [ratio(acc_all[(v, r)], "n_det", NP, iq) for iq in range(4)]
        print("  N_det per patch: " + "  ".join(f"{a:.1f}+-{b:.1f}" for a, b in R["N_det_per_patch"]))
        R["prim_1"] = pr("primary (brightest contributor) within 1' of the peak", v, r, "prim_1", "n_det")
        R["cont10"] = pr("B2: f_cont=(S-c_P)/S > 10%", v, r, "cont10", "n_det")
        R["cont20"] = pr("B2: f_cont > 20%", v, r, "cont20", "n_det")
        R["neg10"] = pr("    f_cont < -10% (net deficit: l=0 removal)", v, r, "neg10", "n_det")
        R["gross10"] = pr("B2 gross (no mean removal) > 10%", v, r, "gross10", "n_det")
        R["gross20"] = pr("B2 gross > 20%", v, r, "gross20", "n_det")
        R["obs10"] = pr("(S-c_P)/(S+N) > 10%", v, r, "obs10", "n_det")
        R["obs20"] = pr("(S-c_P)/(S+N) > 20%", v, r, "obs20", "n_det")
        R["nc10"] = pr(">=1 other halo with c_i > 10% c_P", v, r, "nc10", "n_det")
        R["multi1"] = pr("B1: >=2 q_true>q_cut halos within 1'", v, r, "multi1", "n_det")
        R["multi2"] = pr("B1: >=2 q_true>q_cut halos within 2'", v, r, "multi2", "n_det")
        R["multi3"] = pr("B1: >=2 q_true>q_cut halos within 3'", v, r, "multi3", "n_det")
        R["multiP2"] = pr("B1 | primary q>q_cut within 2': another q>q_cut within 2'", v, r, "multiP2", "multiP2_den")
        R["boost"] = pr("B3: boosted, (c_P+N)/sigma < q_cut", v, r, "boost", "n_det")
        R["dup"] = pr("duplicates / fragments", v, r, "dup", "n_det")
        R["nosig_noise"] = pr("no halo >2 sigma, noise-dominated", v, r, "nosig_noise", "n_det")
        R["nosig_conf"] = pr("no halo >2 sigma, confusion-dominated", v, r, "nosig_conf", "n_det")
        R["boost_real"] = pr("real primary (>2 sigma) but boosted over q_cut", v, r, "boost_real", "n_det")
        R["clean"] = pr("clean (real, not boosted, not duplicate)", v, r, "clean", "n_det")
        R["spur1"] = pr("spurious: brightest contributor < 1 sigma (not dup)", v, r, "spur1", "n_det")
        R["faint12"] = pr("faint: brightest contributor 1-2 sigma (not dup)", v, r, "faint12", "n_det")
        R["real"] = pr("REAL detections (c_P >= 2 sigma, not dup) / all", v, r, "n_real", "n_det")
        R["cont10r"] = pr("REAL: f_cont > 10%", v, r, "cont10r", "n_real")
        R["cont20r"] = pr("REAL: f_cont > 20%", v, r, "cont20r", "n_real")
        R["neg10r"] = pr("REAL: f_cont < -10%", v, r, "neg10r", "n_real")
        R["nc10r"] = pr("REAL: >=1 other halo with c_i > 10% c_P", v, r, "nc10r", "n_real")
        R["boostr"] = pr("REAL: boosted", v, r, "boostr", "n_real")
        R["mP1"] = pr("Slivnyak | primary q>q_cut within 1': another within 1'", v, r, "mP1", "mP1d")
        R["mP3"] = pr("Slivnyak | primary q>q_cut within 3': another within 3'", v, r, "mP3", "mP3d")
        R["faded_lowpos"] = pr("lost halos (any class) with a detection within 3'", v, r, "faded_lowpos", "h_detectable")
        R["h_detected"] = pr("halos q_iso>=q_cut: detected", v, r, "h_detected", "h_detectable")
        R["h_merged"] = pr("halos q_iso>=q_cut: lost, own peak removed by merging", v, r, "h_merged", "h_detectable")
        R["h_unres"] = pr("halos q_iso>=q_cut: lost, absorbed (q_pos>=q_cut, no own peak)", v, r, "h_unres", "h_detectable")
        R["h_faded"] = pr("halos q_iso>=q_cut: lost, faded (q_pos<q_cut)", v, r, "h_faded", "h_detectable")
        R["h_gain"] = pr("primaries with q_iso<q_cut (per detectable halo)", v, r, "h_gain", "h_detectable")
        if v == "V13" and r == "R2":
            R["cat_cont10"] = pr("CATALOG forced phot., q_true>q_cut: X/Y_own > 10%", v, r, "cat_cont10", "cat_den")
            R["cat_cont20"] = pr("CATALOG forced phot., q_true>q_cut: X/Y_own > 20%", v, r, "cat_cont20", "cat_den")
            R["cat_boost"] = pr("CATALOG forced phot.: q_pos>=q_cut but q_iso<q_cut", v, r, "cat_boost", "cat_pos_det")
        res[f"{v}_{r}"] = R

# ------------------------------------------------------------- analytic Poisson comparisons (V13, baseline R2)
print("\n================ Poisson-formula comparisons (V13, R2)")
qs = np.sort(halo_q)
def Sigma_q(q):   # surface density of halos with q_true > q  [arcmin^-2]
    return (qs.size - np.searchsorted(qs, q, side="right")) / (NP * AREA)
A = acc_all[("V13", "R2")]
fits = {}
for iq, qc in enumerate(QC):
    Sq = Sigma_q(qc)
    Sdet = A[:, iq, mi["n_det"]].sum() / (NP * AREA)
    line = f"q>{qc}: Sigma(q_true>q_cut)={Sq*3600:.3f}/deg^2  Sigma_det={Sdet*3600:.3f}/deg^2"
    print(line)
    for rr, key in [(1.0, "multi1"), (2.0, "multi2"), (3.0, "multi3")]:
        pred = 1 - np.exp(-Sq * np.pi * rr ** 2)
        val, err = ratio(A, key, "n_det", iq)
        print(f"   B1 multiple within {rr:.0f}': measured {100*val:.3f}+-{100*err:.3f}%  Poisson 1-exp(-Sigma pi r^2) = {100*pred:.3f}%  ({(val-pred)/err:+.2f} sigma)")
    for rr, kn, kd in [(1.0, "mP1", "mP1d"), (2.0, "multiP2", "multiP2_den"), (3.0, "mP3", "mP3d")]:
        pred = 1 - np.exp(-Sq * np.pi * rr ** 2)
        val, err = ratio(A, kn, kd, iq)
        print(f"   Slivnyak test (detections whose primary has q_true>q_cut, within {rr:.0f}'): P(another q_true>q_cut within {rr:.0f}') "
              f"measured {100*val:.3f}+-{100*err:.3f}%  Poisson 1-exp(-Sigma pi r^2) {100*pred:.3f}%  ({(val-pred)/err:+.2f} sigma)")
    vs, es = ratio(A, "spur1", NP * AREA_DEG2, iq)
    nf = json.load(open(f"{L.DATA}/noise_confusion.json"))["noise_false_rate_per_deg2"]["R2"][iq]
    vf, ef = ratio(A, "faint12", NP * AREA_DEG2, iq)
    s14, e14 = ratio(A, "single_expect14", NP * AREA_DEG2, iq)
    print(f"   spurious (c_P<1 sigma) {vs:.4f}+-{es:.4f}/deg^2 vs pure-noise false rate {nf:.4f}/deg^2 (ratio {vs/max(nf,1e-9):.2f});"
          f" faint primaries (1-2 sigma) {vf:.4f}+-{ef:.4f}/deg^2 | single-cluster sum Phi, M>=1e14 halos: {s14:.4f}+-{e14:.4f}/deg^2")
    # effective theta_b for each blending metric: f = 1 - exp(-Sigma_det pi theta_b^2)
    fq = {}
    for key, den in [("h_merged", "h_detectable"), ("cont10", "n_det"), ("cont20", "n_det"), ("boost", "n_det"), ("nc10", "n_det")]:
        a = A[:, iq, mi[key]]; b = A[:, iq, mi[den]]
        fb = a[BOOT].sum(1) / b[BOOT].sum(1)
        tb = lambda f: np.sqrt(-np.log(1 - f) / (np.pi * Sdet))
        val = a.sum() / b.sum()
        if key == "h_merged":
            val = (A[:, iq, mi["h_merged"]] + A[:, iq, mi["h_unres"]]).sum() / b.sum()
            fb = (A[:, iq, mi["h_merged"]] + A[:, iq, mi["h_unres"]])[BOOT].sum(1) / b[BOOT].sum(1)
            key = "lost(merged+absorbed)"
        fq[key] = (tb(val), tb(fb).std(ddof=1), val)
        print(f"   effective theta_b from {key:24s} f={100*val:6.3f}%: theta_b = {tb(val):.3f} +- {tb(fb).std(ddof=1):.3f} arcmin")
    # lost fraction with brighter-neighbour model: f_lost = < 1 - exp(-Sigma(>q_i) pi theta_b^2) > over detectable halos
    fits[qc] = fq
res["theta_b_fits"] = {str(k): v for k, v in fits.items()}

# ---- B1 pair-merge model for the lost fraction: lost if a brighter (q_true) halo within theta_b
print("\n   lost-halo model: f_lost(theta_b) = < 1-exp(-Sigma(q_true > q_i) pi theta_b^2) >_detectable ; solve for theta_b")
from scipy.optimize import brentq
lost_fit = {}
qdet_all = {}
for iq, qc in enumerate(QC):
    qlist = []
    for p in PIDS:
        dc = np.load(f"{L.DATA}/decomp{SUF}/decomp_{p:03d}.npz")
        qq = np.load(f"{L.DATA}/patches_10deg_M1e13.npz")[f"p{p}_q"] if False else None
        qlist.append(dc["h_qiso"][dc["h_qiso"] >= qc])
    qd = np.concatenate(qlist)   # use q_iso as the brightness ranking of the detectable halos
    Sg = np.array([Sigma_q(max(q, qc)) for q in qd])
    a = A[:, iq, mi["h_merged"]] + A[:, iq, mi["h_unres"]]; b = A[:, iq, mi["h_detectable"]]
    val = a.sum() / b.sum(); fb = a[BOOT].sum(1) / b[BOOT].sum(1)
    g = lambda tb, f: np.mean(1 - np.exp(-Sg * np.pi * tb ** 2)) - f
    tbv = brentq(g, 1e-3, 50, args=(val,)); tbs = np.array([brentq(g, 1e-3, 50, args=(f,)) for f in fb[:200]])
    print(f"   q>{qc}: f_lost = {100*val:.3f}+-{100*fb.std(ddof=1):.3f}%  -> theta_b(brighter-neighbour model) = {tbv:.3f} +- {tbs.std(ddof=1):.3f}'")
    lost_fit[qc] = (tbv, tbs.std(ddof=1))
res["lost_fit_brighter_neighbour"] = {str(k): v for k, v in lost_fit.items()}

# ---- B2 single-neighbour Poisson model: P(f_cont > f) = 1 - exp(-lambda), lambda = sum_bins n_b pi r*_b^2
print("\n================ B2 single-neighbour Poisson model vs map (V13, R2)")
from b2model import B2Model
B2M = B2Model(pids=PIDS)
b2 = {}
for f in (0.1, 0.2):
    for iq, qc in enumerate(QC):
        sel = (det_rows[:, 2] >= qc) & (det_rows[:, 5] / det_rows[:, 6] >= 2.0) & (det_rows[:, 7] == 0)
        rows = det_rows[sel]
        lam = B2M.lam(rows[:, 1].astype(int), f * rows[:, 5])
        pred = np.mean(1 - np.exp(-lam))
        key = "cont10" if f == 0.1 else "cont20"
        val, err = ratio(A, key + "r", "n_real", iq)
        valn, errn = ratio(A, "nc10r", "n_real", iq) if f == 0.1 else (np.nan, np.nan)
        print(f"   f>{f:.1f}, q>{qc}: REAL detections: unselected single-neighbour Poisson <1-exp(-lambda)> = {100*pred:.3f}%  | map f_cont>{f:.1f}: {100*val:.3f}+-{100*err:.3f}%"
              f" ({(val-pred)/err:+.2f} sigma)" + (f" | map >=1 neighbour with c_i>0.1 c_P: {100*valn:.3f}+-{100*errn:.3f}% ({(valn-pred)/errn:+.2f} sigma)" if f == 0.1 else "")
              + f" | mean lambda {lam.mean():.4f}")
        b2[f"{f}_{qc}"] = dict(pred=pred, map=val, err=err, map_nc10=valn, err_nc10=errn)
res["B2_single_neighbour"] = b2
json.dump(res, open(f"{L.DATA}/blend_results{SUF}{L.TAG}.json", "w"), indent=1, default=float)

# ------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
R = res["V13_R2"]
for key, lab in [("cont10", "f_cont>10%"), ("cont20", "f_cont>20%"), ("boost", "boosted (B3)"), ("multi2", ">=2 q>q_cut within 2' (B1)"),
                 ("dup", "duplicates"), ("nosig_noise", "no halo>2sig, noise")]:
    v_ = np.array(R[key]); ax[0].errorbar(QC, 100 * v_[:, 0], 100 * v_[:, 1], marker="o", capsize=2, label=lab)
cb = np.array([[res["B2_single_neighbour"][f"0.1_{q}"]["pred"] for q in QC]])
ax[0].plot(QC, 100 * cb[0], "k--", label="single-neighbour Poisson, f>10%")
ax[0].set_yscale("log"); ax[0].set_xlabel("q_cut"); ax[0].set_ylabel("% of detections"); ax[0].legend(fontsize=7)
ax[0].set_title("V13 (halos>=1e13), merge 2'")
for v, mk in zip(VARS, "os^"):
    vv = np.array(res[f"{v}_R2"]["cont10"]); ax[1].errorbar(QC, 100 * vv[:, 0], 100 * vv[:, 1], marker=mk, capsize=2, label=f"{v}: f_cont>10%")
ax[1].set_xlabel("q_cut"); ax[1].set_ylabel("% of detections"); ax[1].legend(fontsize=7); ax[1].set_title("contamination vs painted mass floor")
for r in RULES:
    vv = np.array(res[f"V13_{r}"]["h_merged"]) + 0; uu = np.array(res[f"V13_{r}"]["h_unres"])
    ax[2].errorbar(QC, 100 * (vv[:, 0] + uu[:, 0]), 100 * np.hypot(vv[:, 1], uu[:, 1]), marker="o", capsize=2, label=f"lost (merged+absorbed), merge {RULE_DESC[r]}")
ax[2].set_xlabel("q_cut"); ax[2].set_ylabel("% of detectable halos lost"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/blending_fractions{SUF}{L.TAG}.png", dpi=130)
print("saved", f"{FIG}/blending_fractions{SUF}{L.TAG}.png")
np.save(f"{L.DATA}/blend_acc{SUF}{L.TAG}.npy", np.stack([acc_all[(v, r)] for v in VARS for r in RULES]))   # REPAIR: per-patch metric sums

