"""Analyse the independent patches with the claim's metric definitions; compare with claimed values in sigma.
sigma = sqrt(err_mine^2 + err_claim^2) where the claim quotes an error, else err_mine."""
import glob, os, json
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq
from scipy.spatial import cKDTree
import indep_lib as I

files = sorted(glob.glob(f"{I.OUT}/patches/p*.npz"))
NP = len(files); AREA_DEG2 = I.LBOX ** 2 / 3600.0
QC = [4, 5, 6, 7]
SIG = I.templates()[1]
print(f"independent patches: {NP}  ({NP*AREA_DEG2:.0f} deg^2)")
O = [dict(np.load(f)) for f in files]
rng = np.random.default_rng(12345); BOOT = rng.integers(0, NP, (1000, NP))


def merge(x, y, Q, r0=2.0):
    o = np.argsort(-Q); acc = []
    for j in o:
        ok = True
        for a in acc:
            dx = (x[j] - x[a] + I.LBOX / 2) % I.LBOX - I.LBOX / 2; dy = (y[j] - y[a] + I.LBOX / 2) % I.LBOX - I.LBOX / 2
            if dx * dx + dy * dy < r0 * r0:
                ok = False; break
        if ok:
            acc.append(j)
    return np.array(acc, int)


MET = ["n13", "n14", "n0", "single", "single14", "n_real", "cont10r", "cont20r", "neg10r", "nc10r", "boost", "boostr",
       "multi1", "multi2", "multi3", "mP1", "mP1d", "mP2", "mP2d", "mP3", "mP3d", "dup", "nosig_noise", "nosig_conf",
       "spur1", "faint12", "prim1", "h_det", "h_detected", "h_merged", "h_abs", "h_faded", "n_real14", "cont10r14",
       "cat_den", "cat_nb10", "cat_nb20", "cat_net10"]
mi = {m: i for i, m in enumerate(MET)}
A = np.zeros((NP, len(QC), len(MET)))
qiso_det = {qc: [] for qc in QC}
realrows = []         # (patch, qc, k, cP) of real V13 detections for the B2 formula
catrows = []          # (patch, k_own, own, maxnb, q) unselected catalogue-level
allq = []
for p, o in enumerate(O):
    allq.append(o["all_q"])
    for v in ("V13", "V14", "V0"):
        c = o[f"cand_{v}"]
        iy, ix, k, Q, S, N = c[:, 0].astype(int), c[:, 1].astype(int), c[:, 2].astype(int), c[:, 3], c[:, 4], c[:, 5]
        x, y = (ix + 0.5) * I.PIX, (iy + 0.5) * I.PIX
        acc = merge(x, y, Q)
        sk = SIG[k]
        for iq, qc in enumerate(QC):
            aa = acc[Q[acc] >= qc]; aa = aa[np.argsort(-Q[aa])]
            if v == "V0":
                A[p, iq, mi["n0"]] = aa.size; continue
            rec = o[f"dec_{v}"]
            ip = rec[:, 0].astype(int); cp = rec[:, 1]; dp = rec[:, 4]; nc10 = rec[:, 5]; qprim = rec[:, 6]
            seen = set(); dup = np.zeros(aa.size, bool)
            for t, j in enumerate(aa):
                if ip[j] in seen and ip[j] >= 0:
                    dup[t] = True
                seen.add(ip[j])
            fc = (S[aa] - cp[aa]) / S[aa]
            nosig = cp[aa] / sk[aa] < 2.0
            real = ~nosig & ~dup
            if v == "V14":
                A[p, iq, mi["n14"]] = aa.size; A[p, iq, mi["n_real14"]] = real.sum(); A[p, iq, mi["cont10r14"]] = (real & (fc > 0.1)).sum()
                continue
            M = A[p, iq]
            M[mi["n13"]] = aa.size
            M[mi["n_real"]] = real.sum()
            M[mi["cont10r"]] = (real & (fc > 0.1)).sum(); M[mi["cont20r"]] = (real & (fc > 0.2)).sum()
            M[mi["neg10r"]] = (real & (fc < -0.1)).sum(); M[mi["nc10r"]] = (real & (nc10[aa] >= 1)).sum()
            qa = (cp[aa] + N[aa]) / sk[aa]
            M[mi["boost"]] = (qa < qc).sum(); M[mi["boostr"]] = (real & (qa < qc)).sum()
            near = o["near_V13"][aa]
            for rr, key in [(1.0, "multi1"), (2.0, "multi2"), (3.0, "multi3")]:
                M[mi[key]] = (((near[:, :, 0] < rr) & (near[:, :, 1] > qc)).sum(1) >= 2).sum()
            for rr, kn, kd in [(1.0, "mP1", "mP1d"), (2.0, "mP2", "mP2d"), (3.0, "mP3", "mP3d")]:
                okr = (ip[aa] >= 0) & (qprim[aa] > qc) & (dp[aa] < rr)
                M[mi[kd]] = okr.sum(); M[mi[kn]] = (okr & (((near[:, :, 0] < rr) & (near[:, :, 1] > qc)).sum(1) >= 2)).sum()
            M[mi["dup"]] = dup.sum()
            M[mi["nosig_noise"]] = (nosig & ~dup & (N[aa] > S[aa] - cp[aa])).sum()
            M[mi["nosig_conf"]] = (nosig & ~dup & (N[aa] <= S[aa] - cp[aa])).sum()
            M[mi["spur1"]] = (~dup & (cp[aa] / sk[aa] < 1.0)).sum()
            M[mi["faint12"]] = (~dup & (cp[aa] / sk[aa] >= 1.0) & nosig).sum()
            M[mi["prim1"]] = (dp[aa] < 1.0).sum()
            M[mi["single"]] = ndtr(o["all_q"].astype(float) - qc).sum()
            M[mi["single14"]] = ndtr(o["all_q"][o["pop_M"] >= 1e14].astype(float) - qc).sum()
            for j in aa[real]:
                realrows.append([p, qc, k[j], cp[j]])
            # halo side
            hs = o["h_idx"]; qiso = o["h_qiso"]; qpos = o["h_qpos13"]
            dt = qiso >= qc
            prim_det = set(ip[aa].tolist()); own_c = set(ip[Q >= qc].tolist())
            detf = np.array([h in prim_det for h in hs]); candf = np.array([h in own_c for h in hs])
            M[mi["h_det"]] = dt.sum(); M[mi["h_detected"]] = (dt & detf).sum()
            M[mi["h_merged"]] = (dt & ~detf & candf).sum()
            lost = dt & ~detf & ~candf
            M[mi["h_abs"]] = (lost & (qpos >= qc)).sum(); M[mi["h_faded"]] = (lost & (qpos < qc)).sum()
            qiso_det[qc].append(qiso[dt])
            # catalogue-level unselected
            cl = o["catlev"]; s = cl[:, 4] > qc
            M[mi["cat_den"]] = s.sum(); M[mi["cat_nb10"]] = (s & (cl[:, 2] > 0.1 * cl[:, 1])).sum()
            M[mi["cat_nb20"]] = (s & (cl[:, 2] > 0.2 * cl[:, 1])).sum(); M[mi["cat_net10"]] = (s & (cl[:, 3] > 0.1 * cl[:, 1])).sum()
    cl = o["catlev"]
    for r in cl:
        catrows.append([p, *r])
allq = np.sort(np.concatenate(allq).astype(float))
realrows = np.array(realrows); catrows = np.array(catrows)


def ratio(num, den, iq):
    a = A[:, iq, mi[num]]
    if isinstance(den, str):
        b = A[:, iq, mi[den]]; return a.sum() / b.sum(), (a[BOOT].sum(1) / b[BOOT].sum(1)).std(ddof=1)
    return a.sum() / den, (a[BOOT].sum(1) / den).std(ddof=1)


def cmp(label, mine, claim, cerr=None, scale=1.0):
    v, e = mine
    tot = np.hypot(e * scale, cerr) if cerr is not None else e * scale
    z = (v * scale - claim) / tot if tot > 0 else np.nan
    print(f"   {label:60s} mine {v*scale:9.4f} +- {e*scale:7.4f}   claimed {claim:9.4f}" + (f" +- {cerr:.4f}" if cerr is not None else "")
          + f"   ({z:+.2f} sigma)")
    return dict(mine=v * scale, err=e * scale, claim=claim, z=z)


R = {}
print("\n== C8 detection counts per deg^2")
claim_c = {"n13": [3.796, 2.206, 1.535, 1.148], "n13e": [0.0165, 0.0114, 0.0114, 0.0091], "single": [3.205, 2.083, 1.492, 1.123],
           "n14": [3.422, 2.110, 1.496, 1.126], "single14": [3.011, 2.041, 1.473, 1.113], "n0": [0.373, 5.93e-3, 9.3e-5, 0.0]}
for iq, qc in enumerate(QC):
    print(f" q>{qc}")
    R[f"n13_{qc}"] = cmp("V13 map detections /deg^2", ratio("n13", NP * AREA_DEG2, iq), claim_c["n13"][iq], claim_c["n13e"][iq])
    R[f"single_{qc}"] = cmp("single-cluster sum Phi(q-qc) /deg^2", ratio("single", NP * AREA_DEG2, iq), claim_c["single"][iq], 0.0159 if qc == 4 else 0.013)
    R[f"n14_{qc}"] = cmp("V14 map detections /deg^2", ratio("n14", NP * AREA_DEG2, iq), claim_c["n14"][iq], 0.0158)
    R[f"single14_{qc}"] = cmp("single-cluster sum over M>=1e14 /deg^2", ratio("single14", NP * AREA_DEG2, iq), claim_c["single14"][iq], 0.0155)
    a = A[:, iq, mi["n0"]].sum(); r0 = a / (NP * AREA_DEG2)
    print(f"   {'V0 pure-noise detections /deg^2 (Poisson err)':60s} mine {r0:9.4e} +- {np.sqrt(max(a,1))/(NP*AREA_DEG2):.1e}  ({int(a)} events)  claimed {claim_c['n0'][iq]:.3e}")
    v13 = ratio("n13", NP * AREA_DEG2, iq)[0]; sg = ratio("single", NP * AREA_DEG2, iq)[0]
    print(f"   V13/single = {v13/sg:.4f}  (claimed {[1.184,1.059,1.029,1.022][iq]})")

print("\n== C11/C13/C14 detection-side fractions (V13, maxQ, merge 2')  [percent]")
cl = {"cont10r": ([13.78, 14.62, 13.96, 13.33], [0.18, 0.22, 0.26, 0.30]), "cont20r": ([7.98, 8.59, 8.22, 7.76], [0.13, 0.17, 0.19, 0.20]),
      "neg10r": ([14.63, 7.69, 3.72, 1.95], [0.32, 0.24, 0.20, 0.15]), "nc10r": ([21.93, 20.81, 18.56, 17.02], [0.21, 0.25, 0.29, 0.33]),
      "boost": ([8.00, 8.25, 7.86, 8.21], [0.16, 0.19, 0.19, 0.27]), "boostr": ([4.58, 6.77, 7.14, 7.74], [0.12, 0.17, 0.18, 0.26]),
      "multi1": ([0.21, 0.18, 0.16, 0.12], [0.02, 0.03, 0.03, 0.03]), "multi2": ([0.64, 0.55, 0.42, 0.37], [0.04, 0.05, 0.05, 0.05]),
      "multi3": ([1.06, 0.91, 0.70, 0.56], [0.05, 0.06, 0.06, 0.06]), "dup": ([3.02, 1.54, 1.20, 0.93], [0.13, 0.13, 0.12, 0.13]),
      "nosig_noise": ([15.47, 1.70, 0.16, 0.02], [0.21, 0.08, 0.03, 0.01]), "nosig_conf": ([0.39, 0.16, 0.04, 0.0], [0.03, 0.03, 0.02, 0.0]),
      "faint12": ([8.62, 1.57, 0.20, 0.02], [0.15, 0.08, 0.03, 0.01]), "spur1": ([7.24, 0.30, 0.01, 0.0], [0.13, 0.03, 0.01, 0.0]),
      "prim1": ([84.93, 93.13, 94.97, 96.05], [0.18, 0.18, 0.19, 0.20])}
den = {"cont10r": "n_real", "cont20r": "n_real", "neg10r": "n_real", "nc10r": "n_real", "boostr": "n_real"}
for key, (cv, ce) in cl.items():
    print(f" {key}")
    for iq, qc in enumerate(QC):
        R[f"{key}_{qc}"] = cmp(f"q>{qc}", ratio(key, den.get(key, "n13"), iq), cv[iq], ce[iq], scale=100)
print(" V14 REAL f_cont>10%  (claimed q>4 8.59, q>5 9.96+-0.18)")
for iq, qc in enumerate(QC[:2]):
    R[f"cont10r14_{qc}"] = cmp(f"q>{qc}", ratio("cont10r14", "n_real14", iq), [8.59, 9.96][iq], [0.15, 0.18][iq], scale=100)

print("\n== C9 Slivnyak test: P(another q_true>qc within r | primary q_true>qc within r of the peak)  [percent]")
for iq, qc in enumerate(QC):
    Sq = (allq.size - np.searchsorted(allq, qc, side="right")) / (NP * I.LBOX ** 2)
    print(f" q>{qc}: Sigma(q_true>qc) = {Sq*3600:.3f}/deg^2 (claimed {[2.720,1.899,1.398,1.067][iq]})")
    for rr, kn, kd, clv in [(1.0, "mP1", "mP1d", [0.37, 0.257, 0.20, 0.15]), (2.0, "mP2", "mP2d", [1.05, 0.749, 0.522, 0.450]),
                             (3.0, "mP3", "mP3d", [1.74, 1.226, 0.875, 0.684])]:
        v, e = ratio(kn, kd, iq); pred = 1 - np.exp(-Sq * np.pi * rr ** 2)
        print(f"   r={rr:.0f}': mine {100*v:.3f}+-{100*e:.3f}%  Poisson {100*pred:.3f}%  ({(v-pred)/e:+.2f} sigma)   claimed measured {clv[iq]:.3f}%")
        R[f"slivnyak_{qc}_{rr:.0f}"] = dict(mine=100 * v, err=100 * e, poisson=100 * pred, z=(v - pred) / e)

print("\n== C10 detectable halos (q_iso>=qc): fates  [percent]")
clh = {"h_detected": [88.04, 90.28, 91.32, 92.71], "h_merged": [0.03, 0.02, 0.02, 0.01], "h_abs": [7.11, 5.13, 3.87, 2.93],
       "h_faded": [4.82, 4.57, 4.79, 4.35]}
clhe = {"h_detected": [0.23, 0.24, 0.22, 0.28], "h_merged": [0.01] * 4, "h_abs": [0.14, 0.16, 0.17, 0.16], "h_faded": [0.15, 0.16, 0.15, 0.21]}
for key in clh:
    print(f" {key}")
    for iq, qc in enumerate(QC):
        R[f"{key}_{qc}"] = cmp(f"q>{qc}", ratio(key, "h_det", iq), clh[key][iq], clhe[key][iq], scale=100)
print(" theta_b fits")
qs = allq
Sig = lambda q: (qs.size - np.searchsorted(qs, q, side="right")) / (NP * I.LBOX ** 2)
for iq, qc in enumerate(QC):
    a = A[:, iq, mi["h_merged"]] + A[:, iq, mi["h_abs"]]; b = A[:, iq, mi["h_det"]]
    f = a.sum() / b.sum(); fb = a[BOOT].sum(1) / b[BOOT].sum(1)
    Sdet = A[:, iq, mi["n13"]].sum() / (NP * I.LBOX ** 2)
    tb = lambda ff: np.sqrt(-np.log(1 - ff) / (np.pi * Sdet))
    qd = np.concatenate(qiso_det[qc]); Sg = np.array([Sig(max(q, qc)) for q in qd])
    g = lambda t, ff: np.mean(1 - np.exp(-Sg * np.pi * t ** 2)) - ff
    tbn = brentq(g, 1e-3, 60, args=(f,)); tbs = np.array([brentq(g, 1e-3, 60, args=(x,)) for x in fb[:200]])
    print(f"   q>{qc}: f_lost {100*f:.3f}+-{100*fb.std(ddof=1):.3f}%  theta_b simple {tb(f):.3f}+-{tb(fb).std(ddof=1):.3f}'"
          f" (claimed {[4.73,5.24,5.44,5.46][iq]}+-{[0.05,0.09,0.12,0.15][iq]})  brighter-neighbour {tbn:.3f}+-{tbs.std(ddof=1):.3f}'"
          f" (claimed {[7.65,7.84,7.96,7.94][iq]}+-{[0.08,0.13,0.18,0.22][iq]})")
    R[f"thetab_{qc}"] = dict(simple=tb(f), simple_err=tb(fb).std(ddof=1), bn=tbn, bn_err=tbs.std(ddof=1))
json.dump(R, open(f"{I.OUT}/analyze_indep.json", "w"), indent=1, default=float)
np.save(f"{I.OUT}/realrows.npy", realrows); np.save(f"{I.OUT}/catrows.npy", catrows)
np.save(f"{I.OUT}/A_metrics.npy", A); json.dump(MET, open(f"{I.OUT}/A_metrics_names.json", "w"))
