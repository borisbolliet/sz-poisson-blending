"""Blending metrics from our own independent map pipeline (v03), patch-bootstrap errors."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.special import ndtr

D = os.path.join(os.path.dirname(__file__), sys.argv[1] if len(sys.argv) > 1 else "maps_out")
files = sorted(glob.glob(f"{D}/patch_*.npz"))
NP = len(files); L = 600.0; PIX = 0.25; AREA_DEG2 = 100.0
TK = np.geomspace(0.5, 10.0, 12)
QC = [4, 5, 6, 7]
print(f"{NP} patches")


def merge(x, y, Q, r0):
    o = np.argsort(-Q); acc = []
    for j in o:
        ok = True
        for a in acc:
            dx = (x[j] - x[a] + L / 2) % L - L / 2; dy = (y[j] - y[a] + L / 2) % L - L / 2
            if dx * dx + dy * dy < r0 * r0:
                ok = False; break
        if ok:
            acc.append(j)
    return np.array(acc, int)


keys = ["n_det", "single", "noise_det", "boost", "dup", "n_real", "cont10r", "cont20r", "neg10r", "nc10r", "multi1", "multi2",
        "multi3", "sl2", "sl2d", "sl1", "sl1d", "sl3", "sl3d", "nosig_noise", "nosig_conf", "spur1", "faint12", "boostr",
        "h_dt", "h_det", "h_merged", "h_abs", "h_faded", "n_halo_q", "lam", "nh", "nh13", "nq5"]
A = np.zeros((NP, 4, len(keys))); ki = {k: i for i, k in enumerate(keys)}
var_noise = []; sanity = []; ybars = []
for p, fn in enumerate(files):
    d = np.load(fn)
    sig = d["sig"]; var_noise.append(d["var_noise"] / sig ** 2); sanity.append(d["sanity"]); ybars.append(float(d["ybar"]))
    x = d["c_ix"] * PIX; y = d["c_iy"] * PIX; Q = d["c_Q"]; S = d["c_S"]; Nn = d["c_N"]; k = d["c_k"]
    ip = d["c_ip"]; cp = d["c_cp"]; dp = d["c_dp"]; nq = d["c_nq"]; sk = sig[k]
    acc = merge(x, y, Q, 2.0)
    nc_ = d["noise_cand"]; nacc = merge(nc_[:, 0], nc_[:, 1], nc_[:, 2], 2.0)
    qh = d["q"]; hs = d["hs"]; qiso = d["q_iso"]; qpos = d["q_pos"]
    f_cont = (S - cp) / S
    for iq, qc in enumerate(QC):
        a = acc[Q[acc] >= qc]; a = a[np.argsort(-Q[a])]
        seen = set(); dup = np.zeros(a.size, bool)
        for t, j in enumerate(a):
            if ip[j] in seen:
                dup[t] = True
            seen.add(ip[j])
        M = dict.fromkeys(keys, 0.0)
        M["n_det"] = a.size
        M["single"] = ndtr(qh - qc).sum()
        M["noise_det"] = (nc_[nacc, 2] >= qc).sum()
        qa = (cp[a] + Nn[a]) / sk[a]
        M["boost"] = (qa < qc).sum()
        nosig = cp[a] / sk[a] < 2
        real = ~nosig & ~dup
        M["dup"] = dup.sum(); M["n_real"] = real.sum()
        M["cont10r"] = (real & (f_cont[a] > 0.1)).sum(); M["cont20r"] = (real & (f_cont[a] > 0.2)).sum()
        M["neg10r"] = (real & (f_cont[a] < -0.1)).sum(); M["nc10r"] = (real & (d["c_nc10"][a] >= 1)).sum()
        M["boostr"] = (real & (qa < qc)).sum()
        for b_, (kn, kd, rr) in enumerate([("sl1", "sl1d", 1.0), ("sl2", "sl2d", 2.0), ("sl3", "sl3d", 3.0)]):
            M[f"multi{b_+1}"] = (nq[a, iq, b_] >= 2).sum()
            okP = (ip[a] >= 0) & (qh[np.maximum(ip[a], 0)] > qc) & (dp[a] < rr)
            M[kd] = okP.sum(); M[kn] = (okP & (nq[a, iq, b_] >= 2)).sum()
        M["nosig_noise"] = (nosig & ~dup & (Nn[a] > S[a] - cp[a])).sum()
        M["nosig_conf"] = (nosig & ~dup & (Nn[a] <= S[a] - cp[a])).sum()
        M["spur1"] = (~dup & (cp[a] / sk[a] < 1)).sum(); M["faint12"] = (~dup & (cp[a] / sk[a] >= 1) & nosig).sum()
        prim = set(ip[a].tolist()); own = set(ip[Q >= qc].tolist())
        dt = qiso >= qc
        detf = np.array([h in prim for h in hs]); candf = np.array([h in own for h in hs])
        M["h_dt"] = dt.sum(); M["h_det"] = (dt & detf).sum(); M["h_merged"] = (dt & ~detf & candf).sum()
        rest = dt & ~detf & ~candf
        M["h_abs"] = (rest & (qpos >= qc)).sum(); M["h_faded"] = (rest & (qpos < qc)).sum()
        M["nq5"] = (qh > qc).sum(); M["nh"] = float(d["nh"]); M["lam"] = float(d["lam"])
        A[p, iq] = [M[kk] for kk in keys]

rng = np.random.default_rng(1)
BOOT = rng.integers(0, NP, (2000, NP))


def R(num, den, iq):
    a = A[:, iq, ki[num]]
    if isinstance(den, str):
        b = A[:, iq, ki[den]]; v = a.sum() / b.sum(); bs = a[BOOT].sum(1) / b[BOOT].sum(1)
    else:
        v = a.sum() / den; bs = a[BOOT].sum(1) / den
    return v, bs.std(ddof=1)


def line(lbl, num, den, sc=100.0, fmt="{:.2f}"):
    out = [R(num, den, iq) for iq in range(4)]
    print(f"{lbl:46s} " + "  ".join(f"q>{qc}: " + fmt.format(sc * v) + "+-" + fmt.format(sc * e) for qc, (v, e) in zip(QC, out)))
    return out


nh = A[:, 0, ki["nh"]]
print(f"halos/patch: mean {nh.mean():.1f}  expected {A[0,0,ki['lam']]:.1f}  z={(nh.mean()-A[0,0,ki['lam']])/np.sqrt(A[0,0,ki['lam']]/NP):+.2f}")
print(f"patch ybar mean {np.mean(ybars):.4e} (patch 0 {ybars[0]:.4e})")
vn = np.array(var_noise)
print("Var(noise)/sigma^2 per scale: " + " ".join(f"{m:.5f}+-{e:.5f}" for m, e in zip(vn.mean(0), vn.std(0, ddof=1) / np.sqrt(NP))))
san = np.concatenate(sanity); dd = san[:, 0] - san[:, 1]
print(f"map vs truth-sum decomposition (brightest 30/patch): mean {dd.mean():+.2e} rms {dd.std():.2e} max {np.abs(dd).max():.2e} sigma")
den = NP * AREA_DEG2
print("\n--- counts per deg^2")
line("map detections", "n_det", den, 1.0, "{:.4f}")
line("single-cluster sum Phi(q_i - qc)", "single", den, 1.0, "{:.4f}")
line("pure-noise (multi-scale, merged 2')", "noise_det", den, 1.0, "{:.5f}")
line("Sigma(q_true > qc)", "nq5", den, 1.0, "{:.4f}")
for iq in range(4):
    v = A[:, iq, ki["n_det"]].sum() / A[:, iq, ki["single"]].sum()
    bs = A[BOOT, iq, ki["n_det"]].sum(1) / A[BOOT, iq, ki["single"]].sum(1)
    print(f"   q>{QC[iq]}: map/single = {v:.4f} +- {bs.std():.4f}")
print("\n--- detection fractions (%)")
line("boosted (all detections)", "boost", "n_det")
line("duplicates", "dup", "n_det")
line("no halo>2sig, noise-dominated", "nosig_noise", "n_det")
line("no halo>2sig, confusion-dominated", "nosig_conf", "n_det")
line("real: f_cont>10%", "cont10r", "n_real")
line("real: f_cont>20%", "cont20r", "n_real")
line("real: f_cont<-10%", "neg10r", "n_real")
line("real: >=1 other c_i>0.1 c_P", "nc10r", "n_real")
line("real: boosted", "boostr", "n_real")
line("B1 >=2 q_true>qc within 1'", "multi1", "n_det")
line("B1 >=2 q_true>qc within 2'", "multi2", "n_det")
line("B1 >=2 q_true>qc within 3'", "multi3", "n_det")
s1 = line("Slivnyak 1' (primary q>qc, dp<1')", "sl1", "sl1d")
s2 = line("Slivnyak 2'", "sl2", "sl2d")
s3 = line("Slivnyak 3'", "sl3", "sl3d")
for iq, qc in enumerate(QC):
    Sig = A[:, iq, ki["nq5"]].sum() / den
    for rr, s in [(1, s1), (2, s2), (3, s3)]:
        pp = 1 - np.exp(-Sig * np.pi * (rr / 60) ** 2)
        print(f"   q>{qc} r={rr}': measured {100*s[iq][0]:.3f}+-{100*s[iq][1]:.3f}%  Poisson {100*pp:.3f}%  ({(s[iq][0]-pp)/s[iq][1]:+.2f} sig)")
print("spurious c_P<1 sigma per deg^2:"); line("", "spur1", den, 1.0, "{:.4f}")
print("faint primaries 1-2 sigma per deg^2:"); line("", "faint12", den, 1.0, "{:.4f}")
print("\n--- detectable halos (q_iso >= qc) (%)")
line("detected", "h_det", "h_dt")
line("merged away", "h_merged", "h_dt", 100, "{:.3f}")
ab = line("absorbed", "h_abs", "h_dt")
line("faded", "h_faded", "h_dt")
for iq, qc in enumerate(QC):
    f = (A[:, iq, ki["h_abs"]].sum() + A[:, iq, ki["h_merged"]].sum()) / A[:, iq, ki["h_dt"]].sum()
    Sd = A[:, iq, ki["n_det"]].sum() / (den * 3600)
    tb = np.sqrt(-np.log(1 - f) / (np.pi * Sd))
    fb = (A[BOOT, iq, ki["h_abs"]].sum(1) + A[BOOT, iq, ki["h_merged"]].sum(1)) / A[BOOT, iq, ki["h_dt"]].sum(1)
    Sb = A[BOOT, iq, ki["n_det"]].sum(1) / (den * 3600)
    tbb = np.sqrt(-np.log(1 - fb) / (np.pi * Sb))
    print(f"   q>{qc}: lost (merged+absorbed) {100*f:.2f}%  theta_b(simple) = {tb:.2f} +- {tbb.std():.2f}'")
