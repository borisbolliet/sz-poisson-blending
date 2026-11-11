"""v08: population-level fractions with the REALISED primaries of the v05 map patches (regenerated exactly from
their seeds; own sampler) instead of the build's analytic primary population H, plus single-cluster counts.

  * N_single(q > q_cut) = sum over realised halos of Phi(q_A - q_cut), scaled to the full sky (400 patches =
    40,000 deg^2), vs the build's expected_detections (claim 'detections').
  * fractions (no dominance) = realised primaries (CIC in ln theta on the 12 nodes, linear in q on the build's
    q grid) x node probabilities from (i) the map P(D) (v07), (ii) the build's P(D) at the same nodes, no dominance,
    (iii) the same with dominance (v06); jackknife over 20 blocks of 20 patches (population noise).
  * redshift tilt Delta N/N in z bins with the realised primaries.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V                                    # noqa: E402
import numpy as np                                  # noqa: E402
from scipy.special import ndtr                      # noqa: E402
from szsim import instrument as ins, signal as sg   # noqa: E402

log = V.Tee(os.path.join(V.HERE, "v08_realprimaries.log"))
t0 = time.time()
mom = np.load(os.path.join(V.OUT, "v05_moments.npz"))
pinfo = mom["pinfo"]
NP = pinfo.shape[0]
NPB = np.load(os.path.join(V.OUT, "v07_nodeprobs.npz"))
B6 = np.load(os.path.join(V.OUT, "v06_pd_builder_nodes.npz"))
nodes, QN, QCUTS, EPS, ZB = NPB["nodes"], NPB["qnodes"], NPB["qcuts"], NPB["eps"], B6["zb"]
NN, NQ = len(nodes), len(QN)
lth0, dlth = np.log(nodes[0]), np.log(nodes[1] / nodes[0])
S = V.PatchSampler(600.0)
scale = 4 * np.pi / (NP * (600.0 * V.ARCMIN) ** 2)
sig = {"white": sg.SZModel().sigma_Y, "baseline": ins.MMFModel(ins.Instrument.so_lat("baseline")).sigma_Y,
       "goal": ins.MMFModel(ins.Instrument.so_lat("goal")).sigma_Y}
EXPECT = {"white": [132124.8, 85912.3, 61629.5, 46574.5], "baseline": [133127.1, 81704.5, 55657.0, 40062.8],
          "goal": [203772.7, 128166.7, 88809.8, 64881.7]}
NBLK = 20
bsz = NP // NBLK
Hr = {m: np.zeros((NBLK, len(ZB) - 1, NN, NQ)) for m in sig}
Ns = {m: np.zeros((NBLK, len(QCUTS))) for m in sig}
Ns_z = {m: np.zeros((NBLK, len(ZB) - 1)) for m in sig}
log(f"[v08] regenerating {NP} patch catalogues; full-sky scale {scale:.5f}")
for p in range(NP):
    rng = np.random.default_rng([2026, 11, 11, p])
    h = S.draw(rng)
    assert h["x"].size == pinfo[p, 1], (p, h["x"].size, pinfo[p, 1])
    b = p // bsz
    zb = np.clip(np.searchsorted(ZB, h["z"], side="right") - 1, 0, len(ZB) - 2)
    for m, f in sig.items():
        q = h["Y"] / f(h["th"])
        for ic, qc in enumerate(QCUTS):
            Ns[m][b, ic] += ndtr(q - qc).sum() * scale
        Ns_z[m][b] += np.bincount(zb, weights=ndtr(q - 5.0), minlength=len(ZB) - 1) * scale
        s = q > 0.5
        qq, tt, zz = q[s], h["th"][s], zb[s]
        fa = np.clip((np.log(tt) - lth0) / dlth, 0, NN - 1.000001)
        ia = fa.astype(int); wa = fa - ia
        k = np.clip(np.searchsorted(QN, qq) - 1, 0, NQ - 2)
        wq = np.clip((qq - QN[k]) / (QN[k + 1] - QN[k]), 0, 1)
        for da, wA in ((0, 1 - wa), (1, wa)):
            for dq, wQ in ((0, 1 - wq), (1, wq)):
                np.add.at(Hr[m][b], (zz, ia + da, k + dq), wA * wQ * scale)
    if p % 100 == 0:
        log(f"   patch {p} [{time.time()-t0:.0f}s]")


def jk(v):
    v = np.asarray(v); n = len(v)
    return np.sqrt((n - 1) / n * ((v - v.mean(0)) ** 2).sum(0))


log("\n(1) single-cluster detections, full sky (realised primaries of 40,000 deg^2 scaled by 1.031; noise marginalised)")
for m in sig:
    tot = Ns[m].sum(0)
    e = jk([(tot - Ns[m][b]) * NBLK / (NBLK - 1) for b in range(NBLK)])
    log(f"   [{m:8s}] " + "  ".join(f"q>{qc:.0f}: {t:9,.0f} +- {ee:5,.0f} (build {x:9,.1f}; {(t-x)/ee:+.2f} sig)"
                                 for qc, t, ee, x in zip(QCUTS, tot, e, EXPECT[m])))


def fr(P, ic, H):
    Phi = ndtr(QN[None, :] - QCUTS[ic])
    ns = (H * Phi).sum(); nd = (H * P["det"][ic]).sum()
    return dict(dN=nd / ns - 1, B2_10=(H * P["b2"][ic, :, :, 1]).sum() / nd, B2_20=(H * P["b2"][ic, :, :, 2]).sum() / nd,
                B3=(H * P["b3"][ic]).sum() / nd, lost=(H * P["lost"][ic]).sum() / ns, mb=(H * P["mean"][ic]).sum() / nd,
                Ns=ns, Nd=nd)


log("\n(2) fractions with the realised primaries (jackknife over 20 blocks of 20 patches = population noise)")
for run in ("baseline", "goal", "white", "baseline_sub", "goal_sub"):
    m = run.replace("_sub", "")
    Pm = {k: NPB[f"{run}_{k}"] for k in ("det", "b2", "b3", "lost", "mean")}
    Pb = {d: {k: B6[f"{run}_{k}"][d] for k in ("det", "b2", "b3", "lost", "mean")} for d in (0, 1)}
    Hall = Hr[m].sum((0, 1))
    Hfull = B6[f"{run}_H"]
    log(f"   [{run}] realised primaries q_A > 0.5: {Hall.sum():,.0f} full-sky equivalent (build H: {Hfull.sum():,.0f})")
    for ic, qc in enumerate(QCUTS):
        f = fr(Pm, ic, Hall)
        loo = [fr(Pm, ic, Hall - Hr[m][b].sum(0)) for b in range(NBLK)]
        f0, f1 = fr(Pb[0], ic, Hall), fr(Pb[1], ic, Hall)
        fh = fr(Pm, ic, Hfull)
        ks = ("dN", "B2_10", "B2_20", "B3", "lost", "mb")
        log(f"      q>{qc:.0f}: realised H x MAP P(D)      : " + " ".join(f"{k} {f[k]:+.4f}+-{jk([l[k] for l in loo]):.4f}" for k in ks))
        log(f"            realised H x BUILD no dom  : " + " ".join(f"{k} {f0[k]:+.4f}" for k in ks))
        log(f"            realised H x BUILD dom     : " + " ".join(f"{k} {f1[k]:+.4f}" for k in ks))
        log(f"            build H    x MAP P(D)      : " + " ".join(f"{k} {fh[k]:+.4f}" for k in ks))
    if run in ("baseline", "goal", "white"):
        zm, ze, z0, z1 = [], [], [], []
        for zb in range(len(ZB) - 1):
            Hz = Hr[m][:, zb].sum(0)
            fz = fr(Pm, 1, Hz)
            zm.append(fz["dN"]); ze.append(jk([fr(Pm, 1, Hz - Hr[m][b, zb])["dN"] for b in range(NBLK)]))
            z0.append(fr(Pb[0], 1, Hz)["dN"]); z1.append(fr(Pb[1], 1, Hz)["dN"])
        log(f"      q>5 Delta N/N in z bins {list(ZB)} (realised primaries):")
        log(f"          x MAP P(D) (no dom): " + " ".join(f"{a*100:+.2f}+-{e*100:.2f}%" for a, e in zip(zm, ze)))
        log(f"          x BUILD no dom     : " + " ".join(f"{a*100:+.2f}%" for a in z0))
        log(f"          x BUILD dom        : " + " ".join(f"{a*100:+.2f}%" for a in z1))
        nz = Ns_z[m].sum(0)
        log(f"          N_single per z bin (q>5): " + " ".join(f"{v:,.0f}" for v in nz))
np.savez(os.path.join(V.OUT, "v08_Hreal.npz"), **{f"Hr_{m}": Hr[m] for m in Hr}, **{f"Ns_{m}": Ns[m] for m in Ns},
         nodes=nodes, qnodes=QN, zb=ZB, scale=scale)
log(f"[v08] done {time.time()-t0:.0f}s")
