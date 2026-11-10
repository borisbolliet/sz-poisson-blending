"""Per-primary comparison of the verifier's neighbour sums (own kernel + own search) with the analysis'
stored sums (data/blending/mc_sky{1,2,3}.npz) for the same catalogue halos; and the q_cut=4 'lost' /
N_det_b/N_single computed from each set with the same mu (verifier exact formula)."""
import numpy as np
import v01_kernel as K
YBAR = np.array([8.97647742e-08, 1.20306859e-07, 1.48898975e-07])
d0 = np.load("v01_kernel.npz"); TH, SIG = d0["TH"], d0["SIG"]
sig = lambda t: np.exp(np.interp(np.log(t), np.log(TH), np.log(SIG)))
A = np.load("/Users/boris/GitHub/sz-poisson-blending/data/blending/analytic.npz")
tot = {"mine": [], "theirs": []}
for s in (1, 2, 3):
    me = np.load(f"v03_sky{s}.npz"); th_ = np.load(f"/Users/boris/GitHub/sz-poisson-blending/data/blending/mc_sky{s}.npz")
    assert np.array_equal(me["p_idx"], th_["p_idx"]), "primary sets differ"
    Dm, Dt = me["D"], th_["D"]
    r = Dm.sum(1) / Dt.sum(1)
    print(f"sky {s}: n={len(r)}  D_mine/D_theirs: median {np.median(r):.5f}, mean {np.mean(r):.5f}, "
          f"1-99% [{np.percentile(r,1):.4f},{np.percentile(r,99):.4f}];  per class sum ratio {np.round(Dm.sum(0)/Dt.sum(0),5)}; "
          f"xmax rel diff median {np.median(np.abs(me['XM'].max(1)/th_['xmax'].max(1)-1)):.2e}")
    thA = me["p_th"]; sA = sig(thA)
    mu_mine = (YBAR[None, :] * sA[:, None] * K.RATIO ** 2 / K.NW).sum(1)
    mu_theirs = np.column_stack([np.interp(np.log(thA), np.log(A["th"]), A["mu_exact"][:, c]) for c in range(3)]).sum(1)
    for nm, D, XM in (("mine", Dm, me["XM"]), ("theirs", Dt, th_["xmax"])):
        tot[nm].append((me["p_q"].astype(float), me["p_qo"].astype(float), D.sum(1) / sA, XM.max(1) / sA, mu_mine, mu_theirs))
print("mu_theirs/mu_mine (primary-weighted):", np.mean(np.concatenate([t[5] for t in tot['mine']])) / np.mean(np.concatenate([t[4] for t in tot['mine']])))
for nm in ("mine", "theirs"):
    q, qo, dl, xm, mm, mt = [np.concatenate(x) for x in zip(*tot[nm])]
    for mun, mu in (("mu_mine", mm), ("mu_theirs", mt)):
        out = []
        for qc in (4, 5):
            dp = dl - mu; dom = q > xm; single = qo > qc; det = (qo + dp > qc) & dom
            out.append(f"q_cut={qc}: lost {(single & dom & (qo+dp<qc)).sum()/single.sum():.4f} ratio {det.sum()/single.sum():.4f} "
                       f"B2neg0.1 {(det & (dp/q<-0.1)).sum()/det.sum():.4f} B2_0.2 {(det & (dp/q>0.2)).sum()/det.sum():.4f}")
        print(f"sums={nm:6s} {mun:9s}: " + " | ".join(out))
