"""Catalog-level blending (no maps, no peak finding): every halo with q_true >= 1 in the 108 patches is 'observed' with
the matched filter of the scale k_own nearest to its theta500, centred on its TRUE position.
   own_i  = Y_i R_k(0; theta_i);   c_j = Y_j R_k(r_ij; theta_j) for all other halos j (M >= 1e13)
   net_i  = sum_j c_j - ybar*kappa_k  (l=0 removed, as in the maps);   maxnb_i = max_j c_j
   N_i    = the map noise of that patch at the halo's nearest pixel, scale k_own (same realisation as the maps)
(i)  UNSELECTED population q_true > q_cut: P(maxnb > f own) vs the single-neighbour Poisson formula 1-exp(-lambda)
     (exact by Slivnyak if the population histogram is right), and P(net > f own).
(ii) SNR-SELECTED catalogue: q_cat = (own + net + N)/sigma_k >= q_cut and own >= maxnb (brightest-contributor rule):
     fraction with net > f own, maxnb > f own, boosted ((own+N)/sigma < q_cut)  -> compare with the map detections.
"""
import os, json
import numpy as np
from scipy.special import ndtr
import q2lib as L
from b2model import B2Model

PIDS = L.pids_from_env(); NP = len(PIDS); QC = [4, 5, 6, 7]
tab = L.build_tables()
rows = []
RCAP = float(os.environ.get('RCAP', 'inf'))   # REPAIR: optional cap on neighbour distance [arcmin]
for ip_, p in enumerate(PIDS):
    o = np.load(f"{L.DATA}/patch_out/patch_{p:03d}.npz"); sig = o["sig"]; ybar = o["means"][:3].sum()
    dc = np.load(f"{L.DATA}/decomp/decomp_{p:03d}.npz")
    d = L.load_patch(p); tr = L.Truth(d, tab)
    hs = dc["h_idx"]; kown = dc["h_kown"]; Nown = dc["h_Nown"]
    for j, h in enumerate(hs):
        k = kown[j]
        I, C = tr.contributions(d["x"][h], d["y"][h], k)
        me = I == h
        if np.isfinite(RCAP):
            dxx = (d['x'][I] - d['x'][h] + L.LBOX / 2) % L.LBOX - L.LBOX / 2; dyy = (d['y'][I] - d['y'][h] + L.LBOX / 2) % L.LBOX - L.LBOX / 2
            keep = me | (np.hypot(dxx, dyy) <= RCAP); I, C, me = I[keep], C[keep], me[keep]
        own = C[me].sum(); oth = C[~me]
        rows.append([ip_, h, k, d["q"][h], d["M500"][h], own, oth.sum() - ybar * L.kappa_mean(sig[k]),
                     oth.max() if oth.size else 0.0, Nown[j], sig[k]])
    if p % 20 == 0:
        print("patch", p, flush=True)
R = np.array(rows)
np.save(f"{L.DATA}/catalog_level_rows{L.TAG}.npy", R)
k, qt, own, net, mx, N, sk = R[:, 2].astype(int), R[:, 3], R[:, 5], R[:, 6], R[:, 7], R[:, 8], R[:, 9]
pid = R[:, 0].astype(int)
rng = np.random.default_rng(3); BOOT = rng.integers(0, NP, (500, NP))
def frac(num, den):
    a = np.bincount(pid, num.astype(float), NP); b = np.bincount(pid, den.astype(float), NP)
    return a.sum() / b.sum(), (a[BOOT].sum(1) / b[BOOT].sum(1)).std(ddof=1)

B2 = B2Model(pids=PIDS)
out = {}
print("\n(i) UNSELECTED halos with q_true > q_cut, filter at own scale, true position")
for qc in QC:
    s = qt > qc
    for f in (0.1, 0.2):
        lam = B2.lam(k[s], f * own[s])
        pred = np.mean(1 - np.exp(-lam))
        v, e = frac(s & (mx > f * own), s)
        vn, en = frac(s & (net > f * own), s)
        print(f"  q>{qc} f={f}: P(single neighbour > f own) measured {100*v:.3f}+-{100*e:.3f}%  Poisson 1-exp(-lambda) {100*pred:.3f}%"
              f" ({(v-pred)/e:+.2f} sigma) | P(net > f own) {100*vn:.3f}+-{100*en:.3f}% | mean lambda {lam.mean():.4f}")
        out[f"unsel_q{qc}_f{f}"] = dict(single=v, err=e, poisson=pred, net=vn, net_err=en)
print("\n(ii) SNR-SELECTED catalogue-level detections: (own+net+N)/sigma >= q_cut and own >= max neighbour")
qcat = (own + net + N) / sk
for qc in QC:
    s = (qcat >= qc) & (own >= mx)
    nsel = s.sum()
    v1, e1 = frac(s & (net > 0.1 * own), s); v2, e2 = frac(s & (net > 0.2 * own), s)
    vm, em = frac(s & (mx > 0.1 * own), s); vb, eb = frac(s & ((own + N) / sk < qc), s)
    vneg, eneg = frac(s & (net < -0.1 * own), s)
    lam = B2.lam(k[s], 0.1 * own[s]); pred = np.mean(1 - np.exp(-lam))
    print(f"  q>{qc}: N_sel/deg^2 {nsel/(NP*100):.4f} | net>10% {100*v1:.3f}+-{100*e1:.3f}  net>20% {100*v2:.3f}+-{100*e2:.3f}"
          f"  net<-10% {100*vneg:.3f}+-{100*eneg:.3f} | single nb>10% {100*vm:.3f}+-{100*em:.3f} (unselected-Poisson formula on the same objects {100*pred:.3f})"
          f" | boosted {100*vb:.3f}+-{100*eb:.3f}  [%]")
    out[f"sel_q{qc}"] = dict(net10=(v1, e1), net20=(v2, e2), nb10=(vm, em), boost=(vb, eb), netneg10=(vneg, eneg), poisson_nb10=pred,
                             n_per_deg2=nsel / (NP * 100))
json.dump(out, open(f"{L.DATA}/catalog_level{L.TAG}.json", "w"), indent=1, default=float)
