"""Is 'confusion = extra Gaussian noise' good enough?  Catalog-level halos (q_true>=1, 108 patches, own scale k_own):
   q_own = own/sigma_k,  X = net contamination/sigma_k (actual, from catalog_level.py), N = actual map noise/sigma_k.
Gaussian model: X ~ N(0, s_c^2) with s_c^2 = analytic Poisson variance of the undetected halos (q_true < 5) at k_own.
   P_det = Phi((q_own - qc)/sqrt(1+s_c^2));  P_boost = P(q_own+N < qc <= q_own+N+X) (bivariate normal)
Compared with the actual-X catalogue (q_cat = q_own+X+N >= qc and own >= max neighbour) and with the map.
Also: the distribution of the filter scale of the map detections (to relate theta_b to the matched-filter scale)."""
import numpy as np, json
from scipy.special import ndtr
from scipy.stats import multivariate_normal
import q2lib as L
from szsim import cosmo_hmf as ch, signal as sg

R = np.load(f"{L.DATA}/catalog_level_rows{L.TAG}.npy")
k, qt, own, net, mx, N, sk = R[:, 2].astype(int), R[:, 3], R[:, 5], R[:, 6], R[:, 7], R[:, 8], R[:, 9]
PIDS = L.pids_from_env(); NP = len(PIDS); AREA = NP * 100.0
# analytic confusion variance of q_true<5 halos per scale (same formula as confusion_qsplit.py)
tab = L.build_tables(); sigH = tab["sigk_arcmin2"]; K = len(L.THETA_K); m = L.MODEL; s = m.sigma_lnY
lw = np.linspace(0, 10 / L.SIGB, 40001); B4 = np.exp(-2 * (lw * L.SIGB) ** 2); thg = np.geomspace(0.005, 120, 400)
W = np.array([[(sigH[kk] ** 2 / L.NW) ** 2 * np.trapezoid(lw * B4 * sg.template_hankel(lw * L.THETA_K[kk]) ** 2 *
               sg.template_hankel(lw * t) ** 2, lw) / (2 * np.pi) for t in thg] for kk in range(K)])
T = ch.default_table(); Zg, LMg = np.meshgrid(T.z, T.lnM, indexing="ij")
thT = m.theta500(np.exp(LMg), Zg); YT = m.Y500_med(np.exp(LMg), Zg); sYT = m.sigma_Y(thT)
wz = T._weights(T.z, T.z[0], T.z[-1]); wM = T._weights(T.lnM, T.lnM[0], T.lnM[-1])
mom = YT ** 2 * np.exp(2 * s * s) * ndtr((np.log(5 * sYT / YT) - 2 * s * s) / s)
sc2 = np.array([wz @ (T.d2N * mom * np.exp(np.interp(np.log(thT), np.log(thg), np.log(W[kk])))) @ wM for kk in range(K)]) \
      / (4 * np.pi / ch.ARCMIN ** 2) / sigH ** 2
print("analytic confusion variance (q_true<5 halos) / sigma_k^2:", np.array2string(sc2, precision=3))
qo = own / sk; X = net / sk; n = N / sk; c2 = sc2[k]
print(f"actual X at halo positions (q_true>=1): mean {X.mean():+.4f}, median {np.median(X):+.4f}, std {X.std():.4f};"
      f" Gaussian-model std at the same scales {np.sqrt(c2.mean()):.4f}")
blend = json.load(open(f"{L.DATA}/blend_results{L.TAG}.json"))
for qc in (4, 5, 6, 7):
    pdet_g = ndtr((qo - qc) / np.sqrt(1 + c2))
    pdet_0 = ndtr(qo - qc)
    # P(q+N < qc, q+N+X >= qc) for N~N(0,1), X~N(0,c2): bivariate normal of (N, N+X)
    a = qc - qo
    pb = np.empty(qo.size)
    for i in range(qo.size):
        if qo[i] < qc - 8 or qo[i] > qc + 8:
            pb[i] = 0.0; continue
        cov = [[1, 1], [1, 1 + c2[i]]]
        # P(N < a, N+X >= a) = P(N < a) - P(N < a, N+X < a)
        pb[i] = ndtr(a[i]) - multivariate_normal.cdf([a[i], a[i]], mean=[0, 0], cov=cov)
    sel = (qo + X + n >= qc) & (own >= mx)
    boost_act = ((qo + n < qc) & sel).sum() / sel.sum()
    mapc = blend[f"counts_q{qc}"]["V13"][0]
    mapb = blend["V13_R2"]["boost"][[4, 5, 6, 7].index(qc)][0]
    print(f"q>{qc}: N_det/deg^2: no confusion {pdet_0.sum()/AREA:.4f} | Gaussian confusion {pdet_g.sum()/AREA:.4f} | actual-X catalogue "
          f"{sel.sum()/AREA:.4f} | map (maxQ, R2) {mapc:.4f}")
    print(f"       boosted fraction: Gaussian {pb.sum()/pdet_g.sum()*100:.2f}% | actual-X catalogue {100*boost_act:.2f}% | map {100*mapb:.2f}%")

# filter scale of map detections (maxQ, V13, R2, q>5) and theta500 of their primaries
ths, th5 = [], []
for p in PIDS:
    o = np.load(f"{L.DATA}/patch_out/patch_{p:03d}.npz"); dc = np.load(f"{L.DATA}/decomp/decomp_{p:03d}.npz")
    rec = dc["cand_V13"]; Q = rec[:, 3]; kk = rec[:, 2].astype(int)
    x = (rec[:, 1] + 0.5) * L.PIX; y = (rec[:, 0] + 0.5) * L.PIX
    acc = L.merge_greedy(x, y, L.THETA_K[kk], Q.copy(), 2.0, 0.0, L.LBOX)
    acc = acc[Q[acc] >= 5]
    d = L.load_patch(p)
    ths.append(L.THETA_K[kk[acc]]); ip = rec[acc, 6].astype(int); th5.append(d["theta500"][ip[ip >= 0]])
ths = np.concatenate(ths); th5 = np.concatenate(th5)
print(f"map detections q>5: best filter scale theta_k: median {np.median(ths):.3f}', 16-84% [{np.percentile(ths,16):.3f}, {np.percentile(ths,84):.3f}];"
      f" primary theta500: median {np.median(th5):.3f}', 16-84% [{np.percentile(th5,16):.3f}, {np.percentile(th5,84):.3f}]; beam FWHM 1.4'")
