"""Catalogue-level BLENDED expectation in (z, q_obs) bins: compound-Poisson P(D) (map mean removed, dominance),
convolved with the naive selection. Fiducial cosmology and scaling relation only (this is the 'truth').

For a halo A (filter scale theta_A = theta500, true SNR q_A incl. Y scatter):
   q_obs = q_A + n + delta - mu(theta_A),   n ~ N(0,1),  delta = compound-Poisson neighbour sum (all halos M>1e13)
   detected as itself only if dominant: no neighbour with x_B >= q_A  -> P_dom = exp(-Lambda(x >= q_A))
   and then delta is the TRUNCATED compound Poisson (x < q_A).                       [as b03_analytic.py]
Kernel K_b(theta_A, q_A; edge e) = P_dom * P(q_obs > e | dom), on 40 theta nodes x ~1060 q_A nodes, from the stored
rate histograms W, WX, WX2, CAP, mu_exact of data/blending/analytic.npz (neighbour floor 1e13).
Counts: N_b = sum_nodes d2N * sum_GH K_b(theta500, q0 e^{slnY x}); the same bilinear interpolation is applied to the
naive kernel Phi(q_A - e) to give N_n; the stored data vector is   Delta_cat = N_b - N_n   (full sky), so that
   data_cat(z, q) = mu_naive(fid) + Delta_cat     (interpolation errors cancel to first order).
Output: data_cosmo/truth_catalogue.npz
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosmolib as CL

AN = f"{CL.ROOT}/data/blending/analytic.npz"
NFFT, DD, OFF = 2 ** 16, 0.01, 12.0
XSMALL = 2 * DD
QA = np.concatenate([np.arange(0.05, 30.0, 0.05), np.geomspace(30.0, 3000.0, 466)])
EDGES = np.unique(np.concatenate([CL.default_bins(5.0)[1][:-1], CL.default_bins(6.0)[1][:-1], [4.0, 7.0]]))
X_CAP = 500.0


def node(i):
    d = np.load(AN)
    W, WX, WX2 = d["W"][i].sum(0), d["WX"][i].sum(0), d["WX2"][i].sum(0)
    CAP = d["CAP"][i].sum(); mu = d["mu_exact"][i].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        xb = np.where(W > 0, WX / W, 0.0)
    k = 2 * np.pi * np.fft.rfftfreq(NFFT, d=DD)
    grid_edge = np.arange(NFFT) * DD - OFF + 0.5 * DD
    K = np.zeros((len(QA), len(EDGES))); Pdom = np.zeros(len(QA)); mean_trunc = np.zeros(len(QA))
    for j, qa in enumerate(QA):
        xcut = min(qa, X_CAP)
        keep = (W > 0) & (xb < xcut)
        lam_above = W[(W > 0) & (xb >= xcut)].sum() + CAP
        small = keep & (xb < XSMALL); big = keep & ~small
        mu_s = WX[small].sum(); v_s = WX2[small].sum()
        R = np.zeros(NFFT)
        f = xb[big] / DD; i0 = np.floor(f).astype(np.int64); fr = f - i0
        np.add.at(R, i0, W[big] * (1 - fr)); np.add.at(R, i0 + 1, W[big] * fr)
        lam = W[big].sum()
        # grid index 0 <-> value -OFF (shift by +OFF); small-x terms and the unit noise are Gaussian
        lnphi = np.fft.rfft(R) - lam - 1j * k * (mu_s + OFF) - 0.5 * k ** 2 * (v_s + 1.0)
        p = np.clip(np.fft.irfft(np.exp(lnphi), n=NFFT), 0, None)
        p /= p.sum()
        cdf = np.cumsum(p)                                    # P(delta + n <= grid_edge)
        g = grid_edge - 0.5 * DD
        mean_trunc[j] = (p * g).sum()
        t = EDGES - qa + mu                                   # q_obs > e  <=>  delta + n > e - q_A + mu
        K[j] = 1.0 - np.interp(t, grid_edge, cdf, left=0.0, right=1.0)
        Pdom[j] = np.exp(-lam_above)
    return i, K * Pdom[:, None], Pdom, mean_trunc, mu


def main():
    t0 = time.time()
    d = np.load(AN)
    th = d["th"]; NT = len(th)
    out = f"{CL.DC}/kernel_catalogue.npz"
    if not os.path.exists(out):
        Kb = np.zeros((NT, len(QA), len(EDGES))); Pd = np.zeros((NT, len(QA))); mt = np.zeros((NT, len(QA))); mus = np.zeros(NT)
        with Pool(3) as p:
            for i, K, P, m_, mu in p.imap_unordered(node, range(NT)):
                Kb[i], Pd[i], mt[i], mus[i] = K, P, m_, mu
                print(f"  theta node {i:2d} ({th[i]:6.2f}'): mu={mu:.3f}, Pdom(q_A=5)={P[np.argmin(np.abs(QA-5))]:.5f}, "
                      f"mean(delta+n | q_A=3000)-mu={m_[-1]-mu:+.2e}  [{time.time()-t0:.0f}s]", flush=True)
        np.savez(out, Kb=Kb, Pdom=Pd, mean_trunc=mt, mu=mus, th=th, QA=QA, EDGES=EDGES)
    k = np.load(out)
    Kb, th, QA_, E = k["Kb"], k["th"], k["QA"], k["EDGES"]
    Kn = ndtr(QA_[:, None] - E[None, :])[None].repeat(len(th), 0)     # naive kernel on the same grid
    print(f"untruncated check: mean(delta+n) - mu at q_A=3000 is {np.abs(k['mean_trunc'][:, -1]-k['mu']).max():.2e} (max over theta)")
    print(f"kernel at q_A=3000, e<=100: K_b/K_n - 1 max |.| = {np.abs(Kb[:, -1, E<=100]/Kn[:, -1, E<=100]-1).max():.2e}")

    lth = np.log(th)

    def make_kernel(Ktab, edges_sel):
        Ks = Ktab[:, :, edges_sel]

        def kern(th500, q):
            fa = np.clip((np.log(th500) - lth[0]) / (lth[1] - lth[0]), 0, len(th) - 1.000001)
            ia = fa.astype(int); wa = (fa - ia)[..., None]
            qq = np.clip(q, QA_[0], QA_[-1])
            jb = np.clip(np.searchsorted(QA_, qq) - 1, 0, len(QA_) - 2)
            wb = ((qq - QA_[jb]) / (QA_[jb + 1] - QA_[jb]))[..., None]
            C = (Ks[ia, jb] * (1 - wa) * (1 - wb) + Ks[ia + 1, jb] * wa * (1 - wb)
                 + Ks[ia, jb + 1] * (1 - wa) * wb + Ks[ia + 1, jb + 1] * wa * wb)        # P(q_obs > e), (..., ne)
            C = np.concatenate([C, np.zeros(C.shape[:-1] + (1,))], -1)
            return C[..., :-1] - C[..., 1:]
        return kern

    res = {}
    for qc in (5.0, 6.0):
        M = CL.NaiveModel(qcut=qc)
        sel = np.array([int(np.argmin(np.abs(E - e))) for e in M.qed[:-1]])
        assert np.allclose(E[sel], M.qed[:-1])
        Nb = M.counts_kernel(CL.FID, make_kernel(Kb, sel))
        Nn = M.counts_kernel(CL.FID, make_kernel(Kn, sel))
        N0 = M.counts(CL.FID)
        D = Nb - Nn
        print(f"\n=== q_cut={qc}: full sky ===")
        print(f" naive exact {N0.sum():,.1f} | naive via kernel grid {Nn.sum():,.1f} (rel {Nn.sum()/N0.sum()-1:+.2e}, "
              f"max per-bin |rel| (mu>10) {np.abs(Nn/N0-1)[N0>10].max():.2e}) | blended {Nb.sum():,.1f}")
        print(f" blended/naive total = {Nb.sum()/Nn.sum():.4f} (b03_analytic.log, q>{qc:.0f}, floor 1e13, mean removed: "
              f"{ {5.0: '84,494/85,898 = 0.9837', 6.0: '61,118/61,630 = 0.9917'}[qc] })")
        print(" Delta/naive per q bin:", np.round(D.sum(0) / Nn.sum(0), 4))
        print(" Delta/naive per z bin:", np.round(D.sum(1) / Nn.sum(1), 4))
        res[f"D_q{qc:.0f}"] = D; res[f"Nb_q{qc:.0f}"] = Nb; res[f"Nn_q{qc:.0f}"] = Nn; res[f"N0_q{qc:.0f}"] = N0
    np.savez(f"{CL.DC}/truth_catalogue.npz", **res)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
