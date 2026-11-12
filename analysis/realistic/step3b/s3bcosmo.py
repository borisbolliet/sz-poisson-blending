"""Step 3b cosmology helpers: the analyst's naive count model with the finder's selection kernel, data vectors, fits.

SmoothModelK = step 2's SmoothModelS (cosmolib2's converged count model: HMF grid, lognormal Y-M scatter as a smooth
convolution in ln q, lnM refined x4, bins of cosmolib.default_bins) with
  * sigma(theta500) = sigma_eff = rbar(theta) sigma_Y(theta): rbar = the variant's mean r = sigma_finder / sigma_Y after
    the iteration over the 324 realistic tiles, linear in ln theta between the 15 scales, constant beyond
    (s2lib.LnSigma, as step 2's map-sigma models); q' = Y500 / sigma_eff;
  * the variant's kernel K(theta, q'; e) in place of Phi(q - e), with p05's machinery (s2lib.bin_tables): per kernel
    theta node the kernel is interpolated linearly in q' onto the model's fine ln q grid, differenced between the bin
    edges, convolved with the lognormal scatter; counts are linear in ln theta between the nodes (counts_table).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
for _p in (f"{B.ROOT}/analysis/realistic/step2",):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import s2lib as S2                                 # noqa: E402
from s2lib import CL, C2                            # noqa: E402
from scipy.special import ndtr                     # noqa: E402

F = B.F
FID = S2.FID.copy()
SURVEYS = {"SO40": 0.4, "full": 1.0}
CFGS = ["a", "b", "c", "d", "d3", "d5", "e"]
QCUTS = (5.0, 6.0)
AREA_TILES = 324 * 100.0
AREA_SKY = 4 * np.pi * (180 / np.pi) ** 2


class SmoothModelK(S2.SmoothModelS):
    def __init__(self, qcut, lnsig, th, QA, K, edges=B.EDGES, grid=None):
        super().__init__(qcut, lnsig, grid=grid)
        self.kth = np.asarray(th, float)
        self.klth = np.log(self.kth)
        assert np.allclose(np.diff(self.klth), self.klth[1] - self.klth[0])
        sel = [int(np.argmin(np.abs(edges - e))) for e in self.qed[:-1]]
        assert np.allclose(np.asarray(edges)[sel], self.qed[:-1])
        q = np.clip(np.exp(self.lq), QA[0], QA[-1])
        P = []
        for i in range(self.kth.size):
            Ci = np.stack([np.interp(q, QA, K[i][:, j]) for j in sel], 0)
            Ci = np.concatenate([Ci, np.zeros((1, Ci.shape[1]))], 0)
            P.append(Ci[:-1] - Ci[1:])
        self.PrawK = np.stack(P)                                                   # (nth, nqb, nlq), no scatter
        self._PK = {}

    def tableK(self, slnY):
        key = round(float(slnY), 12)
        if key not in self._PK:
            while len(self._PK) >= 4:                                   # keep at most 4 tables (26 MB each)
                self._PK.pop(next(iter(self._PK)))
            self._PK[key] = np.ascontiguousarray(np.transpose(C2.smooth_lnq(self.PrawK, slnY, self.h), (0, 2, 1)))
        return self._PK[key]

    def counts(self, th, fsky=1.0):
        return self.counts_table(th, self.tableK(th[4]), self.klth, fsky)


_G = {}


def grid():
    if "G" not in _G:
        _G["G"] = CL.HMFGrid()
    return _G["G"]


def rbar(variant, mode):
    return np.load(os.path.join(B.OUT, f"halos_{variant}_{mode}.npz"))["r1"].mean(0)


def lnsig(variant, mode):
    return S2.LnSigma(S2.sz_model(mode), F.THETAS15, rbar(variant, mode))


def analytic_kernel(k, QA=None, th=None):
    """Phi(sqrt(q'^2 + k) - e) on a fine q' grid, the same for every theta node."""
    QA = np.concatenate([np.arange(0.0, 30.0, 0.01), np.geomspace(30.0, 3000.0, 800)]) if QA is None else QA
    th = B.TH_K if th is None else th
    Kq = ndtr(np.sqrt(QA[:, None] ** 2 + k) - B.EDGES[None, :])
    return th, QA, np.broadcast_to(Kq, (th.size,) + Kq.shape).copy()


def model(variant, mode, qcut, kernel="virtual", k=None):
    """The analyst's naive model: kernel 'virtual' (b05), 'even' / 'odd' (half kernels), 'analytic' (Phi(sqrt(q'^2 + k) - e))
    or 'analytic_cl' (the same, zero below the credit limit q' = 2)."""
    if kernel == "virtual":
        d = np.load(os.path.join(B.OUT, f"kernel_{variant}_{mode}.npz"))
        th, QA, K = d["th"], d["QA"], d["K"]
    elif kernel in ("even", "odd"):
        d = np.load(os.path.join(B.OUT, f"kernel_{variant}_{mode}.npz"))
        th, QA, K = d["th"], d["QA"], d["K_" + kernel]
    elif kernel == "analytic_cl":                                     # analytic kernel, zero below the credit limit q' = 2
        th, QA, K = analytic_kernel(k)
        K[:, QA < B.CRED_MIN, :] = 0.0
    else:
        th, QA, K = analytic_kernel(k)
    return SmoothModelK(qcut, lnsig(variant, mode), th, QA, K, grid=grid())


def linear_parts(M, truth, R, cn, fsky):
    """Linearised bias (Fisher sigma) of S8, Om, s8 for data = mu (R): total and p06b's split into amplitude, z tilt,
    q shape.  Also returns (Ft, J, sig, gS, N0) for reuse."""
    free, pri = S2.config(cn, truth)
    fi = [CL.PNAMES.index(p) for p in free]
    N0 = M.counts(truth)
    F_ = S2.Fitter3(M, N0 * fsky, free, fsky=fsky, priors=pri, truth=truth)
    Ft = F_.fisher(truth[fi]); J = F_.jac(truth[fi])
    sig, Cov, gS = S2.derived_sigma3(Ft, fi, truth)
    return dict(Ft=Ft, J=J, sig=sig, gS=gS, N0=N0, fi=fi, free=free)


def lin_bias(L, R):
    """Linearised biases for one or many R arrays (..., nz, nq): S8, Om, s8 in Fisher sigma."""
    R = np.asarray(R, float)
    X = (R - 1.0).reshape(-1, R.shape[-2] * R.shape[-1])                         # (n, nbin)
    x = np.linalg.solve(L["Ft"], L["J"].T @ X.T).T                               # (n, nfree)
    s8 = x @ L["gS"] / L["sig"]["S8"]
    om = x[:, 0] / L["sig"]["Om"]; s8_ = x[:, 1] / L["sig"]["s8"]
    return s8, om, s8_, x


def split_parts(L, R):
    """p06b's decomposition of Delta = N0 (R - 1): amplitude, z tilt, q shape -> linearised S8 bias of each part."""
    N0 = L["N0"]
    D = N0 * (R - 1.0)
    eps = D.sum() / N0.sum()
    rz = D.sum(1) / np.maximum(N0.sum(1), 1e-300)
    parts = {"amplitude": eps * N0, "z tilt": (rz[:, None] - eps) * N0}
    parts["q shape"] = D - parts["amplitude"] - parts["z tilt"]
    out = {}
    for k, Dk in parts.items():
        out[k] = float(lin_bias(L, 1.0 + Dk / N0)[0][0])
    out["sum"] = float(lin_bias(L, R)[0][0])
    return out


# ============================================================================ pooled blending correction (R_s)
# q' nodes of the correction Delta K(theta, q'; e) = P(credited, q_obs > e | blended sky) - K (bilinear tents in ln theta
# on the kernel nodes TH_K and in q' on QDK; constant beyond the last q' node, as np.interp).
QDK = np.round(np.concatenate([np.arange(0.0, 2.0, 0.25), np.arange(2.0, 10.0, 0.25), np.arange(10.0, 20.0, 0.5),
                               np.arange(20.0, 40.0, 1.0), np.arange(40.0, 80.0, 2.5), np.arange(80.0, 120.0, 5.0),
                               [120.0, 150.0, 200.0, 300.0]]), 6)


def cell_weights(theta, qp, qnodes=None):
    """Bilinear (tent) weights of points (theta, q') to the cells (TH_K node i, q' node k of qnodes, default QDK):
    (idx (n, 4), w (n, 4)), flat cell index i * nq + k; the same interpolation as counts_table in ln theta and np.interp
    in q'."""
    QDK_ = QDK if qnodes is None else np.asarray(qnodes, float)
    nK, nq = B.TH_K.size, QDK_.size
    x = np.clip((np.log(theta) - np.log(B.TH_K[0])) / np.log(B.TH_K[1] / B.TH_K[0]), 0, nK - 1.000001)
    i0 = x.astype(np.int64); wt = x - i0
    qq = np.clip(qp, QDK_[0], QDK_[-1])
    k0 = np.clip(np.searchsorted(QDK_, qq, side="right") - 1, 0, nq - 2)
    wq = np.clip((qq - QDK_[k0]) / (QDK_[k0 + 1] - QDK_[k0]), 0.0, 1.0)
    idx = np.stack([i0 * nq + k0, (i0 + 1) * nq + k0, i0 * nq + k0 + 1, (i0 + 1) * nq + k0 + 1], 1)
    w = np.stack([(1 - wt) * (1 - wq), wt * (1 - wq), (1 - wt) * wq, wt * wq], 1)
    return idx, w


def cell_integrals(M, truth, qnodes=None):
    """I[z bin, i, k] = full-sky count integral (model M at the truth, lognormal scatter included) of the bilinear basis
    function of cell (TH_K node i, QDK node k): the count of halos that the cell's tent picks up, per z bin.  Exact
    counterpart of SmoothModelK.counts for a kernel defined on (TH_K, QDK)."""
    QDK_ = QDK if qnodes is None else np.asarray(qnodes, float)
    lnN, th500, lnq0 = M.nodes(truth)
    Wn = np.exp(lnN) * M.wM[None, :]
    nK, nq = B.TH_K.size, QDK_.size
    fa = np.clip((np.log(th500) - M.klth[0]) / (M.klth[1] - M.klth[0]), 0, nK - 1.000001)
    ia = fa.astype(np.int64); wa = fa - ia
    f = np.clip((lnq0 - M.lq[0]) / M.h, 0, len(M.lq) - 1.000001); i0 = f.astype(np.int64); w = f - i0
    q = np.exp(M.lq)
    out = np.zeros((M.nzb, nK, nq))
    eye = np.eye(nq)
    for k in range(nq):
        T = np.interp(q, QDK_, eye[k])
        Ts = C2.smooth_lnq(T[None, :], truth[4], M.h)[0]
        v = Wn * (Ts[i0] * (1 - w) + Ts[i0 + 1] * w)                              # (nz, nMf)
        A = np.zeros((v.shape[0], nK))
        np.add.at(A, (np.arange(v.shape[0])[:, None].repeat(v.shape[1], 1), ia), v * (1 - wa))
        np.add.at(A, (np.arange(v.shape[0])[:, None].repeat(v.shape[1], 1), ia + 1), v * wa)
        out[:, :, k] = M.Wz @ A
    return out


def rs_from_dk(Icell, mu, dK, sel):
    """R_s(z, b) = 1 + sum_c I_c(z) [dK_c(e_b) - dK_c(e_b+1)] / mu(z, b); dK (..., nK, nq, ne), sel = edge index of each
    bin's lower edge (the last bin is open).  Returns (..., nz, nb)."""
    X = np.einsum("zik,...ike->...ze", Icell, dK[..., sel])
    D = X - np.concatenate([X[..., 1:], np.zeros(X.shape[:-1] + (1,))], -1)
    return 1.0 + D / mu
