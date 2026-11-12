"""Step-3b verification, lens 'fits': my own library (data vector, kernel interpolation, count model, fitter).

Imports from the project ONLY szsim (instrument: sigma_Y of step 1's MMF; finder: the finder itself, for f03).  Nothing
from analysis/realistic/step3b (s3blib, s3bcosmo, b*), step 2's s2lib or cosmolib(2) is imported.  The build's stored
arrays (per-halo tables, kernel tables, datavec files) are read as data.

Count model (adapted from verify_real/step2_catalogue/vcmodel.py, which was written independently of step 2's build):
  mu[z bin, q bin] = fsky * sum_{z, M nodes} w d2N/dz/dlnM(Om, s8) P(q_obs in bin | theta500, q'_med, slnY)
  q'_med = Y_med(M, z; Om, (1-b), alpha) / sigma_eff(theta500),  sigma_eff = rbar(theta) sigma_Y(theta),
  rbar = mean over the 324 Q2 tiles of the variant's r = sigma_finder/sigma_Y (15 scales), linear in ln theta, constant
  beyond; P(q_obs > e | theta, q') = the kernel K (build's tables, MY interpolation) or an analytic kernel.
  The kernel is put on a fine u = ln q' grid per kernel theta node, differenced between bin edges, convolved with the
  lognormal scatter by exact hat-function weights (FFT), then interpolated linearly in u and in ln theta.
  HMF: out/hmf_grid_f01s.npz (f01b: direct colossus at every node, Tinker08 with unrounded Delta_m), tensor cubic-spline
  weights in (Om, s8).
Fitter: Nelder-Mead on the exact Poisson lnL (+ Gaussian prior) in Fisher-whitened coordinates, restarted, then Newton
with the finite-difference Hessian of lnL; linearised estimator F^-1 J^T (d - mu)/mu.
"""
import os, sys
_NT = os.environ.get("VF_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _NT
import numpy as np
from scipy.special import ndtr
from scipy.signal import fftconvolve
from scipy.interpolate import CubicSpline, PchipInterpolator

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
D3 = os.path.join(ROOT, "data_real", "step3")
D3B = os.path.join(ROOT, "data_real", "step3b")
MODES = ("baseline", "goal")
ARC = np.pi / 180 / 60
LOGY0 = np.log(1e-4) - 0.19 * np.log(10.0)
FID = np.array([0.3111, 0.8102, 0.8, 1.79, 0.173])          # Om, s8, (1-b), alpha, slnY
PN = ["Om", "s8", "omb", "alpha", "slnY"]
ZED = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
THETAS15 = np.geomspace(0.5, 15.0, 15)                       # the finder's 15 filter sizes
CRED = 2.0                                                    # credit limit q' = 2 (no credited halo below)
SURV = {"SO40": 0.4, "full": 1.0}


def qlow(qcut):
    """the 9 lower bin edges (q_cut .. 100, geometric); the last bin is q > 100"""
    return np.geomspace(qcut, 100.0, 9)


def qbins(qcut):
    return np.concatenate([qlow(qcut), [np.inf]])


EDGES = np.unique(np.round(np.concatenate([qlow(5.0), qlow(6.0), [4.0, 7.0]]), 12))     # 19 edges
_E = np.load(os.path.join(ROOT, "data_cosmo", "kernel_catalogue.npz"))["EDGES"]
assert np.allclose(EDGES, _E), "edge definition differs from the kernel tables' EDGES"


def edge_idx(qcut):
    return np.array([int(np.argmin(np.abs(EDGES - e))) for e in qlow(qcut)])


def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


# ============================================================================ sigma_Y and finder units
_SIG = {}


def sigma_Y(mode, theta):
    """step 1's MMF sigma_Y [arcmin^2] from szsim.instrument (SO LAT, l = 100-20000)."""
    if mode not in _SIG:
        from szsim import instrument as ins
        _SIG[mode] = ins.MMFModel(ins.Instrument.so_lat(mode, lmin=100.0, lmax=20000.0))
        _SIG[mode].sigma_Y(1.0)
    return _SIG[mode].sigma_Y(np.asarray(theta, float))


def r_interp(theta, r15):
    """r(theta) linear in ln theta between the 15 filter sizes, constant beyond (np.interp)."""
    return np.interp(np.log(np.asarray(theta, float)), np.log(THETAS15), np.asarray(r15, float))


# ============================================================================ per-halo data (build's arrays, read as data)
def load_halos(mode, variant, tiles="q2", merges=("none", "fof10")):
    """Per-halo arrays: t (tile index), z, th (theta500), Y (Y500), qp (stored q'), qo_<merge> (credited q_obs or -inf),
    qiso (isolated copy q_obs or -inf), sig1 (per tile, 15 scales).  tiles 'q2' = the 324 Q2 tiles; 'all' adds the
    1224 new tiles (production only), tile index 324 + k in sorted order of the files."""
    C = np.load(os.path.join(D3B, f"halos_{mode}.npz"))
    V = np.load(os.path.join(D3B, f"halos_{variant}_{mode}.npz"))
    out = dict(t=C["pid"].astype(np.int64), z=C["z"].astype(float), th=C["theta500"].astype(float),
               Y=C["Y500"].astype(float), qt=C["q_true"].astype(float), qp=V["qp"].astype(float))
    for m in merges:
        out["qo_" + m] = V["qobs_" + m].astype(float)
    sig1 = [V["sig1"]]
    n = out["t"].size
    qi = np.full(n, -np.inf); seen = np.zeros(n, bool)
    for p in range(324):
        with np.load(os.path.join(D3B, "virtual", variant, f"v_{mode}_{p:03d}.npz")) as d:
            r = d["iso_row"]; qi[r] = d["iso_qobs"]; seen[r] = True
    out["qiso"] = qi; out["iso_seen"] = seen
    if tiles == "all":
        assert variant == "prod"
        fs = sorted(f for f in os.listdir(os.path.join(D3B, "newtiles")) if f.endswith(".npz"))
        A = {k: [] for k in ["t", "z", "th", "Y", "qt", "qp", "qiso"] + ["qo_" + m for m in merges]}
        for k, f in enumerate(fs):
            pid = int(f[2:6])
            with np.load(os.path.join(D3B, "newtiles", f)) as d, np.load(os.path.join(D3B, "truth_new", f"truth_{pid:04d}.npz")) as tr:
                rows = d[mode + "_rows"]
                A["t"].append(np.full(rows.size, 324 + k, np.int64))
                A["z"].append(tr["z"][rows].astype(float)); A["th"].append(tr["theta500"][rows].astype(float))
                A["Y"].append(tr["Y500"][rows].astype(float)); A["qt"].append(tr["q_" + mode][rows].astype(float))
                A["qp"].append(d[mode + "_qp"].astype(float)); A["qiso"].append(d[mode + "_qiso"].astype(float))
                for m in merges:
                    A["qo_" + m].append(d[f"{mode}_qobs_{m}"].astype(float))
                sig1.append(d[mode + "_sig1"][None, :])
        for k, v in A.items():
            out[k] = np.concatenate([out[k], np.concatenate(v)])
        out["iso_seen"] = np.concatenate([seen, np.ones(out["t"].size - n, bool)])
    out["sig1"] = np.concatenate(sig1, 0)
    out["nt"] = out["sig1"].shape[0]
    return out


def rbar(variant, mode):
    """the analyst's rbar: mean r over the 324 Q2 tiles (the variant's realistic-tile runs)."""
    V = np.load(os.path.join(D3B, f"halos_{variant}_{mode}.npz"))
    return (V["sig1"] / sigma_Y(mode, THETAS15)[None, :]).mean(0)


# ============================================================================ kernel tables (build's), my interpolation
class KTable:
    """K(theta node i, q'; e) from a build kernel file.  Interpolation:
       'bilin' : linear in q' between the QA nodes and linear in ln theta between the nodes, clamped (the build's scheme)
       'alt'   : monotone cubic (PCHIP) in q' per (theta node, edge); in ln theta a 4-point Lagrange (cubic) through the
                 neighbouring nodes (clamped to [0, 1]).
    K = 0 below the credit limit q' = 2 in both."""

    def __init__(self, path=None, th=None, QA=None, K=None):
        if path is not None:
            d = np.load(path)
            th, QA, K = d["th"], d["QA"], d["K"]
        self.th = np.asarray(th, float); self.lth = np.log(self.th); self.QA = np.asarray(QA, float); self.K = np.asarray(K, float)
        self.dl = self.lth[1] - self.lth[0]
        assert np.allclose(np.diff(self.lth), self.dl)

    def on_q(self, q, how="bilin", eidx=None):
        """K at the theta nodes for an array of q': (nth, nq, ne) for the edges eidx (default all 19)."""
        q = np.asarray(q, float)
        ei = np.arange(self.K.shape[2]) if eidx is None else np.asarray(eidx)
        Ks = self.K[:, :, ei]
        qq = np.clip(q, self.QA[0], self.QA[-1])
        out = np.empty((self.th.size, q.size, ei.size))
        if how == "bilin":
            k = np.clip(np.searchsorted(self.QA, qq) - 1, 0, self.QA.size - 2)
            u = (qq - self.QA[k]) / (self.QA[k + 1] - self.QA[k])
            out[:] = (1 - u)[None, :, None] * Ks[:, k, :] + u[None, :, None] * Ks[:, k + 1, :]
        else:
            for i in range(self.th.size):
                out[i] = PchipInterpolator(self.QA, Ks[i], axis=0, extrapolate=False)(qq)
        out[:, q < CRED, :] = 0.0
        return out

    def theta_weights(self, theta, how="bilin"):
        """indices (n, m) and weights (n, m) in the theta nodes."""
        x = np.clip((np.log(np.asarray(theta, float)) - self.lth[0]) / self.dl, 0.0, self.th.size - 1.000001)
        i0 = x.astype(np.int64); f = x - i0
        if how == "bilin":
            return np.stack([i0, i0 + 1], 1), np.stack([1 - f, f], 1)
        # cubic Lagrange through i0-1 .. i0+2 (shifted at the ends)
        s = np.clip(i0 - 1, 0, self.th.size - 4)
        t = x - s                                           # position relative to node s, in [0, 3]
        idx = s[:, None] + np.arange(4)[None, :]
        w = np.stack([-(t - 1) * (t - 2) * (t - 3) / 6, t * (t - 2) * (t - 3) / 2, -t * (t - 1) * (t - 3) / 2,
                      t * (t - 1) * (t - 2) / 6], 1)
        return idx, w

    def at(self, theta, q, eidx, how="bilin", chunk=200000):
        """K(theta_h, q_h; e) for halos h and the edges eidx: (n, len(eidx))."""
        theta = np.asarray(theta, float); q = np.asarray(q, float); eidx = np.atleast_1d(eidx)
        out = np.empty((q.size, eidx.size))
        for a in range(0, q.size, chunk):
            sl = slice(a, a + chunk)
            idx, w = self.theta_weights(theta[sl], how)
            qq = np.clip(q[sl], self.QA[0], self.QA[-1])
            if how == "bilin":
                k = np.clip(np.searchsorted(self.QA, qq) - 1, 0, self.QA.size - 2)
                u = (qq - self.QA[k]) / (self.QA[k + 1] - self.QA[k])
                v = np.zeros((qq.size, eidx.size))
                for c in range(idx.shape[1]):
                    Kc = (1 - u)[:, None] * self.K[idx[:, c], k][:, eidx] + u[:, None] * self.K[idx[:, c], k + 1][:, eidx]
                    v += w[:, c][:, None] * Kc
            else:
                if not hasattr(self, "_pch"):
                    self._pch = [PchipInterpolator(self.QA, self.K[i], axis=0, extrapolate=False) for i in range(self.th.size)]
                v = np.zeros((qq.size, eidx.size))
                for i in np.unique(idx):
                    m = idx == i                                  # (n, 4) mask of where node i enters
                    rows = np.flatnonzero(m.any(1))
                    if rows.size == 0:
                        continue
                    Ki = self._pch[i](qq[rows])[:, eidx]
                    v[rows] += (w[rows] * m[rows]).sum(1)[:, None] * Ki
                v = np.clip(v, 0.0, 1.0)
            v[q[sl] < CRED] = 0.0
            out[sl] = v
        return out


class AnalyticKernel:
    """Phi(sqrt(q'^2 + k) - e), zero below the credit limit q' = 2 when cl; one theta node (no theta dependence)."""

    def __init__(self, k, cl=True):
        self.k, self.cl = float(k), cl
        self.th = np.array([1.0]); self.lth = np.array([0.0])

    def on_q(self, q, how=None, eidx=None):
        q = np.asarray(q, float)
        E = EDGES if eidx is None else EDGES[np.asarray(eidx)]
        v = ndtr(np.sqrt(q[:, None] ** 2 + self.k) - E[None, :])
        if self.cl:
            v[q < CRED] = 0.0
        return v[None]


# ============================================================================ count model
_G = {}


GRID = os.environ.get("VF_GRID", "f01s")                       # f01s: smooth Tinker08 (default); f01: colossus' rounded Delta_m


def grid():
    """HMF grid of f01b (smooth Tinker08; VF_GRID=f01 for f01's rounded one); the float64 table ln d2N (nOm*ns8, nz*nM) is
    memory-mapped (shared page cache between processes)."""
    if not _G:
        d = np.load(os.path.join(OUT, f"hmf_grid_{GRID}.npz"))
        _G.update({k: d[k] for k in d.files if k != "L"})
        _G["Lf"] = np.load(os.path.join(OUT, f"hmf_grid_{GRID}_Lf64.npy"), mmap_mode="r")
        _G["spE"] = CubicSpline(_G["OM"], _G["LE"], axis=0)
        _G["spD"] = CubicSpline(_G["OM"], _G["LD"], axis=0)
        _G["wO"] = CubicSpline(_G["OM"], np.eye(len(_G["OM"])))
        _G["wS"] = CubicSpline(_G["S8"], np.eye(len(_G["S8"])))
    return _G


def hat_gauss_weights(s, h, nsig=9.0):
    """w_k = int phi_s(t) Lambda(t/h - k) dt (exact convolution of the piecewise-linear interpolant), k = -K..K."""
    K = int(np.ceil(nsig * max(s, h) / h)) + 2
    k = np.arange(-K, K + 1)
    if s <= 0:
        return (k == 0).astype(float), K
    t = np.arange(-K - 1, K + 2) * h
    F = ndtr(t / s)
    G = -s * s * np.exp(-0.5 * (t / s) ** 2) / (s * np.sqrt(2 * np.pi))
    i = np.arange(1, 2 * K + 2)
    left = (G[i] - G[i - 1] - t[i - 1] * (F[i] - F[i - 1])) / h
    right = (t[i + 1] * (F[i + 1] - F[i]) - (G[i + 1] - G[i])) / h
    w = left + right
    return w / w.sum(), K


class KModel:
    """mu(theta) in 16 z bins x 9 q_obs bins (q_cut..100 geometric, + q > 100) for a kernel object (KTable or
    AnalyticKernel), sigma_eff = rbar sigma_Y."""
    H = 0.002
    U = np.arange(np.log(1e-3), np.log(3e4) + 0.001, 0.002)

    def __init__(self, mode, qcut, r15, kern, how="alt"):
        G = grid()
        self.mode, self.qcut = mode, qcut
        self.r15 = np.asarray(r15, float)
        self.ZN, self.ZW, self.ZB, self.LNM, self.WM = G["ZN"], G["ZW"], G["ZB"], G["LNM"], G["WM"]
        self.nz = len(ZED) - 1
        self.Zsum = np.zeros((self.nz, len(self.ZN)))
        self.Zsum[self.ZB, np.arange(len(self.ZN))] = self.ZW
        self.kern = kern
        self.lth = kern.lth; self.nth = kern.lth.size
        self.dlth = (self.lth[1] - self.lth[0]) if self.nth > 1 else 1.0
        ei = edge_idx(qcut)
        Kq = kern.on_q(np.exp(self.U), how, eidx=ei)                     # (nth, nU, 9): P(q_obs > e_b | th_i, q')
        P = Kq.copy(); P[:, :, :-1] -= Kq[:, :, 1:]
        self.P0 = np.ascontiguousarray(np.transpose(P, (0, 2, 1)))        # (nth, 9, nU), no scatter
        # sigma_Y on a fine ln theta grid (cubic spline in ln-ln; the theta500 of the model nodes is inside)
        self._lt = np.linspace(np.log(0.01), np.log(300.0), 4000)
        sY = sigma_Y(mode, np.exp(self._lt))
        self._lsig = CubicSpline(self._lt, np.log(sY) + np.log(r_interp(np.exp(self._lt), self.r15)))
        self._cache = {}

    def table(self, slnY):
        key = round(float(slnY), 13)
        T = self._cache.get(key)
        if T is None:
            w, K = hat_gauss_weights(float(slnY), self.H)
            P0 = self.P0
            gp = np.concatenate([np.repeat(P0[:, :, :1], K, 2), P0, np.repeat(P0[:, :, -1:], K, 2)], 2)
            T = fftconvolve(gp, w[None, None, :], mode="valid", axes=2)    # (nth, 9, nU)
            T = np.ascontiguousarray(np.transpose(T, (0, 2, 1)))            # (nth, nU, 9)
            if len(self._cache) > 3:
                self._cache.clear()
            self._cache[key] = T
        return T

    def nodes(self, th):
        om, s8, omb, alpha, slnY = th
        G = grid()
        wO = G["wO"](om); wS = G["wS"](s8)
        lnN = (np.outer(wO, wS).ravel() @ G["Lf"]).reshape(len(self.ZN), len(self.LNM))
        lnE = G["spE"](om)[:, None]; lnD = G["spD"](om)[:, None]
        lnM = self.LNM[None, :]
        lnY = LOGY0 + (2.0 / 3.0) * lnE + alpha * (np.log(omb) + lnM - np.log(6e14)) - 2 * lnD - 2 * np.log(ARC)
        lnR = (np.log(3.0) + lnM - np.log(4 * np.pi * 500 * float(G["rhoc0"])) - 2 * lnE) / 3.0
        lth = lnR - lnD - np.log(ARC)                                     # ln theta500 [arcmin]
        u = lnY - self._lsig(lth)                                          # ln q'_med
        return lnN, u, lth

    def sel(self, th):
        """P(q_obs in bin | node) for every (z, M) node: (nZN, nM, 9)."""
        lnN, u, lth = self.nodes(th)
        T = self.table(th[4])
        f = np.clip((u - self.U[0]) / self.H, 0, len(self.U) - 1.000001)
        i0 = f.astype(np.int64); w = (f - i0)[..., None]
        if self.nth == 1:
            Pn = T[0, i0] * (1 - w) + T[0, i0 + 1] * w
        else:
            g = np.clip((lth - self.lth[0]) / self.dlth, 0, self.nth - 1.000001)
            ia = g.astype(np.int64); wa = (g - ia)[..., None]
            Pn = (1 - wa) * (T[ia, i0] * (1 - w) + T[ia, i0 + 1] * w) + wa * (T[ia + 1, i0] * (1 - w) + T[ia + 1, i0 + 1] * w)
        return lnN, Pn, u, lth

    def counts(self, th, fsky=1.0):
        lnN, Pn, u, lth = self.sel(th)
        A = np.exp(lnN) * self.WM[None, :]
        perz = np.einsum("zmj,zm->zj", Pn, A)
        return fsky * (self.Zsum @ perz)

    def cell_integrals(self, th, lth_edges, q_edges):
        """I[z bin, i, k] = full-sky model count of halos with theta500 in cell i (edges in ln theta, open ends) and
        q' in [q_edges[k], q_edges[k+1]) (lognormal scatter exact), at parameters th.  theta: each (z, M) node is
        spread over a cell of width = the local node spacing in ln theta (fractional overlap with the cells)."""
        lnN, u, lth = self.nodes(th)
        s = th[4]
        A = np.exp(lnN) * self.WM[None, :] * self.ZW[:, None]               # (nZN, nM)
        # q' cell probabilities
        with np.errstate(divide="ignore"):
            lq = np.log(np.asarray(q_edges, float))
        cdf = ndtr((lq[None, None, :] - u[..., None]) / s)                  # (nZN, nM, nqe)
        Pq = cdf[..., 1:] - cdf[..., :-1]                                    # (nZN, nM, nqc)
        # theta cell fractions: node interval [lth - d/2, lth + d/2], d = half the distance to the neighbouring nodes
        d = np.gradient(lth, axis=1)
        lo, hi = lth - 0.5 * np.abs(d), lth + 0.5 * np.abs(d)
        E = np.concatenate([[-np.inf], np.asarray(lth_edges, float), [np.inf]])
        nci = E.size - 1
        frac = np.clip(np.minimum(hi[..., None], E[None, None, 1:]) - np.maximum(lo[..., None], E[None, None, :-1]), 0, None)
        frac /= np.maximum(hi - lo, 1e-300)[..., None]                       # (nZN, nM, nci)
        out = np.zeros((self.nz, nci, Pq.shape[-1]))
        for zb in range(self.nz):
            m = self.ZB == zb
            out[zb] = np.einsum("zm,zmi,zmk->ik", A[m], frac[m], Pq[m])
        return out


def S8_of(om, s8):
    return s8 * np.sqrt(om / 0.3)


CONFIGS = {"a": (["Om", "s8"], None), "b": (["Om", "s8", "omb"], None), "c": (["Om", "s8", "omb", "slnY"], None),
           "d": (["Om", "s8", "omb", "alpha", "slnY"], 0.04), "e": (["Om", "s8", "omb", "alpha", "slnY"], None)}
BOUNDS = {"omb": (0.3, 1.5), "alpha": (1.0, 2.6), "slnY": (1e-3, 1.0)}
STEPS = np.array([0.001, 0.001, 0.003, 0.003, 0.003])


class Fit:
    """Poisson likelihood of data d (already scaled to the survey) under model.counts(theta) * fsky (adapted from the
    step-2 verifier's vcmodel.Fit)."""

    def __init__(self, model, data, cfg, truth, fsky):
        self.m, self.d, self.truth, self.fsky = model, np.asarray(data, float), np.asarray(truth, float), fsky
        free, rel = CONFIGS[cfg]
        self.cfg = cfg
        self.free = [PN.index(p) for p in free]
        self.prior = None if rel is None else (2, float(truth[2]), rel * float(truth[2]))
        G = grid()
        bnd = dict(BOUNDS, Om=(float(G["OM"][0]) + 5e-4, float(G["OM"][-1]) - 5e-4), s8=(float(G["S8"][0]) + 5e-4, float(G["S8"][-1]) - 5e-4))
        self.lo = np.array([bnd[PN[i]][0] for i in self.free]); self.hi = np.array([bnd[PN[i]][1] for i in self.free])
        self.nev = 0

    def full(self, x):
        th = self.truth.copy(); th[self.free] = x
        return th

    def mu(self, x):
        self.nev += 1
        return self.m.counts(self.full(x), self.fsky)

    def lnL(self, x):
        x = np.asarray(x, float)
        if np.any(x < self.lo) or np.any(x > self.hi):
            return -np.inf
        mu = self.mu(x); d = self.d
        t = np.where(d > 0, d * np.log(np.where(d > 0, np.maximum(mu, 1e-300) / np.where(d > 0, d, 1.0), 1.0)), 0.0)
        L = float(np.sum(t - mu + d))
        if self.prior is not None:
            k, m_, s_ = self.prior
            L -= 0.5 * ((self.full(x)[k] - m_) / s_) ** 2
        return L

    def jac(self, x, steps):
        J = []
        for k in range(len(x)):
            e = np.zeros(len(x)); e[k] = steps[k]
            a = np.clip(x - e, self.lo, self.hi); b = np.clip(x + e, self.lo, self.hi)
            J.append(((self.mu(b) - self.mu(a)) / (b[k] - a[k])).ravel())
        return np.stack(J, 1)

    def fisher(self, x, steps):
        mu = self.mu(x).ravel(); J = self.jac(x, steps)
        ok = mu > 0
        F = J[ok].T @ (J[ok] / mu[ok, None])
        if self.prior is not None:
            k = self.free.index(self.prior[0]); F[k, k] += 1.0 / self.prior[2] ** 2
        return F, J, mu

    def sigmas(self, F):
        C = np.linalg.inv(F)
        s = np.sqrt(np.diag(C))
        a, b = self.free.index(0), self.free.index(1)
        S8 = S8_of(self.truth[0], self.truth[1])
        g = np.zeros(len(self.free)); g[a] = 0.5 * S8 / self.truth[0]; g[b] = S8 / self.truth[1]
        return s, float(np.sqrt(g @ C @ g)), g, C

    def fit_nm(self, F, x0=None, tol=1e-4, rounds=5, maxfev=3000, steps=None):
        from scipy.optimize import minimize
        xt = self.truth[self.free]
        sc = np.sqrt(np.diag(np.linalg.inv(F)))
        x_best = xt.copy() if x0 is None else np.clip(np.asarray(x0, float), self.lo, self.hi)

        def obj_factory(center, Fw):
            Lc = np.linalg.cholesky(Fw)
            Linv_T = np.linalg.inv(Lc).T
            to_x = lambda y: center + Linv_T @ y

            def f(y):
                x = to_x(y)
                xc = np.clip(x, self.lo, self.hi)
                return -self.lnL(xc) + 1e3 * float(np.sum(((x - xc) / sc) ** 2))
            return f, to_x
        info = []
        Fw = F
        for r in range(rounds):
            f, to_x = obj_factory(x_best, Fw)
            y0 = np.zeros(len(xt))
            f0 = f(y0)
            res = minimize(f, y0, method="Nelder-Mead",
                           options=dict(xatol=tol, fatol=1e-8, maxiter=maxfev, maxfev=maxfev, adaptive=True,
                                        initial_simplex=np.vstack([y0, 0.3 * np.eye(len(y0))])))
            moved = float(np.linalg.norm(res.x))
            info.append((int(res.nfev), bool(res.success), moved))
            if res.fun <= f0:
                x_best = np.clip(to_x(res.x), self.lo, self.hi)
            if moved < 10 * tol:
                break
            if steps is not None:
                try:
                    Fn, _, _ = self.fisher(x_best, steps)
                    np.linalg.cholesky(Fn)
                    Fw = Fn
                except np.linalg.LinAlgError:
                    Fw = F
        self.nm_info = info
        return x_best, self.lnL(x_best)

    def newton_polish(self, x, h_sig, iters=6):
        x = np.array(x, float)
        free_k = [k for k in range(len(x)) if (x[k] - self.lo[k]) > 3 * h_sig[k] and (self.hi[k] - x[k]) > 3 * h_sig[k]]
        L0 = self.lnL(x)
        for it in range(iters):
            n = len(free_k)
            if n == 0:
                break
            g = np.zeros(n); Hs = np.zeros((n, n))
            f0 = L0
            for a_, ka in enumerate(free_k):
                ea = np.zeros(len(x)); ea[ka] = h_sig[ka]
                fp, fm = self.lnL(x + ea), self.lnL(x - ea)
                g[a_] = (fp - fm) / (2 * h_sig[ka]); Hs[a_, a_] = (fp - 2 * f0 + fm) / h_sig[ka] ** 2
                for b_ in range(a_):
                    kb = free_k[b_]
                    eb = np.zeros(len(x)); eb[kb] = h_sig[kb]
                    Hs[a_, b_] = Hs[b_, a_] = (self.lnL(x + ea + eb) - self.lnL(x + ea - eb) - self.lnL(x - ea + eb)
                                               + self.lnL(x - ea - eb)) / (4 * h_sig[ka] * h_sig[kb])
            try:
                step = np.linalg.solve(-Hs, g)
            except np.linalg.LinAlgError:
                break
            xn = x.copy(); xn[free_k] += step
            Ln = self.lnL(xn)
            if not np.isfinite(Ln) or Ln < L0 - 1e-9:
                break
            x, L0 = xn, Ln
            if np.max(np.abs(step / h_sig[free_k])) < 1e-3:
                break
        return x, L0

    def deviance(self, x):
        mu = self.mu(x); d = self.d
        t = np.where(d > 0, d * np.log(np.where(d > 0, d / np.maximum(mu, 1e-300), 1.0)), 0.0)
        return float(2 * np.sum(mu - d + t))


def power_of(dev, ndof):
    from scipy.stats import chi2, ncx2
    return float(ncx2.sf(chi2.ppf(0.95, ndof), ndof, max(dev, 1e-9))) if dev > 1e-9 else 0.05
