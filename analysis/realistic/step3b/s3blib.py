"""Step 3b helpers: finder variants, per-tile noise normalisation, virtual isolated clusters on noise-only tiles, the
analyst's no-blending selection kernel K(theta500, q'; e), and the per-halo tables of the realistic tiles.

New code; it imports (never modifies) szsim.finder and step 3a's s3lib.

Finder variants (szsim.finder.run_finder; the masked covariance of the one masking iteration):
  prod        'whitened' (step 3a's production finder; its realistic-tile runs are step 3a's data_real/step3/tiles)
  szifi3000   SZiFi's own estimate with its defaults: diffusive inpainting (100 sweeps), cosine blend, binned spectra above
              l_sep = 3000 divided by the unmasked fraction (step 3a's tiles_szifi for pids 0-99)
  szifi12000  the same with l_sep = 12000 (SZiFi's SO example script)

Units.  q' = Y500 / sigma_finder(theta500) with sigma_finder(theta) = r_tile(theta) sigma_Y(theta), r_tile = the tile's
sigma_finder / sigma_Y at the 15 filter scales after the iteration, linear in ln theta between the scales and constant
beyond them (np.interp), sigma_Y = step 1's MMF noise (s3lib.sigma_Y).  The same convention for realistic tiles,
noise-only tiles and injection tiles.

Virtual isolated cluster (virtual_credit): its filtered response through the tile's final filter, from step 3a's angle-
averaged lattice response tables (szsim.finder.response_tables, template = the painter's node mixture), is added to
the noise-only S/N cube in a window around it; the finder's peak rules (szsim.finder.find_peaks: q > 4, strict maximum
along both map axes, SZiFi's scale rules) are applied inside the window; a peak is credited when the cluster's own
filtered S/N there is >= 2 (t05's 'real cluster' rule: c_p >= 2 sigma); q_obs = the highest credited peak.
"""
import os, sys, time, resource
import numpy as np
import numba as nb

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
for _p in (f"{ROOT}/analysis/realistic/step3", ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import s3lib as S3                                 # noqa: E402
F = S3.F

D3 = os.path.join(ROOT, "data_real", "step3")
OUT = os.path.join(ROOT, "data_real", "step3b")
FIG = os.path.join(ROOT, "figures", "realistic")
MODES = ("baseline", "goal")
VARIANTS = {
    "prod": dict(mask_method="whitened", mask_kw={}),
    "szifi3000": dict(mask_method="szifi", mask_kw=dict(lsep=3000.0, blend=True, n_inpaint=100)),
    "szifi12000": dict(mask_method="szifi", mask_kw=dict(lsep=12000.0, blend=True, n_inpaint=100)),
}
VLAB = {"prod": "production (whitened mask)", "szifi3000": "SZiFi defaults (l_sep 3000, blend)",
        "szifi12000": "SZiFi SO setting (l_sep 12000, blend)"}
MERGES = ("none", "closest1", "fof10")
SEED_CMB, SEED_NOISE = S3.SEED_CMB, S3.SEED_NOISE
EDGES = np.load(os.path.join(ROOT, "data_cosmo", "kernel_catalogue.npz"))["EDGES"]      # step 2's 19 bin edges
Q_TH = F.Q_TH                       # 4: the finder's detection threshold
CRED_MIN = 2.0                      # credited: own filtered signal >= 2 sigma
# kernel theta nodes: 0.3 x 2^((k-1)/3), k = 0..24: 0.238'-61', every injected size of t03 / a02 is a node
TH_K = 0.3 * 2.0 ** ((np.arange(25) - 1) / 3.0)
# q' nodes of the virtual clusters (discrete); above Q_MODEL the kernel is a Gaussian model fitted to log-uniform q'
QN_LO = np.array([1.0, 1.5, 1.75])
QN_MID = np.round(np.arange(2.0, 10.0 + 1e-9, 0.125), 6)
QN_HI = np.round(np.arange(10.5, 30.0 + 1e-9, 0.5), 6)
Q_NODES = np.concatenate([QN_LO, QN_MID, QN_HI])
Q_LOGU = (25.0, 400.0)
Q_MODEL = 27.5


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def threads(n=1):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(n)


def sigma_Y(mode, theta):
    return S3.sigma_Y(mode, theta)


def r_of(sig15, mode):
    """r = sigma_finder / sigma_Y at the 15 scales."""
    return np.asarray(sig15, float) / sigma_Y(mode, F.THETAS15)


def sigma_finder(sig15, mode, theta):
    """The tile's sigma_finder(theta) [arcmin^2]: r linear in ln theta between the scales (constant beyond) x sigma_Y."""
    theta = np.asarray(theta, float)
    return np.interp(np.log(theta), np.log(F.THETAS15), r_of(sig15, mode)) * sigma_Y(mode, theta)


def tile_dir(variant):
    """Realistic-tile finder outputs (t04 layout) per variant: list of (pid range, folder)."""
    if variant == "prod":
        return {p: os.path.join(D3, "tiles") for p in range(324)}
    if variant == "szifi3000":
        return {p: (os.path.join(D3, "tiles_szifi") if p < 100 else os.path.join(OUT, "tiles_szifi3000")) for p in range(324)}
    return {p: os.path.join(OUT, f"tiles_{variant}") for p in range(324)}


_BAS = {}


def basis():
    if "b" not in _BAS:
        _BAS["tables"] = F.build_paint_tables()
        _BAS["b"] = F.ResponseBasis(F.grid(), _BAS["tables"])
    return _BAS["b"], _BAS["tables"]


def table_args(Rn, Rf, bas):
    return (Rn, Rf, bas.lnth0, bas.dln, F.R_SWITCH, bas.rn[1] - bas.rn[0], F.R_FAR_MIN, bas.rf[1] - bas.rf[0], F.LBOX)


def vargs(Rn, Rf, A, targs):
    """Arguments of virtual_credit / own_grid after the per-cluster ones: Rn, Rf, A, lnth0, dln, lnk0, dlnk, rswitch,
    drn, rf_min, drf, dra, box."""
    return (Rn, Rf, A, targs[2], targs[3], float(np.log(TH_K[0])), float(np.log(TH_K[1] / TH_K[0])), targs[4], targs[5],
            targs[6], targs[7], float(R_ANISO[1] - R_ANISO[0]), targs[8])


def filter_tables(filt, bas, G=None):
    """(Rn, Rf, A, targs) of a final filter: angle-averaged response tables and lattice harmonics."""
    Rn, Rf = F.response_tables(filt, bas)
    return Rn, Rf, aniso_tables(filt, bas, G), table_args(Rn, Rf, bas)


# ============================================================================ virtual isolated clusters
# Lattice anisotropy.  The tables Rn, Rf are the angle average of the filter's response.  On the square lattice of
# Fourier modes the response also has cos(m phi) terms, m = 4, 8, 12, ... (Jacobi-Anger: cos(k.D) = J0(kr) +
# 2 sum_{m = 4, 8, ...} J_m(kr) cos(m (phi_k - phi_D)); the sin terms cancel by the lattice's reflection symmetry):
#   R(D) = R0(r) + sum_m cos(m phi_D) A_m(r),  A_m(r) = 2 sum_k c(k) J_m(|k| r) cos(m phi_k),
# c(k) = w_k tau_j(k) aCa(k) G(k) / norm_j (as the tables).  A_m is summed exactly over the |k| shells with
# |k| <= LCUT_ANISO for m = 4, 8, 12 at the 25 kernel nodes (the painter's node mixture) and interpolated linearly in
# ln theta between them (clamped) and in r (0-40', 0.05').  phi = atan2(dy, dx), D = pixel - cluster.
LCUT_ANISO = 3000.0
R_ANISO = np.round(np.arange(0.0, 40.0 + 1e-9, 0.05), 6)
M_ANISO = (4, 8, 12)
_AJ = {}


def aniso_bessel(G=None):
    """Shell weights W_m = sum_shell w cos(m phi_k), Bessel matrices J_m(L r) for the shells with L <= LCUT_ANISO."""
    if "J" not in _AJ:
        from scipy.special import jv
        G = G or F.grid()
        ns = G.shell_n2.size
        sel = np.flatnonzero(G.shell_L <= LCUT_ANISO)
        phi = np.arctan2(G.m_k0, G.m_k1)
        W = np.stack([np.bincount(G.m_shell, weights=G.m_w * np.cos(m * phi), minlength=ns)[sel] for m in M_ANISO])
        first = np.full(ns, -1, np.int64)
        first[G.m_shell[::-1]] = np.arange(G.m_shell.size)[::-1]
        L = G.shell_L[sel]
        J = np.stack([jv(m, np.outer(L, R_ANISO * F.AR)) for m in M_ANISO])          # (3, nshell, nr)
        _AJ.update(sel=sel, W=W, first=first[sel], L=L, J=J)
    return _AJ


def aniso_tables(filt, bas, G=None):
    """A[m, j, k, r] for m in M_ANISO, the 15 scales, the 25 kernel nodes TH_K and r in R_ANISO (dimensionless, as R0)."""
    G = G or F.grid()
    d = aniso_bessel(G)
    L = d["L"]
    aCa = filt.aCa[d["first"]]
    Gk = []
    for th in TH_K:
        i0, w = F.theta_node(np.log(th), bas.lnth0, bas.dln, len(bas.th_nodes))
        Gk.append(((1 - w) * F.gt(L * bas.th_nodes[i0] * F.AR) + w * F.gt(L * bas.th_nodes[i0 + 1] * F.AR)) / G.Om)
    Gk = np.array(Gk)                                                                  # (nK, ns)
    tau = np.array([filt.gfun(L * t * F.AR) / G.Om / filt.norm[j] for j, t in enumerate(filt.thetas)])   # (nsc, ns)
    nsc, nK, nr = len(filt.thetas), TH_K.size, R_ANISO.size
    A = np.empty((len(M_ANISO), nsc, nK, nr))
    for a in range(len(M_ANISO)):
        C = 2.0 * d["W"][a][None, None, :] * tau[:, None, :] * aCa[None, None, :] * Gk[None, :, :]
        A[a] = (C.reshape(-1, L.size) @ d["J"][a]).reshape(nsc, nK, nr)
    return A


@nb.njit(cache=True, inline="always")
def _own(Rn, Rf, A, j, i0, wt, k0, wk, r, c4, c8, c12, rswitch, drn, rf_min, drf, dra):
    """Own response (Y-hat per unit Y500) at distance r, cos(m phi) = c4, c8, c12: angle-averaged table + harmonics."""
    v = F._resp_eval(Rn, Rf, j, i0, wt, r, rswitch, drn, rf_min, drf)
    u = r / dra
    ia = int(u)
    if ia < A.shape[3] - 1:
        fa = u - ia
        a0 = (1 - fa) * A[0, j, k0, ia] + fa * A[0, j, k0, ia + 1]
        a1 = (1 - fa) * A[0, j, k0 + 1, ia] + fa * A[0, j, k0 + 1, ia + 1]
        v += c4 * ((1 - wk) * a0 + wk * a1)
        a0 = (1 - fa) * A[1, j, k0, ia] + fa * A[1, j, k0, ia + 1]
        a1 = (1 - fa) * A[1, j, k0 + 1, ia] + fa * A[1, j, k0 + 1, ia + 1]
        v += c8 * ((1 - wk) * a0 + wk * a1)
        a0 = (1 - fa) * A[2, j, k0, ia] + fa * A[2, j, k0, ia + 1]
        a1 = (1 - fa) * A[2, j, k0 + 1, ia] + fa * A[2, j, k0 + 1, ia + 1]
        v += c12 * ((1 - wk) * a0 + wk * a1)
    return v


@nb.njit(cache=True, inline="always")
def _kidx(lnth, lnk0, dlnk, nK):
    t = (lnth - lnk0) / dlnk
    if t < 0.0:
        t = 0.0
    if t > nK - 1.000001:
        t = nK - 1.000001
    k0 = int(t)
    return k0, t - k0


@nb.njit(cache=True, inline="always")
def _harm(ddy, ddx, r):
    if r <= 0.0:
        return 0.0, 0.0, 0.0
    c = ddx / r
    s = ddy / r
    c2 = c * c - s * s
    s2 = 2.0 * c * s
    c4 = 2.0 * c2 * c2 - 1.0
    s4 = 2.0 * s2 * c2
    c8 = 2.0 * c4 * c4 - 1.0
    s8 = 2.0 * s4 * c4
    return c4, c8, c8 * c4 - s8 * s4


@nb.njit(cache=True)
def own_grid(Y, py, px, lnth, cy, cx, w, sig, Rn, Rf, A, lnth0, dln, lnk0, dlnk, rswitch, drn, rf_min, drf, dra, box, pix):
    """Own S/N of one cluster on the pixels (cy + a, cx + b), |a|, |b| <= w, all scales (accuracy check)."""
    nsc = Rn.shape[0]
    L = 2 * w + 1
    out = np.empty((nsc, L, L))
    i0, wt = F.theta_node(lnth, lnth0, dln, Rn.shape[1])
    k0, wk = _kidx(lnth, lnk0, dlnk, A.shape[2])
    for a in range(L):
        ddy = ((cy + a - w) * pix - py + 0.5 * box) % box - 0.5 * box
        for b in range(L):
            ddx = ((cx + b - w) * pix - px + 0.5 * box) % box - 0.5 * box
            r = np.sqrt(ddy * ddy + ddx * ddx)
            c4, c8, c12 = _harm(ddy, ddx, r)
            for j in range(nsc):
                out[j, a, b] = Y * _own(Rn, Rf, A, j, i0, wt, k0, wk, r, c4, c8, c12, rswitch, drn, rf_min, drf, dra) / sig[j]
    return out


@nb.njit(cache=True)
def virtual_credit(qn, sig, cy, cx, py, px, Y, lnth, W, Rn, Rf, A, lnth0, dln, lnk0, dlnk, rswitch, drn, rf_min, drf,
                   dra, box, pix, qth, cmin, out_q, out_j, out_dy, out_dx, out_own, out_nc, out_q0):
    """Virtual isolated clusters c: positions (py, px) [arcmin], window centre pixel (cy, cx), Y [arcmin^2], ln theta500,
    window half width W [pixels].  S/N cube in the window = qn (noise-only S/N cube of the final filter) + own S/N
    Y R_j(D) / sigma_j (angle-averaged tables + lattice harmonics).  Peaks: the finder's rules (find_peaks) on the
    window interior, neighbours from the window (1-pixel margin).  Credited: own S/N at the peak >= cmin.
    out_q: highest credited peak (-inf if none); out_j, out_dy, out_dx: its scale and pixel offset from (cy, cx);
    out_own: own S/N there; out_nc: number of credited peaks; out_q0: own S/N at the window centre, best scale."""
    nsc, n0, n1 = qn.shape
    nthn = Rn.shape[1]
    nK = A.shape[2]
    wmax = 0
    for c in range(W.size):
        if W[c] > wmax:
            wmax = W[c]
    Lm = 2 * wmax + 3
    own = np.empty((nsc, Lm, Lm))
    tot = np.empty((nsc, Lm, Lm))
    for c in range(cy.size):
        w = W[c]
        L = 2 * w + 3
        i0, wt = F.theta_node(lnth[c], lnth0, dln, nthn)
        k0, wk = _kidx(lnth[c], lnk0, dlnk, nK)
        for a in range(L):
            gy = cy[c] + a - w - 1
            ddy = (gy * pix - py[c] + 0.5 * box) % box - 0.5 * box
            iy = gy % n0
            for b in range(L):
                gx = cx[c] + b - w - 1
                ddx = (gx * pix - px[c] + 0.5 * box) % box - 0.5 * box
                ix = gx % n1
                r = np.sqrt(ddy * ddy + ddx * ddx)
                c4, c8, c12 = _harm(ddy, ddx, r)
                for j in range(nsc):
                    o = Y[c] * _own(Rn, Rf, A, j, i0, wt, k0, wk, r, c4, c8, c12, rswitch, drn, rf_min, drf, dra) / sig[j]
                    own[j, a, b] = o
                    tot[j, a, b] = qn[j, iy, ix] + o
        q0 = -1e300
        for j in range(nsc):
            if own[j, w + 1, w + 1] > q0:
                q0 = own[j, w + 1, w + 1]
        best = -np.inf
        bj = -1
        by = 0
        bx = 0
        bo = 0.0
        nc = 0
        for j in range(nsc):
            for a in range(1, L - 1):
                for b in range(1, L - 1):
                    v = tot[j, a, b]
                    if v <= qth:
                        continue
                    if not (v > tot[j, a + 1, b] and v > tot[j, a - 1, b] and v > tot[j, a, b + 1] and v > tot[j, a, b - 1]):
                        continue
                    if j == 0:
                        ok = (v > 0.0) and (nsc == 1 or v >= tot[1, a, b])
                    elif j == nsc - 1:
                        ok = v > tot[j - 1, a, b]
                    else:
                        ok = (v > tot[j - 1, a, b]) and (v >= tot[j + 1, a, b])
                    if not ok:
                        continue
                    if own[j, a, b] < cmin:
                        continue
                    nc += 1
                    if v > best:
                        best = v
                        bj = j
                        by = a - w - 1
                        bx = b - w - 1
                        bo = own[j, a, b]
        out_q[c] = best
        out_j[c] = bj
        out_dy[c] = by
        out_dx[c] = bx
        out_own[c] = bo
        out_nc[c] = nc
        out_q0[c] = q0


def window_pix(theta, wtab_th, wtab_r, wmin=6.0, scale=1.0):
    """Window half width [pixels] for clusters of size theta: max(wmin, scale x r_w(theta)), r_w tabulated per kernel
    node (linear in ln theta, constant beyond)."""
    r = np.interp(np.log(np.asarray(theta, float)), np.log(wtab_th), wtab_r)
    return np.ceil(np.maximum(wmin, scale * r) / F.PIX).astype(np.int64)


def run_virtual(qn, sig, filt_tabs, y, x, Y, theta, Wpix, chunk=4000):
    """Virtual isolated clusters on one S/N cube; filt_tabs = (Rn, Rf, A, targs).  Returns dict of per-cluster arrays."""
    Rn, Rf, A, targs = filt_tabs
    n = y.size
    out = dict(q=np.full(n, -np.inf), j=np.full(n, -1, np.int64), dy=np.zeros(n, np.int64), dx=np.zeros(n, np.int64),
               own=np.zeros(n), nc=np.zeros(n, np.int64), q0=np.zeros(n))
    cy = np.rint(y / F.PIX).astype(np.int64) % F.N
    cx = np.rint(x / F.PIX).astype(np.int64) % F.N
    lnth = np.log(np.asarray(theta, float))
    order = np.argsort(Wpix, kind="stable")             # similar windows together (one scratch allocation per chunk)
    for a in range(0, n, chunk):
        s = order[a:a + chunk]
        oq = np.empty(s.size); oj = np.empty(s.size, np.int64); ody = np.empty(s.size, np.int64)
        odx = np.empty(s.size, np.int64); oo = np.empty(s.size); onc = np.empty(s.size, np.int64); oq0 = np.empty(s.size)
        virtual_credit(qn, sig, cy[s], cx[s], y[s], x[s], Y[s], lnth[s], Wpix[s], *vargs(Rn, Rf, A, targs), F.PIX, Q_TH,
                       CRED_MIN, oq, oj, ody, odx, oo, onc, oq0)
        out["q"][s] = oq; out["j"][s] = oj; out["dy"][s] = ody; out["dx"][s] = odx; out["own"][s] = oo
        out["nc"][s] = onc; out["q0"][s] = oq0
    return out


def noise_only_filter(pid, mode, variant, G=None, X=None):
    """The variant's finder on the noise-only version of realistic tile pid (CMB + noise from the tile's seeds).
    Returns (its, X): its[-1] holds the final filter and the S/N cube ('q', keep_cube=True)."""
    G = G or F.grid()
    if X is None:
        X = F.sim_cmb(SEED_CMB + pid, G) + F.sim_noise(SEED_NOISE + pid, mode, G)
    v = VARIANTS[variant]
    its = F.run_finder(X, G, keep_cube=True, mask_method=v["mask_method"], mask_kw=v["mask_kw"])
    return its, X


# ============================================================================ kernel tables
def pava_increasing(y, w):
    """Weighted isotonic (non-decreasing) regression, pool adjacent violators."""
    y = np.asarray(y, float); w = np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            ww = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / ww if ww > 0 else 0.5 * (v1 + v2)); wts.append(ww); cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


class Kernel:
    """K(theta500, q'; e) on (TH, QA, EDGES): linear in ln theta between nodes (clamped), linear in q' (clamped to the
    grid; QA[0] = 0 with K = 0)."""

    def __init__(self, th, qa, K, edges=EDGES):
        self.th, self.qa, self.K, self.E = np.asarray(th, float), np.asarray(qa, float), np.asarray(K, float), np.asarray(edges, float)
        self.lth = np.log(self.th)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["th"], d["QA"], d["K"], d["EDGES"])

    def __call__(self, theta, q, e_idx=None):
        """K at (theta_k, q_k) for the edges e_idx (default all): (n, ne)."""
        theta = np.asarray(theta, float); q = np.asarray(q, float)
        ei = np.arange(self.E.size) if e_idx is None else np.atleast_1d(e_idx)
        x = np.clip(np.log(theta), self.lth[0], self.lth[-1])
        i = np.clip(np.searchsorted(self.lth, x) - 1, 0, self.th.size - 2)
        w = (x - self.lth[i]) / (self.lth[i + 1] - self.lth[i])
        qq = np.clip(q, self.qa[0], self.qa[-1])
        k = np.clip(np.searchsorted(self.qa, qq) - 1, 0, self.qa.size - 2)
        u = (qq - self.qa[k]) / (self.qa[k + 1] - self.qa[k])
        out = np.empty((q.size, ei.size))
        for m, e in enumerate(ei):
            Ke = self.K[:, :, e]
            a = (1 - u) * Ke[i, k] + u * Ke[i, k + 1]
            b = (1 - u) * Ke[i + 1, k] + u * Ke[i + 1, k + 1]
            out[:, m] = (1 - w) * a + w * b
        return out
