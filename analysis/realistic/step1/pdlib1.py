"""Step 1 helper: compound-Poisson (P(D)) blending machinery for a SIGNED filter kernel.

Generalises analysis/blending/blendlib.py + b03_analytic.py (which assume rho >= 0) to the compensated MMF:
  * rho table in the blendlib format (th, t, rho, sigY, R_BEAM) so that blendlib.rho_at can be reused for the
    trilinear interpolation; R_BEAM is used as the support margin: rho = 0 for s > S = 5 thA + 5 thB + R_BEAM.
    For the MMF the margin is large (default 150') and the variance missing beyond the support is measured
    against Parseval and added back as a Gaussian term.
  * signed rate histogram: x = rho Y_B / sigma_A binned in log|x| separately for x > 0 and x < 0.
  * signed FFT compound Poisson (negative contributions wrap around the periodic box: offset `off`).
  * per-(theta_A, q_A) blending probabilities, convention c=1 (map mean removed: shift by the untruncated
    mean mu of delta under the same kernel), dominance rule x_B < q_A (truncated compound Poisson),
    exactly as b03 (B2, B2neg, B3 = boosted, lost, satellites) plus the mean Y bias of detections.
Units: arcmin, Y500 in arcmin^2, x in units of sigma_Y(theta_A).
"""
import os, sys, time
import numpy as np
import numba as nb
from scipy.special import ndtr

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, f"{ROOT}/analysis/blending")
import blendlib as bl                                          # noqa: E402
from szsim import signal as sg, cosmo_hmf as ch, instrument as ins   # noqa: E402

X_CAP = 500.0
XPOS_EDGES = np.logspace(-8, np.log10(X_CAP), 2601)          # |x| bin edges (bin 0 = |x| < 1e-8)
NBIN = len(XPOS_EDGES)                                        # bins: 0 = tiny, 1..2600 = log bins, cap separate
QNODES = np.concatenate([np.arange(0.1, 20.0, 0.1), np.geomspace(20.0, 3000.0, 60)])
EPS = np.array([0.05, 0.1, 0.2, 0.5, 1.0])


# ============================================================================ rho table
_WM = None


def _init_worker(spec):
    global _WM
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    _WM = make_model(spec)


def make_model(spec):
    """spec = ('so', mode, lmin) or ('white',)."""
    if spec[0] == "white":
        return ins.MMFModel(ins.Instrument.white_y())
    return ins.MMFModel(ins.Instrument.so_lat(spec[1], lmin=spec[2]))


def _pair_job(args):
    i, j, a, b, S, t = args
    return i, j, _WM.cross_response(a, b, S * t ** 2)


def build_rho_table(spec, th, NU=401, R_extra=150.0, nproc=4):
    from multiprocessing import get_context
    m = make_model(spec)
    sig = m.sigma_Y_exact(th)
    t = np.linspace(0.0, 1.0, NU)
    NT = len(th)
    jobs = [(i, j, th[i], th[j], 5 * th[i] + 5 * th[j] + R_extra, t) for i in range(NT) for j in range(i, NT)]
    rho = np.zeros((NT, NT, NU))
    with get_context("spawn").Pool(nproc, initializer=_init_worker, initargs=(spec,)) as p:
        for i, j, r in p.imap_unordered(_pair_job, jobs, chunksize=8):
            rho[i, j] = r
            if j != i:
                rho[j, i] = r * sig[j] ** 2 / sig[i] ** 2      # rho_AB / sigma_A^2 symmetric
    return dict(th=th, t=t, rho=rho, sigY=sig, R_BEAM=R_extra)


# ============================================================================ signed rate histogram
@nb.njit(parallel=True, cache=False)
def _rate_signed(thA, sigA, thB, YB, cB, wB, tnodes, rho, lth0, dlth, R_BEAM, llo, dlog, nbin,
                 Wp, WXp, WX2p, Wn, WXn, WX2n, capp, capn):
    n = thB.shape[0]
    NU = tnodes.shape[0]
    nthr = nb.get_num_threads()
    C = Wp.shape[0]
    tWp = np.zeros((nthr, C, nbin)); tWXp = np.zeros_like(tWp); tWX2p = np.zeros_like(tWp)
    tWn = np.zeros_like(tWp); tWXn = np.zeros_like(tWp); tWX2n = np.zeros_like(tWp)
    tcp = np.zeros((nthr, C, 3)); tcn = np.zeros((nthr, C, 3))
    chunk = (n + nthr - 1) // nthr
    dt = tnodes[1] - tnodes[0]
    for tid in nb.prange(nthr):
        for k in range(tid * chunk, min(n, (tid + 1) * chunk)):
            S = 5.0 * thA + 5.0 * thB[k] + R_BEAM
            c = cB[k]
            for it in range(1, NU):
                t = tnodes[it]
                wt = 0.5 * dt if it == NU - 1 else dt
                dA = 4.0 * np.pi * S * S * t * t * t * wt
                s = S * t * t
                rr = bl.rho_at(rho, lth0, dlth, R_BEAM, thA, thB[k], s)
                x = rr * YB[k] / sigA
                w = wB[k] * dA
                ax = abs(x)
                if ax >= 500.0:
                    if x > 0:
                        tcp[tid, c, 0] += w; tcp[tid, c, 1] += w * x; tcp[tid, c, 2] += w * x * x
                    else:
                        tcn[tid, c, 0] += w; tcn[tid, c, 1] += w * x; tcn[tid, c, 2] += w * x * x
                    continue
                if ax < 1e-8:
                    b = 0
                else:
                    b = 1 + int((np.log10(ax) - llo) / dlog)
                    if b > nbin - 1:
                        b = nbin - 1
                if x >= 0:
                    tWp[tid, c, b] += w; tWXp[tid, c, b] += w * x; tWX2p[tid, c, b] += w * x * x
                else:
                    tWn[tid, c, b] += w; tWXn[tid, c, b] += w * x; tWX2n[tid, c, b] += w * x * x
    for tid in range(nthr):
        Wp += tWp[tid]; WXp += tWXp[tid]; WX2p += tWX2p[tid]
        Wn += tWn[tid]; WXn += tWXn[tid]; WX2n += tWX2n[tid]
        capp += tcp[tid]; capn += tcn[tid]


def rate_signed(thA, sigA, pop, tab):
    """Returns dict of (3, NBIN) arrays Wp, WXp, WX2p, Wn, WXn, WX2n and (3, 3) caps [w, wx, wx2] for x >= 500
    (capp) and x <= -500 (capn).  Note: the t = 0 node carries zero area and is skipped."""
    thB, YB, cB, wB = pop
    o = {k: np.zeros((3, NBIN)) for k in ("Wp", "WXp", "WX2p", "Wn", "WXn", "WX2n")}
    o["capp"] = np.zeros((3, 3)); o["capn"] = np.zeros((3, 3))
    llo = np.log10(XPOS_EDGES[0]); dlog = np.log10(XPOS_EDGES[1] / XPOS_EDGES[0])
    _rate_signed(float(thA), float(sigA), thB, YB, cB.astype(np.int64), wB, tab.t, *tab.args(), llo, dlog, NBIN,
                 o["Wp"], o["WXp"], o["WX2p"], o["Wn"], o["WXn"], o["WX2n"], o["capp"], o["capn"])
    return o


def moments(o, cl):
    """mean and variance (Campbell) of delta from the binned rates, classes cl, incl. caps."""
    cl = list(cl)
    mu = o["WXp"][cl].sum() + o["WXn"][cl].sum() + o["capp"][cl, 1].sum() + o["capn"][cl, 1].sum()
    var = o["WX2p"][cl].sum() + o["WX2n"][cl].sum() + o["capp"][cl, 2].sum() + o["capn"][cl, 2].sum()
    return mu, var


# ============================================================================ signed P(D)
class PDs:
    """FFT compound-Poisson distribution of delta = sum_i x_i for a signed binned rate.
    Grid: delta_j = j dd - off, j = 0..N-1 (periodic box of length N dd)."""

    def __init__(self, N=2 ** 18, dd=0.001, x_small=0.002, off=60.0):
        self.N, self.dd, self.xs, self.off = N, dd, x_small, off
        self.k = 2 * np.pi * np.fft.rfftfreq(N, d=dd)
        self.grid = np.arange(N) * dd - off

    def pdf(self, W, X, X2, extra_var=0.0):
        """W, X, X2: 1D arrays of (rate, rate*x, rate*x^2) per signed bin (already restricted / truncated).
        Bins whose mean |x| < x_small enter through their first two cumulants (Gaussian); extra_var adds a
        Gaussian term (e.g. variance beyond the kernel support).  Returns p_j (probability per cell)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            xb = np.where(W > 0, X / W, 0.0)
        keep = W > 0
        small = keep & (np.abs(xb) < self.xs)
        big = keep & ~small
        mu_s = X[small].sum(); v_s = X2[small].sum() + extra_var
        R = np.zeros(self.N)
        # [guard] contributions below -off would wrap around the periodic box and reappear as large POSITIVE
        # delta; clip them to -off (+1 cell): any x < -off already makes the cluster undetectable unless q_A > off.
        self.last_clipped = W[big][xb[big] < -self.off + self.dd].sum()
        f = np.maximum(xb[big], -self.off + self.dd) / self.dd
        i0 = np.floor(f).astype(np.int64); fr = f - i0
        np.add.at(R, i0 % self.N, W[big] * (1 - fr))
        np.add.at(R, (i0 + 1) % self.N, W[big] * fr)
        lam = W[big].sum()
        lnphi = (np.fft.rfft(R) - lam - 1j * self.k * (mu_s + self.off) - 0.5 * self.k ** 2 * (v_s + (2 * self.dd) ** 2))
        p = np.fft.irfft(np.exp(lnphi), n=self.N)
        return np.clip(p, 0, None)


def signed_arrays(o, cl, xcut=np.inf):
    """Concatenate positive and negative bins of classes cl; drop positive bins with mean x >= xcut
    (and the positive cap).  Returns W, X, X2 (1D) and Lambda(x >= xcut)."""
    cl = list(cl)
    Wp, Xp, X2p = o["Wp"][cl].sum(0), o["WXp"][cl].sum(0), o["WX2p"][cl].sum(0)
    Wn, Xn, X2n = o["Wn"][cl].sum(0), o["WXn"][cl].sum(0), o["WX2n"][cl].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xp = np.where(Wp > 0, Xp / Wp, 0.0)
    keep = xp < xcut
    lam_above = Wp[~keep].sum() + o["capp"][cl, 0].sum()
    # negative cap (x <= -500): include as a single bin at its mean (it is far outside any detection range)
    cn = o["capn"][cl].sum(0)
    W = np.concatenate([Wp[keep], Wn, [cn[0]]])
    X = np.concatenate([Xp[keep], Xn, [cn[1]]])
    X2 = np.concatenate([X2p[keep], X2n, [cn[2]]])
    return W, X, X2, lam_above


# ============================================================================ primary population
def primary_population(model, th_nodes, qmin, qmax=np.inf, zsub=2, msub=2, ngh=24, zlo=None, zhi=None):
    """Expected full-sky counts of halos (M>1e13) CIC-binned on (ln theta node, q node) with quadratic
    Lagrange weights in q (copy of b03_analytic.primary_population, with the model's sigma_Y).
    [arbiter addition] zlo, zhi: restrict to zlo <= z <= zhi with the exact partial-cell trapezoid weights of
    cosmo_hmf.HMFTable._weights (the per-bin H then sum to the full-range H); default (None, None): the original
    code path, unchanged."""
    T = ch.default_table()
    z = T.z[::zsub]; lnM = T.lnM[::msub]; d2N = T.d2N[::zsub, ::msub]
    if zlo is None and zhi is None:
        wz = np.full(len(z), z[1] - z[0]); wz[0] *= .5; wz[-1] *= .5
    else:
        wz = ch.HMFTable._weights(z, -np.inf if zlo is None else zlo, np.inf if zhi is None else zhi)
    wM = np.full(len(lnM), lnM[1] - lnM[0]); wM[0] *= .5; wM[-1] *= .5
    x, wg = np.polynomial.hermite_e.hermegauss(ngh); wg /= wg.sum()
    NT = len(th_nodes); lth0 = np.log(th_nodes[0]); dlth = np.log(th_nodes[1] / th_nodes[0])
    H = np.zeros((NT, len(QNODES)))
    M = np.exp(lnM)
    for iz, zz in enumerate(z):
        if wz[iz] == 0:
            continue
        th = model.theta500(M, zz)
        q0 = model.Y500_med(M, zz) / model.sigma_Y(th)
        q = (q0[:, None] * np.exp(model.sigma_lnY * x)[None, :]).ravel()
        w = ((d2N[iz] * wz[iz] * wM)[:, None] * wg[None, :]).ravel()
        tt = np.repeat(th, ngh)
        s = (q > qmin) & (q < qmax)
        q, w, tt = q[s], w[s], tt[s]
        fa = np.clip((np.log(tt) - lth0) / dlth, 0, NT - 1.000001)
        ia = fa.astype(int); wa = fa - ia
        k = np.clip(np.searchsorted(QNODES, q) - 1, 0, len(QNODES) - 2)
        j0 = np.clip(np.where(q - QNODES[k] < QNODES[k + 1] - q, k - 1, k), 0, len(QNODES) - 3)
        x0, x1, x2 = QNODES[j0], QNODES[j0 + 1], QNODES[j0 + 2]
        L = [(q - x1) * (q - x2) / ((x0 - x1) * (x0 - x2)), (q - x0) * (q - x2) / ((x1 - x0) * (x1 - x2)),
             (q - x0) * (q - x1) / ((x2 - x0) * (x2 - x1))]
        for da, fa_ in ((0, 1 - wa), (1, wa)):
            for db in range(3):
                np.add.at(H, (ia + da, j0 + db), w * fa_ * L[db])
    return H


# ============================================================================ blending probabilities
def node_probs(o, cl, qa, qcuts, mu_tot, pd_small, pd_big, extra_var=0.0):
    """For one (theta_A, q_A) node: probabilities (not yet x P_dom) for each q_cut of
    det, B3 (boosted), lost, B2(eps), B2neg(eps), and E[(delta-mu) 1(det)], plus P_dom.
    Convention c=1: q_b = q_A + n + delta - mu_tot."""
    xcut = min(qa, X_CAP)
    W, X, X2, lam_above = signed_arrays(o, cl, xcut)
    Pdom = np.exp(-lam_above)
    pd_ = pd_small if xcut < 20 else pd_big
    p = pd_.pdf(W, X, X2, extra_var=extra_var)
    p /= p.sum()
    g = pd_.grid
    sh = mu_tot
    out = {}
    tail = np.concatenate([np.cumsum(p[::-1])[::-1], [0.0]])
    jmax = len(g)

    def jidx(d):
        return int(min(max(np.floor((d + pd_.off) / pd_.dd) + 1, 0), jmax))

    j0 = jidx(sh)
    for qc in qcuts:
        phi = ndtr(qa + g - sh - qc)
        pg = p * phi
        part = np.concatenate([np.cumsum(pg[::-1])[::-1], [0.0]])
        Iall = part[0]
        Ipos = part[j0]
        phi0 = ndtr(qa - qc)
        r = dict(det=Iall,
                 b3=max(Ipos - phi0 * tail[j0], 0.0),
                 lost=max(phi0 * (1 - tail[j0]) - (Iall - Ipos), 0.0),
                 b2=np.array([part[jidx(sh + e * qa)] for e in EPS]),
                 b2n=np.array([Iall - part[jidx(sh - e * qa)] for e in EPS]),
                 mean=np.sum(pg * (g - sh)))
        out[qc] = r
    return Pdom, out


def node_probs_ext(o, cl, qa, qcuts, mu_tot, pd_small, pd_big, extra_var=0.0, fdom=1.0):
    """[arbiter addition, s09] node_probs generalised:
      * dominance rule x_B < fdom * q_A.  fdom = 1 reproduces node_probs exactly (same operations).
        fdom = np.inf: NO dominance rule (every primary counted; the P(D) is untruncated up to the x >= X_CAP cap,
        whose events, rate <= 1e-5 per primary, are dropped).
      * det2 = sum_j p_j Phi_j^2 per q_cut (not yet x P_dom): E[(1_dom Phi(q_b - q_cut))^2] = P_dom det2, the
        second moment needed for the per-sky (Poisson-primary) scatter of the blended counts.
    Returns (P_dom, {q_cut: dict(det, b3, lost, b2, b2n, mean, det2)})."""
    xcut = min(fdom * qa, X_CAP)
    W, X, X2, lam_above = signed_arrays(o, cl, xcut)
    Pdom = np.exp(-lam_above) if np.isfinite(fdom) else 1.0
    pd_ = pd_small if xcut < 20 else pd_big
    p = pd_.pdf(W, X, X2, extra_var=extra_var)
    p /= p.sum()
    g = pd_.grid
    sh = mu_tot
    out = {}
    tail = np.concatenate([np.cumsum(p[::-1])[::-1], [0.0]])
    jmax = len(g)

    def jidx(d):
        return int(min(max(np.floor((d + pd_.off) / pd_.dd) + 1, 0), jmax))

    j0 = jidx(sh)
    for qc in qcuts:
        phi = ndtr(qa + g - sh - qc)
        pg = p * phi
        part = np.concatenate([np.cumsum(pg[::-1])[::-1], [0.0]])
        Iall = part[0]
        Ipos = part[j0]
        phi0 = ndtr(qa - qc)
        r = dict(det=Iall,
                 b3=max(Ipos - phi0 * tail[j0], 0.0),
                 lost=max(phi0 * (1 - tail[j0]) - (Iall - Ipos), 0.0),
                 b2=np.array([part[jidx(sh + e * qa)] for e in EPS]),
                 b2n=np.array([Iall - part[jidx(sh - e * qa)] for e in EPS]),
                 mean=np.sum(pg * (g - sh)),
                 det2=np.sum(pg * phi))
        out[qc] = r
    return Pdom, out
