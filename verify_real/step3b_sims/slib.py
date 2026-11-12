"""Step 3b verification, lens 'sims': own pipeline library on top of vlib3b.py (a copy of the step-3a independent
check's vlib3.py, written independently of the step-3a build).  NOTHING is imported from szsim/finder.py,
analysis/realistic/step3/ or analysis/realistic/step3b/.

What is new here (own code):
  * SZiFi-default masked covariance (read from SZiFi's source maps.diffusive_inpaint, circular_cosine_blend_adaptive,
    spec.correct_for_inpainting; periodic neighbours and holes because the tiles are periodic):
      holes = q > 4 detection discs (3 theta_est); every hole pixel starts at 0 and is replaced 100 times, row-major
      and in place, by the mean of its 8 neighbours; each 8-connected hole is blended back towards the original map
      with w = (1 + cos(pi d / R)) / 2 for 0 < d <= R (d = distance to the nearest unmasked pixel, R = sqrt(area/pi)),
      0 beyond; the l-filtered maps' binned spectra are divided by the unmasked fraction f in bins with centre > lsep.
  * finder wrapper returning the final S/N cube (float32) and both covariances;
  * virtual clusters: a cluster's filtered signal (angle-averaged response tables of the tile's final filter at the
    painter's ln-theta nodes, i.e. the same template mixture the painter uses) added to the final S/N maps of a tile in
    a window around it; the finder's peak rules (q > 4, strict maximum against the 4 axis neighbours, SZiFi's scale
    rules) applied in the window interior; credited if the cluster's own signal at the peak is >= 2 sigma; q_obs = the
    highest credited peak.  One placement is evaluated for a whole grid of q' at once (the same noise window).
  * realistic tiles: decomposition of the q > 5 detections (table ranking of all halos within 30', exact lattice sums
    where the ranking or the 2-sigma test is close), credit rule, merges, isolated copies.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "out", "numba_cache"))
sys.path.insert(0, HERE)
import vlib3b as V                                   # noqa: E402  (sets the thread variables before numpy)
import numpy as np                                    # noqa: E402
import numba as nb                                    # noqa: E402
import scipy.fft as sfft                              # noqa: E402
from scipy import ndimage                             # noqa: E402
from scipy.special import j0                          # noqa: E402

ROOT = V.ROOT
OUT = os.path.join(HERE, "out")
MODES = V.MODES
THETAS = V.THETAS
LTH_S = np.log(THETAS)
PIX, N, LBOX = V.PIX, V.N, V.LBOX
# kernel theta nodes: 0.3' x 2^((i-1)/3), i = 0..24 (0.238'-61', the build's node values, chosen for comparison)
TH_K = 0.3 * 2.0 ** ((np.arange(25) - 1) / 3.0)
# q' grid of the kernel: 0, 1, 1.5-30 (step 0.1), 30-3000 log-spaced
QG = np.unique(np.round(np.concatenate([[0.0, 1.0], np.arange(1.5, 30.0 + 1e-9, 0.1), np.geomspace(30.0, 3000.0, 81)]), 10))
EDGES = np.array(np.load(os.path.join(ROOT, "data_cosmo", "kernel_catalogue.npz"))["EDGES"], float)
LNTH0, DNODE = np.log(V.TH0_NODE), V.DNODE
QTH = 4.0


def threads_info():
    return {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                           "NUMBA_NUM_THREADS")}


# ============================================================================ SZiFi-default masked covariance
@nb.njit(cache=True)
def _diffuse(img, hy, hx, nsweep):
    n0, n1 = img.shape
    for s in range(nsweep):
        for k in range(hy.size):
            y = hy[k]; x = hx[k]
            ym = (y - 1) % n0; yp = (y + 1) % n0; xm = (x - 1) % n1; xp = (x + 1) % n1
            img[y, x] = (img[ym, xm] + img[y, xm] + img[yp, xm] + img[ym, x] + img[yp, x]
                         + img[ym, xp] + img[y, xp] + img[yp, xp]) / 8.0


def periodic_labels(holes):
    """8-connected components of a boolean image with periodic boundaries.  Returns (labels (0 = not a hole), area per
    label index)."""
    st = np.ones((3, 3), bool)
    lab, n = ndimage.label(holes, structure=st)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    n0, n1 = holes.shape
    # rows 0 and n0-1 (diagonals included), columns 0 and n1-1
    top, bot = lab[0], lab[-1]
    for d in (-1, 0, 1):
        a = top; b = np.roll(bot, d)
        m = (a > 0) & (b > 0)
        for u, v in zip(a[m], b[m]):
            union(u, v)
    lef, rig = lab[:, 0], lab[:, -1]
    for d in (-1, 0, 1):
        a = lef; b = np.roll(rig, d)
        m = (a > 0) & (b > 0)
        for u, v in zip(a[m], b[m]):
            union(u, v)
    root = np.array([find(a) for a in range(n + 1)])
    lab2 = root[lab]
    area = np.bincount(lab2.ravel(), minlength=n + 1).astype(float)
    area[0] = 0.0
    return lab2, area


def periodic_edt(holes, pad=160):
    """Distance of every hole pixel to the nearest non-hole pixel, periodic (wrap padding)."""
    hp = np.pad(holes, pad, mode="wrap")
    d = ndimage.distance_transform_edt(hp)
    return d[pad:-pad, pad:-pad]


def szifi_blend_weights(mask):
    """w per pixel (0 outside holes) of SZiFi's circular_cosine_blend_adaptive with inner_zero_frac = 0."""
    holes = mask < 0.5
    lab, area = periodic_labels(holes)
    R = np.sqrt(area / np.pi)[lab]
    d = periodic_edt(holes)
    w = np.zeros(mask.shape)
    ann = holes & (d > 0) & (d <= R)
    w[ann] = 0.5 * (1.0 + np.cos(np.pi * d[ann] / R[ann]))
    return w, holes


def masked_cov_szifi(Xc, mask, G, lsep=3000.0, nsweep=100, blend=True):
    """SZiFi's iteration-1 covariance with inpainting (defaults: diffusive, 100 sweeps, cosine blend, lsep 3000)."""
    f = float(mask.mean())
    holes = mask < 0.5
    hy, hx = np.nonzero(holes)                       # row-major order (as numpy.where)
    hy = hy.astype(np.int64); hx = hx.astype(np.int64)
    if blend:
        w, _ = szifi_blend_weights(mask)
    Xm = np.empty_like(Xc)
    for i in range(Xc.shape[1]):
        orig = sfft.ifft2(G.to_full(Xc[:, i]), workers=V.NTH).real
        img = orig * mask
        _diffuse(img, hy, hx, nsweep)
        if blend:
            img = np.where(holes, w * orig + (1.0 - w) * img, orig)
        Xm[:, i] = sfft.fft2(img, workers=V.NTH).ravel()[G.idx]
    C = V.binned_cov(Xm, G)
    centre = (np.arange(G.nbin) + 0.5) * V.BW
    C[centre > lsep] /= f
    return C, f


# ============================================================================ finder
def run_finder(Xc, G, method="whitened", keep_cube=False, lsep=3000.0):
    """Iteration 0 with the tile covariance; q > 4 detections masked (3 theta discs); covariance re-estimated with
    the whitened mask ('whitened', production) or SZiFi's inpainting ('szifi', lsep 3000; 'szifi12000');
    filter rerun once.  keep_cube: the final S/N maps (15, N, N) float32."""
    C0 = V.binned_cov(Xc, G)
    F0 = V.MMF(G, C=C0)
    det0 = V.detect(F0, F0.combine(Xc))
    sig0 = F0.sigma_am.copy()
    del F0
    mask, nmask = V.cluster_mask(det0, THETAS)
    if method == "whitened":
        C1, f = V.masked_cov(Xc, mask, G, Cref=C0, method="whitened")
    elif method == "szifi":
        C1, f = masked_cov_szifi(Xc, mask, G, lsep=lsep)
    else:
        raise ValueError(method)
    F1 = V.MMF(G, C=C1)
    D = F1.combine(Xc)
    cube = np.empty((len(THETAS), N, N), np.float32) if keep_cube else None
    det1 = detect32(F1, D, cube)
    out = dict(F1=F1, det0=det0, det1=det1, sig0=sig0, sig1=F1.sigma_am.copy(), fsky=f, nmask=nmask, C1=C1, mask=mask)
    if keep_cube:
        out["cube"] = cube
    return out


def detect32(filt, D, cube=None, qth=QTH):
    """vlib3's detect (rolling window of three S/N maps, SZiFi maxima_lomem rules); the maps are optionally written
    to a float32 cube (memory)."""
    nth = len(filt.thetas)
    J, IY, IX, Q = [], [], [], []
    prev, cur = None, filt.qmap(D, 0)
    for j in range(nth):
        nxt = filt.qmap(D, j + 1) if j + 1 < nth else cur
        if cube is not None:
            cube[j] = cur
        a, b = V.peaks_one_scale(prev if prev is not None else cur, cur, nxt, j, nth, qth)
        J.append(np.full(a.size, j, np.int64)); IY.append(a); IX.append(b); Q.append(cur[a, b])
        prev, cur = cur, nxt
    return dict(j=np.concatenate(J), iy=np.concatenate(IY), ix=np.concatenate(IX), q=np.concatenate(Q))


def sigma_finder(mode, theta, r15):
    """sigma_finder(theta) = r(theta) sigma_Y(theta), r linear in ln theta between the 15 scales, constant beyond."""
    theta = np.asarray(theta, float)
    return np.interp(np.log(theta), LTH_S, r15) * V.sigma_Y(mode, theta)


# ============================================================================ response tables
_J0 = {}


class RTab:
    """vlib3's ResponseTables (angle-averaged response of filter F to a unit-Y500 halo at the painter's ln-theta nodes,
    discrete-lattice Gauss-Legendre weights per covariance bin) on r = 0..(nr-1) x 0.02'."""

    def __init__(self, F, hn, nodes, J0mat):
        self.nodes = np.asarray(nodes, np.int64)
        aCa = np.einsum("li,lij,lj->l", hn.a, F.Cinv[hn.b], hn.a)
        th_n = V.TH0_NODE * np.exp(V.DNODE * self.nodes)
        Gn = V.g_cub(np.outer(th_n * V.AR, hn.l))
        self.R = np.empty((len(F.thetas), self.nodes.size, J0mat.shape[1]))
        for j in range(len(F.thetas)):
            wj = hn.w * F.gfun(hn.l * F.thetas[j] * V.AR) * aCa * F.sigma[j] ** 2
            self.R[j] = (Gn * wj[None, :]) @ J0mat
        self.n0 = int(self.nodes[0])
        self.dr = 0.02


def response_tables(F, hn, nodes, rmax):
    """angle-averaged response tables of filter F at painter nodes `nodes` for r = 0..rmax (step 0.02')."""
    nr = int(round(rmax / 0.02)) + 1
    key = (id(hn), nr)
    if key not in _J0:
        if len(_J0) > 1:
            _J0.clear()
        _J0[key] = j0(np.outer(hn.l, np.arange(nr) * 0.02 * V.AR))
    return RTab(F, hn, nodes, _J0[key])


def nodes_for(thetas, margin=2):
    i0, _ = V.lagrange4(np.log(np.asarray(thetas, float)))
    return np.arange(max(0, int(i0.min()) - margin), int(i0.max()) + 2 + margin)


# ============================================================================ virtual clusters
@nb.njit(cache=True)
def virt_eval(cube, y0, x0, lnth, yunit, R, n0, dr, lnth0, dnode, sig, w, qgrid, out, offs):
    """One virtual cluster at (y0, x0) [arcmin] of ln theta500 = lnth; Y = q' * yunit for every q' in qgrid.
    out[k] = highest credited peak (q > 4, local max in space and scale, own >= 2), -inf if none.
    offs[k] = distance [arcmin] of that peak from the cluster (diagnostic)."""
    nsc = cube.shape[0]; n = cube.shape[1]
    W = 2 * w + 1
    cy = int(np.floor(y0 / 0.5 + 0.5)); cx = int(np.floor(x0 / 0.5 + 0.5))
    S = np.empty((nsc, W, W)); Nw = np.empty((nsc, W, W)); Rr = np.empty((W, W))
    for a in range(W):
        iy = (cy + a - w) % n
        dy = (cy + a - w) * 0.5 - y0
        for b in range(W):
            ix = (cx + b - w) % n
            dx = (cx + b - w) * 0.5 - x0
            r = np.sqrt(dy * dy + dx * dx)
            Rr[a, b] = r
            for j in range(nsc):
                S[j, a, b] = yunit * V.resp_eval(R, j, n0, lnth, r, dr, lnth0, dnode) / sig[j]
                Nw[j, a, b] = cube[j, iy, ix]
    for k in range(qgrid.size):
        qp = qgrid[k]
        best = -np.inf; boff = -1.0
        for j in range(nsc):
            for a in range(1, W - 1):
                for b in range(1, W - 1):
                    own = qp * S[j, a, b]
                    if own < 2.0:
                        continue
                    v = Nw[j, a, b] + own
                    if v <= 4.0 or v <= best:
                        continue
                    if not (v > Nw[j, a - 1, b] + qp * S[j, a - 1, b] and v > Nw[j, a + 1, b] + qp * S[j, a + 1, b]
                            and v > Nw[j, a, b - 1] + qp * S[j, a, b - 1] and v > Nw[j, a, b + 1] + qp * S[j, a, b + 1]):
                        continue
                    if j == 0:
                        ok = v > 0.0 and (nsc == 1 or v >= Nw[1, a, b] + qp * S[1, a, b])
                    elif j == nsc - 1:
                        ok = v > Nw[j - 1, a, b] + qp * S[j - 1, a, b]
                    else:
                        ok = (v > Nw[j - 1, a, b] + qp * S[j - 1, a, b]) and (v >= Nw[j + 1, a, b] + qp * S[j + 1, a, b])
                    if ok:
                        best = v; boff = Rr[a, b]
        out[k] = best
        offs[k] = boff


def half_width_pix(theta):
    """window half width [pixels]: max(6', 0.2 theta + 2')."""
    return int(np.ceil(max(6.0, 0.2 * float(theta) + 2.0) / PIX))


def virt_many(cube, ys, xs, ths, yunits, RT, sig, qgrid, wmult=1.0):
    """q_obs (n, nq) of n virtual clusters, each over the whole q' grid (qgrid (nq,) or per-cluster (n, nq))."""
    n = len(ys)
    qg2 = np.atleast_2d(qgrid)
    out = np.empty((n, qg2.shape[1])); off = np.empty((n, qg2.shape[1]))
    for i in range(n):
        w = int(np.ceil(half_width_pix(ths[i]) * wmult))
        q = qg2[0] if qg2.shape[0] == 1 else qg2[i]
        virt_eval(cube, float(ys[i]), float(xs[i]), float(np.log(ths[i])), float(yunits[i]), RT.R, RT.n0, RT.dr,
                  LNTH0, DNODE, sig, w, np.ascontiguousarray(q, float), out[i], off[i])
    return out, off


# ============================================================================ realistic tiles: decomposition and credit
def decompose(det, F1, RT, P, hy, hx, hY, hth, qcut=5.0, rcut=30.0, K=3, tol_rank=0.97, tol_cp=0.1):
    """Primary (largest contributor) and its signal c_p / sigma at every detection with q > qcut.  Ranking by the
    angle-averaged tables (all halos within rcut); exact lattice sums of the top K where the top two are within
    tol_rank or c_p/sigma is within tol_cp of 2."""
    q = det["q"]
    s5 = np.flatnonzero(q > qcut)
    qy = det["iy"][s5] * PIX; qx = det["ix"][s5] * PIX; qj = det["j"][s5].astype(np.int64)
    lth = np.log(hth)
    cidx = np.full((s5.size, K), -1, np.int64); cval = np.zeros((s5.size, K)); csum = np.zeros(s5.size)
    V.table_topk(qy, qx, qj, np.full(s5.size, -1, np.int64), hy, hx, hY, lth, RT.R, RT.n0, RT.dr, LNTH0, DNODE,
                 rcut, LBOX, K, cidx, cval, csum)
    sig = F1.sigma_am
    EX = None
    nexact = 0; nchanged = 0
    for a in range(s5.size):
        c1 = cval[a, 0]; c2 = cval[a, 1] if K > 1 else -1e300
        cp = c1 / sig[qj[a]]
        if (c2 > tol_rank * c1) or (abs(cp - 2.0) < tol_cp):
            if EX is None:
                EX = V.Exact(F1, P)
            ex = np.full(K, -1e300)
            for k in range(K):
                h = cidx[a, k]
                if h < 0:
                    continue
                dy = (qy[a] - hy[h] + 0.5 * LBOX) % LBOX - 0.5 * LBOX
                dx = (qx[a] - hx[h] + 0.5 * LBOX) % LBOX - 0.5 * LBOX
                ex[k] = EX.contrib(int(qj[a]), hY[h], hth[h], dy, dx, template="nodes")
            o = np.argsort(-ex)
            if o[0] != 0:
                nchanged += 1
            cidx[a] = cidx[a, o]; cval[a] = ex[o]
            nexact += 1
    prim = cidx[:, 0]
    cps = np.where(prim >= 0, cval[:, 0] / sig[qj], 0.0)
    return dict(s5=s5, prim=prim, cp=cps, c2=np.where(cidx[:, 1] >= 0, cval[:, 1] / sig[qj], 0.0) if K > 1 else None,
                nexact=nexact, nchanged=nchanged)


def credit(det, dec, nh, merge="none", qcut=5.0):
    """q_obs per halo (n halos; -inf if not credited) under a merge rule; and the 'no real cluster' kept entries."""
    q = det["q"]
    keep = V.apply_merge(det["iy"] * PIX, det["ix"] * PIX, q, merge, qcut=qcut)       # indices into det (q > qcut)
    pos = {int(d): k for k, d in enumerate(dec["s5"])}
    qo = np.full(nh, -np.inf)
    nreal = 0; nnr = 0; nr_list = []
    for d in keep:
        k = pos[int(d)]
        p = dec["prim"][k]
        if p >= 0 and dec["cp"][k] >= 2.0:
            nreal += 1
            if q[d] > qo[p]:
                qo[p] = q[d]
        else:
            nnr += 1
            nr_list.append((float(q[d]), int(p), float(dec["cp"][k])))
    return qo, nreal, nnr, nr_list


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
