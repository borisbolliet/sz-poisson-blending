"""The verifier's own helpers for reading SZiFi run files and step-3a outputs (lens 'indep'; numpy and scipy only).

Positions are in 'index units' of the 1200 x 1200 array (axis 0 = the finder's y, axis 1 = its x); the zero-padded
tile edges sit at -0.5 and 1199.5.  SZiFi's S/N at pixel n belongs to the point n - 1/2 (measured in j04 (a)), so a
SZiFi detection at pixel (i, j) is placed at (i - 1/2, j - 1/2); a finder detection at (iy, ix) at (iy, ix).
Interior: distance to every edge >= 9' (18 pixels), half open: positions in [17.5, 1181.5) on both axes, 1164 samples
per axis for both catalogues, 582' x 582' = 94.09 deg^2.
"""
import os
import numpy as np
from scipy.spatial import cKDTree

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
BRUNS = os.path.join(ROOT, "data_real", "step3_szifi_real", "runs")
MRUNS = os.path.join(OUT, "runs")
D3 = os.path.join(ROOT, "data_real", "step3")
N, PIXA, INT = 1200, 0.5, 9.0
AREA = ((N * PIXA - 2 * INT) / 60.0) ** 2           # deg^2 of interior per tile
TH = np.geomspace(0.5, 15.0, 15)
MODES = ("baseline", "goal")
I05, I445, I15 = 0, 9, 14
SC6 = [0, 3, 6, 9, 12, 14]                           # 0.50 1.04 2.15 4.45 9.23 15.00'


def load(fn, keys=None):
    with np.load(fn) as d:
        return {k: d[k] for k in (keys or d.files) if k in d.files}


def run_path(src, cfg, mode, kind, pid):
    return os.path.join(BRUNS if src == "build" else MRUNS, f"{cfg}_{mode}_{kind}_{pid:03d}.npz")


def run_pids(src, cfg, mode, kind):
    d = BRUNS if src == "build" else MRUNS
    pre = f"{cfg}_{mode}_{kind}_"
    return sorted(int(f[len(pre):len(pre) + 3]) for f in os.listdir(d) if f.startswith(pre) and f.endswith(".npz")
                  and len(f) == len(pre) + 7)


def szifi_cat(r, k, shift=0.5):
    jt = r[f"it{k}_jt"] if f"it{k}_jt" in r else r[f"it{k}_j_theta"]
    return dict(q=np.asarray(r[f"it{k}_q"], float), py=r[f"it{k}_i"] - shift, px=r[f"it{k}_j"] - shift,
                jt=np.asarray(jt, np.int64), y0=np.asarray(r[f"it{k}_y0"], float))


def szifi_sigma(r, k):
    """sigma at the 15 scales: SZiFi's sigma_vec, whose first entry SZiFi leaves at 0 ('maxima_lomem'), with the first
    entry from the verifier's recorder if present (own runs), else as the median y0/q of the detections at scale 0."""
    s = np.asarray(r[f"it{k}_sigvec"], float).copy()
    if f"it{k}_sig_rec" in r:
        s[0] = r[f"it{k}_sig_rec"][0]
    else:
        c = szifi_cat(r, k)
        m = c["jt"] == 0
        s[0] = np.median(c["y0"][m] / c["q"][m]) if m.any() else np.nan
    return s


def finder_cat(st, pre, k):
    return dict(q=np.asarray(st[f"{pre}it{k}_q"], float), py=st[f"{pre}it{k}_iy"] * 1.0, px=st[f"{pre}it{k}_ix"] * 1.0,
                jt=np.asarray(st[f"{pre}it{k}_j"], np.int64))


def prod_tile(pid):
    return load(os.path.join(D3, "tiles", f"tile_{pid:03d}.npz"),
                [f"{m}_it{k}_{v}" for m in MODES for k in (0, 1) for v in ("q", "iy", "ix", "j", "sig")])


def prod_noise(pid):
    return load(os.path.join(D3, "noise_only", f"noise_{pid:03d}.npz"),
                [f"{m}_whitened_it{k}_{v}" for m in MODES for k in (0, 1) for v in ("q", "iy", "ix", "j", "sig")])


def edge_dist(py, px):
    """arcmin from the nearest tile edge (edges at -0.5 and N - 0.5 in index units)."""
    return np.minimum(np.minimum(py + 0.5, N - 0.5 - py), np.minimum(px + 0.5, N - 0.5 - px)) * PIXA


LO, HI = INT / PIXA - 0.5, N - 0.5 - INT / PIXA        # 17.5, 1181.5: the interior is [LO, HI) on both axes


def inside(c):
    """Half-open interior [17.5, 1181.5) in index units: 1164 samples per axis for the finder (integer positions)
    and for SZiFi (half-integer positions after the -1/2 shift) alike, i.e. equal areas."""
    return (c["py"] >= LO) & (c["py"] < HI) & (c["px"] >= LO) & (c["px"] < HI)


def match(A, B, thA, thB=None, rad=1.0, scale_step=None):
    """A's interior entries with q > thA, matched to B's entries (q > thB if given, else all) within rad arcmin
    (nearest; with scale_step, only partners within that many scale steps).  Returns (n, matched, dq = qB - qA,
    djt = jtB - jtA, offset (dy, dx) = B - A in pixels) for the matched entries."""
    sa = inside(A) & (A["q"] > thA)
    sb = np.ones(B["q"].size, bool) if thB is None else (B["q"] > thB)
    ia, ib = np.flatnonzero(sa), np.flatnonzero(sb)
    if ia.size == 0 or ib.size == 0:
        return ia.size, np.zeros(ia.size, bool), np.zeros(0), np.zeros(0, int), np.zeros((0, 2))
    tree = cKDTree(np.c_[B["py"][ib], B["px"][ib]])
    cand = tree.query_ball_point(np.c_[A["py"][ia], A["px"][ia]], r=rad / PIXA + 1e-9)
    ok = np.zeros(ia.size, bool)
    dq, dj, off = [], [], []
    for n, (a, cl) in enumerate(zip(ia, cand)):
        if not cl:
            continue
        cl = ib[np.asarray(cl)]
        if scale_step is not None:
            cl = cl[np.abs(B["jt"][cl] - A["jt"][a]) <= scale_step]
            if cl.size == 0:
                continue
        d2 = (B["py"][cl] - A["py"][a]) ** 2 + (B["px"][cl] - A["px"][a]) ** 2
        b = cl[np.argmin(d2)]
        ok[n] = True
        dq.append(B["q"][b] - A["q"][a])
        dj.append(B["jt"][b] - A["jt"][a])
        off.append((B["py"][b] - A["py"][a], B["px"][b] - A["px"][a]))
    return ia.size, ok, np.array(dq), np.array(dj, int), np.array(off).reshape(-1, 2)


def n_int(c, th):
    return int((inside(c) & (c["q"] > th)).sum())


def boot_ratio(num, den, nb=2000, seed=1):
    """ratio of sums over tiles and its tile-bootstrap standard deviation."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, (nb, num.size))
    rb = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), rb.std(ddof=1)


def auto_ratio(s1, s0, ell, lo, hi):
    """mean over the six channels and the bins with lo <= l < hi of the auto spectra after / before."""
    m = (ell >= lo) & (ell < hi)
    d1 = np.einsum("bii->bi", s1)[m]
    d0 = np.einsum("bii->bi", s0)[m]
    return float(np.mean(d1 / d0))


def ms(x, axis=0):
    """mean, tile sd, error of the mean."""
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x), axis=axis)
    m = np.nanmean(x, axis=axis)
    s = np.nanstd(x, axis=axis, ddof=1) if np.all(n > 1) else np.full_like(m, np.nan)
    return m, s, s / np.sqrt(n)


def fmt(v, f="{:.4f}"):
    return " ".join(f.format(x) for x in np.atleast_1d(v))
