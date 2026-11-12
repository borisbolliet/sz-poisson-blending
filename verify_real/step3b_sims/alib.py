"""Step 3b verification, lens 'sims': analysis helpers (no finder code).  Loads the per-tile outputs of y02 (each npz
read once, only the keys needed), builds the kernel, N_det / N_exp / N_iso per (z, q_obs) bin and tile bootstraps.
Bins: step 2's (cosmolib.default_bins: 16 z bins 0.01-3, 9 q_obs bins from q_cut, the last q > 100)."""
import os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(HERE, "out")
sys.path.insert(0, f"{ROOT}/analysis/cosmo/bias")
import cosmolib as CL                                   # noqa: E402  (bins only)

TH_K = 0.3 * 2.0 ** ((np.arange(25) - 1) / 3.0)
LTK = np.log(TH_K)
DLT = LTK[1] - LTK[0]
QG = np.unique(np.round(np.concatenate([[0.0, 1.0], np.arange(1.5, 30.0 + 1e-9, 0.1), np.geomspace(30.0, 3000.0, 81)]), 10))
EDGES = np.array(np.load(os.path.join(ROOT, "data_cosmo", "kernel_catalogue.npz"))["EDGES"], float)
MODES = ("baseline", "goal")
ZC = ((0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0))


def bins(qcut):
    zed, qed = CL.default_bins(qcut)
    return np.asarray(zed, float), np.asarray(qed, float)


def edge_index(qed):
    """indices into EDGES of the finite q edges; the last edge (inf) -> -1."""
    out = []
    for e in qed:
        if not np.isfinite(e):
            out.append(-1)
        else:
            k = int(np.argmin(np.abs(EDGES - e)))
            assert abs(EDGES[k] - e) < 1e-6, e
            out.append(k)
    return np.array(out)


TILE_KEYS = ("rows", "qp", "qiso", "z", "th", "qobs_none", "qobs_closest1", "qobs_fof10", "r1", "r0", "rf", "kc", "kn",
             "ncred_none", "ncred_closest1", "ncred_fof10", "ncred_below_none", "ndet5", "nnr_none", "nreal_none",
             "noise_q", "noise_n5_closest1", "k_below2", "k_offmax", "nr_q", "nr_z", "fsky")


def load_tiles(meth, mode, tiles=None, keys=TILE_KEYS, need_kernel=True):
    """list of dicts (one per tile) with the requested keys for one noise level."""
    files = sorted(glob.glob(os.path.join(OUT, f"tiles_{meth}", "t_*.npz")))
    out = []
    for fn in files:
        t = int(os.path.basename(fn)[2:6])
        if tiles is not None and t not in tiles:
            continue
        d = np.load(fn)
        rec = {"tile": t}
        for k in keys:
            kk = f"{mode}_{k}"
            if kk in d.files:
                if k in ("kc",) and not need_kernel:
                    continue
                rec[k] = d[kk]
        for k in ("wc_changed", "wc_n"):
            kk = f"{mode}_{k}"
            if kk in d.files:
                rec[k] = int(d[kk])
        rec["nh"] = int(d["nh"])
        out.append(rec)
        d.close()
    return out


def kernel_from(recs, weights=None, monotone=True):
    """K (25, nq, 19) = credited fraction with q_obs > e per (theta node, q'); weights: per-tile bootstrap weights."""
    kc = np.zeros((TH_K.size, QG.size, EDGES.size)); kn = np.zeros(TH_K.size)
    for i, r in enumerate(recs):
        w = 1.0 if weights is None else weights[i]
        if w == 0:
            continue
        kc += w * r["kc"]; kn += w * r["kn"]
    K = kc / kn[:, None, None]
    Kraw = K.copy()
    if monotone:
        K = np.maximum.accumulate(K, axis=1)
    return K, Kraw, kn


def kernel_stack(recs):
    """per-tile kernel counts as float32 arrays (ntile, 25*nq*19) and (ntile, 25) for fast resampling."""
    KC = np.stack([r["kc"].reshape(-1) for r in recs]).astype(np.float32)
    KN = np.stack([r["kn"] for r in recs]).astype(np.float64)
    return KC, KN


def K_at(K, th, qp):
    """K(theta, q'; e) for every halo (n, 19): linear in ln theta between nodes (clipped), linear in q' (np.interp)."""
    f = np.clip((np.log(th) - LTK[0]) / DLT, 0.0, TH_K.size - 1 - 1e-9)
    i0 = f.astype(int); w = f - i0
    j = np.clip(np.searchsorted(QG, qp, side="right") - 1, 0, QG.size - 2)
    u = np.clip((qp - QG[j]) / (QG[j + 1] - QG[j]), 0.0, 1.0)
    A = K[i0, j] * (1 - u)[:, None] + K[i0, j + 1] * u[:, None]
    B = K[i0 + 1, j] * (1 - u)[:, None] + K[i0 + 1, j + 1] * u[:, None]
    return A * (1 - w)[:, None] + B * w[:, None]


def pbin(Kh, qed):
    """P(q_obs in each q bin) per halo from K at the edges (n, nqb)."""
    ei = edge_index(qed)
    cols = []
    for a, b in zip(ei[:-1], ei[1:]):
        lo = Kh[:, a]
        hi = Kh[:, b] if b >= 0 else 0.0
        cols.append(lo - hi)
    return np.stack(cols, 1)


def hist_zq(z, q, zed, qed, w=None):
    ok = np.isfinite(q) & (q > qed[0])
    H, _, _ = np.histogram2d(z[ok], q[ok], bins=[zed, qed[:-1].tolist() + [1e30]], weights=None if w is None else w[ok])
    return H


def per_tile_counts(recs, K, qcut, merge="none", qkey=None):
    """per-tile arrays (ntile, nz, nq): N_det, N_exp (with K), N_iso."""
    zed, qed = bins(qcut)
    nz, nq = zed.size - 1, qed.size - 1
    Nd = np.zeros((len(recs), nz, nq)); Ne = np.zeros_like(Nd); Ni = np.zeros_like(Nd)
    for i, r in enumerate(recs):
        z, th, qp = r["z"], r["th"], r["qp"]
        qo = r[qkey or f"qobs_{merge}"]
        Nd[i] = hist_zq(z, qo, zed, qed)
        Ni[i] = hist_zq(z, r["qiso"], zed, qed)
        if K is not None:
            P = pbin(K_at(K, th, qp), qed)
            iz = np.clip(np.searchsorted(zed, z, side="right") - 1, 0, nz - 1)
            okz = (z >= zed[0]) & (z < zed[-1])
            for b in range(nq):
                Ne[i, :, b] = np.bincount(iz[okz], weights=P[okz, b], minlength=nz)
    return Nd, Ne, Ni


def ratio(num, den, guard):
    """num/den per bin; 1 where guard < 1 (fewer than one expected cluster in the tiles) or den == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((guard >= 1.0) & (den > 0), num / np.where(den > 0, den, 1.0), 1.0)
    return r


def boot_weights(ntile, B, seed):
    rng = np.random.default_rng(seed)
    W = np.zeros((B, ntile))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, ntile, ntile), minlength=ntile)
    return W


def zcoarse(Nnum, Nden, zed):
    out = []
    for a, b in ZC:
        s = (zed[:-1] >= a - 1e-9) & (zed[1:] <= b + 1e-9)
        out.append(Nnum[..., s, :].sum((-1, -2)) / Nden[..., s, :].sum((-1, -2)))
    return np.stack(out, -1)
