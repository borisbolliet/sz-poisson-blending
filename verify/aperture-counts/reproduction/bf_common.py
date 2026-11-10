"""Brute-force (sorted-z window) neighbour counting, independent of scipy cKDTree."""
import numpy as np
import numba as nb

AM = np.pi / 180 / 60


def chord(theta_arcmin):
    return 2 * np.sin(np.asarray(theta_arcmin, float) * AM / 2)


class SortedPts:
    def __init__(self, vec, ids):
        o = np.argsort(vec[:, 2], kind="stable")
        self.v = np.ascontiguousarray(vec[o])
        self.ids = np.ascontiguousarray(ids[o]).astype(np.int64)
        self.zs = np.ascontiguousarray(self.v[:, 2])


@nb.njit(parallel=True, cache=True)
def _counts(cv, cid, pv, pz, pid, c2, cmax):
    n = cv.shape[0]; T = c2.shape[0]
    out = np.zeros((n, T), np.int32)
    for i in nb.prange(n):
        x, y, z = cv[i, 0], cv[i, 1], cv[i, 2]
        lo = np.searchsorted(pz, z - cmax)
        hi = np.searchsorted(pz, z + cmax, side="right")
        for j in range(lo, hi):
            if pid[j] == cid[i]:
                continue
            dx = pv[j, 0] - x; dy = pv[j, 1] - y; dz = pv[j, 2] - z
            d2 = dx * dx + dy * dy + dz * dz
            for t in range(T):
                if d2 <= c2[t]:
                    out[i, t] += 1
    return out


def counts(cen_vec, cen_ids, pts: SortedPts, thetas):
    """N(<=theta) of pts around each centre, excluding pts with the same id as the centre (id -1 = none)."""
    c = chord(thetas)
    return _counts(np.ascontiguousarray(cen_vec), np.ascontiguousarray(cen_ids, dtype=np.int64),
                   pts.v, pts.zs, pts.ids, c * c, float(c.max()) * (1 + 1e-12))


@nb.njit(cache=True)
def _triangles(v, z, c2, cmax):
    # v sorted by z; count unordered triples with all three squared chords <= c2
    n = v.shape[0]
    tot = 0
    nbr = np.empty(4096, np.int64)
    for i in range(n):
        hi = np.searchsorted(z, z[i] + cmax, side="right")
        k = 0
        for j in range(i + 1, hi):
            dx = v[j, 0] - v[i, 0]; dy = v[j, 1] - v[i, 1]; dz = v[j, 2] - v[i, 2]
            if dx * dx + dy * dy + dz * dz <= c2:
                nbr[k] = j; k += 1
        for a in range(k):
            for b in range(a + 1, k):
                p, q = nbr[a], nbr[b]
                dx = v[p, 0] - v[q, 0]; dy = v[p, 1] - v[q, 1]; dz = v[p, 2] - v[q, 2]
                if dx * dx + dy * dy + dz * dz <= c2:
                    tot += 1
    return tot


def triangles(pts: SortedPts, theta):
    c = float(chord(theta))
    return _triangles(pts.v, pts.zs, c * c, c * (1 + 1e-12))
