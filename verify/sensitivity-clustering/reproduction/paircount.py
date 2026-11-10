"""Own neighbour counter (dec-strip / sorted-RA sweep in numba), independent of scipy cKDTree."""
import numpy as np
import numba as nb


@nb.njit(cache=True)
def _count(ra, dec, thetas_deg, h, out):
    n = ra.shape[0]
    # strips
    ns = int(np.ceil(180.0 / h))
    sid = np.empty(n, np.int64)
    for i in range(n):
        s = int((dec[i] + 90.0) / h)
        sid[i] = min(max(s, 0), ns - 1)
    key = sid * 1000.0 + ra  # sort by strip then ra
    order = np.argsort(key)
    sra = ra[order]; sdec = dec[order]; ss = sid[order]
    start = np.searchsorted(ss, np.arange(ns + 1))
    d2r = np.pi / 180.0
    cx = np.cos(sdec * d2r) * np.cos(sra * d2r); cy = np.cos(sdec * d2r) * np.sin(sra * d2r); cz = np.sin(sdec * d2r)
    tmax = thetas_deg.max()
    cosT = np.cos(thetas_deg * d2r)
    nt = thetas_deg.shape[0]
    res = np.zeros((n, nt), np.int32)
    for i in range(n):
        si = ss[i]
        for s in range(max(si - 1, 0), min(si + 1, ns - 1) + 1):
            a, b = start[s], start[s + 1]
            if b <= a:
                continue
            adec = abs(sdec[i]) + tmax
            p0 = a; q0 = b; p1 = 0; q1 = 0
            if adec < 89.0:
                dra = tmax / np.cos(adec * d2r) * 1.0001
                r0 = sra[i] - dra; r1 = sra[i] + dra
                if r0 < 0:
                    p0 = a + np.searchsorted(sra[a:b], r0 + 360.0); q0 = b
                    p1 = a; q1 = a + np.searchsorted(sra[a:b], r1, side='right')
                elif r1 >= 360.0:
                    p0 = a + np.searchsorted(sra[a:b], r0); q0 = b
                    p1 = a; q1 = a + np.searchsorted(sra[a:b], r1 - 360.0, side='right')
                else:
                    p0 = a + np.searchsorted(sra[a:b], r0); q0 = a + np.searchsorted(sra[a:b], r1, side='right')
            for rr in range(2):
                if rr == 0:
                    p = p0; q = q0
                else:
                    p = p1; q = q1
                for j in range(p, q):
                    if j == i:
                        continue
                    c = cx[i] * cx[j] + cy[i] * cy[j] + cz[i] * cz[j]
                    for t in range(nt):
                        if c > cosT[t]:
                            res[i, t] += 1
    # back to input order
    for k in range(n):
        for t in range(nt):
            out[order[k], t] = res[k, t]


def count_within(ra, dec, thetas_arcmin):
    th = np.asarray(thetas_arcmin, float) / 60.0
    out = np.zeros((len(ra), len(th)), np.int32)
    _count(np.asarray(ra, float), np.asarray(dec, float), th, max(th.max(), 0.05), out)
    return out
