"""zA01 (arbiter): S0's covariance estimate, claim 'S0cov'.  The build (z03b) and the config check disagree on how many
in-band bins of SZiFi's NaMaster spectra ('none', ones mask) differ from a binned periodogram (126 vs 44) and by how much
between l = 1000 and 3000 (<= 1.4% vs up to about 5%).  Both agree that the differences come from modes that lie
exactly on a bin edge.  This decides it with exact integers.

Tile: realistic 0, SO baseline, regenerated (zA00), l-filtered to [100, 20000] with SZiFi's own maps.filter_tmap
(float32, as in S0).  Grid: l = 36 sqrt(n2), n2 = kx^2 + ky^2 (integers; the tile is exactly 10 deg, so the
fundamental is exactly 36).  SZiFi's bins (bin_fac 4): [72 k, 72 (k + 1)), i.e. 4 k^2 <= n2 < 4 (k + 1)^2.  A mode is
ON an edge when n2 = 4 k^2.
Steps:
  (a) NaMaster's binned auto spectra of this map (SZiFi's spec.power_spectrum, decouple 'none'), checked against the
      build's stored S0 run of this tile (identical if the regenerated map is the build's);
  (b) the exact periodogram with each edge mode in the bin ABOVE the edge (exact rule l0 <= l < lf), the same with
      every edge mode in the bin BELOW, and the build's reference (floor(l / 72) with SZiFi's floating-point l);
  (c) per bin, which edge-mode groups (modes with the same {|kx|, |ky|}, hence the same floating-point l) NaMaster
      assigns to the bin below: bins are taken in increasing l, and the groups on the edge between bins k - 1 and k are
      placed so that NaMaster's value of bin k - 1 is reproduced (its other modes are placed by then);
  (d) counts of in-band bins (lower edge 72 ... 19944) that differ by > 1e-6 from each reference, and the largest
      deviation below 1000, in 1000-3000 and above 3000.
Usage: szifi-venv/bin/python -I -B zA01_bins.py
"""
import os, sys, time, itertools, math
sys.dont_write_bytecode = True
t0 = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zA_lib as A                                     # noqa: E402  (only for paths and SZiFi on sys.path)
Z = A.Z
import numpy as np                                     # noqa: E402
from szifi import maps, spec                           # noqa: E402

N, DX = Z.N, Z.DX
pix = maps.pixel(N, DX)
tmap = np.load(os.path.join(Z.MAPD, "baseline_real_000.npy"))
tf = maps.filter_tmap(tmap, pix, [100, 20000])
del tmap
ones = np.ones((N, N))
ps = spec.power_spectrum(pix, mask=ones, cm_compute=False, bin_fac=4)
nb = ps.l0_bins.size                                   # 300 bins, l0 = 0, 72, ..., 21528
print(f"(a) SZiFi's bins: {nb}, width {ps.lf_bins[0] - ps.l0_bins[0]:.6f}; fundamental 2 pi / (N dx) = "
      f"{2 * np.pi / (N * DX):.10f}")
nm = np.zeros((nb, 6))
for i in range(6):
    _, cl = ps.get_power_spectrum(tf[:, :, i], decouple_type="none")
    nm[:, i] = cl
with np.load(os.path.join(Z.OUT, "runs", "S0_baseline_real_000.npz")) as d:
    s0 = d["spec0"]
same = np.array_equal(nm[1:], np.stack([s0[:, i, i] for i in range(6)], 1))
print(f"    NaMaster auto spectra of the regenerated map == the build's stored S0 run (bins 1-299): {same}")

# (b) periodograms
k = np.rint(np.fft.fftfreq(N) * N).astype(np.int64)
n2 = (k[:, None] ** 2 + k[None, :] ** 2)
isq = np.vectorize(math.isqrt, otypes=[np.int64])(n2)
b_up = isq // 2                                         # exact: bin with 4 b^2 <= n2 < 4 (b + 1)^2
on_edge = (n2 == 4 * b_up ** 2) & (n2 > 0)
b_dn = np.where(on_edge, b_up - 1, b_up)
ellf = maps.rmap(pix).get_ell()
b_fl = np.floor(ellf / (ps.lf_bins[0] - ps.l0_bins[0])).astype(np.int64)   # the build's z03 reference
P = np.stack([np.abs(maps.get_fft(tf[:, :, i].astype(np.float64), pix)) ** 2 for i in range(6)], 0)
P = P.reshape(6, -1)


def binned(b, w=None):
    b = b.ravel()
    sel = b < nb
    cnt = np.bincount(b[sel], minlength=nb)[:nb]
    out = np.zeros((nb, 6))
    for i in range(6):
        out[:, i] = np.bincount(b[sel], weights=P[i][sel], minlength=nb)[:nb] / np.maximum(cnt, 1)
    return out


REFS = {"exact, edge modes in the bin above (l0 <= l < lf)": b_up, "exact, edge modes in the bin below": b_dn,
        "build's z03 reference: floor(l / bin width), floating-point l": b_fl}
inb = np.arange(nb)
inb = (ps.l0_bins >= 72) & (ps.l0_bins < 20000)
print(f"(b),(d) in-band bins (lower edge 72 ... 19944): {inb.sum()}; modes exactly on a bin edge: {on_edge.sum()} "
      f"({100 * on_edge.mean():.2f}% of the grid)")


def report(lab, ref):
    dev = np.max(np.abs(nm / ref - 1), axis=1)
    bad = inb & (dev > 1e-6)
    parts = []
    for lo, hi in ((72, 1000), (1000, 3000), (3000, 20000)):
        m = bad & (ps.l0_bins >= lo) & (ps.l0_bins < hi)
        parts.append(f"{lo}-{hi}: {m.sum()} bins, max {dev[m].max() if m.any() else 0:.4f}")
    ok = inb & ~bad
    print(f"    vs {lab}: {bad.sum()} in-band bins differ by > 1e-6 ({'; '.join(parts)}); other bins agree to "
          f"{dev[ok].max():.1e}")
    return dev, bad


devs = {lab: report(lab, binned(b)) for lab, b in REFS.items()}
dev_up, bad_up = devs["exact, edge modes in the bin above (l0 <= l < lf)"]
print(f"    bins differing from the exact rule, lower edges 72-3000: "
      f"{[int(x) for x in ps.l0_bins[bad_up & (ps.l0_bins < 3000)]]}")

# (c) which edge-mode groups NaMaster puts below the edge
ex, ey = np.nonzero(on_edge)
grp = {}
for x_, y_ in zip(ex, ey):
    key = (int(b_up[x_, y_]), tuple(sorted((abs(int(k[x_])), abs(int(k[y_]))))))
    grp.setdefault(key, []).append(x_ * N + y_)
by_bin = {}
for (bb, g), idx in grp.items():
    by_bin.setdefault(bb, []).append((g, np.array(idx)))
assign = b_up.ravel().copy()
n_groups = n_down = 0
unexplained = []
for bb in sorted(by_bin):
    if bb >= nb or bb < 1:
        continue
    groups = by_bin[bb]
    n_groups += len(groups)
    best = None
    for choice in itertools.product((0, 1), repeat=len(groups)):
        a = assign.copy()
        for c, (g, idx) in zip(choice, groups):
            if c:
                a[idx] = bb - 1
        rows = [bb - 1]                  # bin bb-1 is complete once the groups on its upper edge (= bin bb's lower edge) are placed
        val = binned(a.reshape(N, N))[rows]
        err = np.max(np.abs(nm[rows] / val - 1))
        if best is None or err < best[0]:
            best = (err, choice)
    err, choice = best
    for c, (g, idx) in zip(choice, groups):
        if c:
            assign[idx] = bb - 1
            n_down += 1
    if err > 1e-12:
        unexplained.append((bb, err))
fit = binned(assign.reshape(N, N))
dev_fit = np.max(np.abs(nm / fit - 1), axis=1)
print(f"(c) edge-mode groups (same |kx|, |ky|) on the lower edges of bins 1-299: {n_groups}; NaMaster puts {n_down} of them "
      f"in the bin below")
print(f"    with that assignment, NaMaster's spectra equal the periodogram in every in-band bin to "
      f"{dev_fit[inb].max():.1e} (all bins 1-299: {dev_fit[1:].max():.1e}); bins not explained to 1e-12: {len(unexplained)}")
moved = [(bb, g) for (bb, g), idx in grp.items() if 1 <= bb < nb and assign[idx[0]] == bb - 1]
inb_moved = [m for m in moved if 144 <= ps.l0_bins[m[0]] <= 19944]
print(f"    groups placed below an edge between two in-band bins (edges l = 144 ... 19944): {len(inb_moved)}; at the band "
      f"edges or outside the band: {len(moved) - len(inb_moved)}; unexplained bins: "
      f"{[int(ps.l0_bins[b]) for b, _ in unexplained][:30]}")
print(f"    in-band edges with a group below: l = {sorted(int(ps.l0_bins[b]) for b, _ in inb_moved)}")
ax = sum(1 for (bb, g) in grp if bb < nb and bb >= 1 and g[0] == 0)
ax_dn = sum(1 for (bb, g), idx in grp.items() if 1 <= bb < nb and g[0] == 0 and assign[idx[0]] == bb - 1)
print(f"    axis groups (0, 2k): {ax}, of which below: {ax_dn}; off-axis groups: {n_groups - ax}, of which below: "
      f"{n_down - ax_dn}")
print(f"done [{time.time() - t0:.0f} s]; peak RSS {Z.peak_rss_gb():.2f} GB")
