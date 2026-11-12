"""j07: SZiFi on periodic tiles without an edge buffer (S0): spurious edge peaks (claim 1f), the edge buffer (the build's
'buffer' claim), and claim 2 (no inpainting: iteration 1 = iteration 0), recomputed by the verifier from the build's
stored S0 runs (6 tiles per level, read as data) and from the verifier's own S0 run(s).
Distances to the edge in index units (edges at -0.5 and 1199.5), SZiFi positions shifted by -1/2 pixel (j04 (a)).
A SZiFi peak is 'spurious' if the production finder's iteration-0 catalogue (stored; periodic tile, so it has no edges)
has no detection within 1'.  Partners for dq: finder q > 4.5 detections matched to SZiFi's catalogue (any q) within 1'.
Usage: .venv/bin/python -I -B j07_edges.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
import jlib as L

BANDS = [(0, 2.5), (2.5, 5), (5, 8), (8, 60)]
BINS = [(5, 6), (6, 7.5), (7.5, 10), (10, 12.5), (12.5, 15), (15, 20), (20, 30), (30, 60), (60, 301)]
print("j07: SZiFi S0 (no edge buffer), iteration 0, vs the production finder's iteration 0")
for src in ("build", "mine"):
    for mode in L.MODES:
        pids = L.run_pids(src, "S0", mode, "real")
        if not pids:
            continue
        print(f"\n  ---- {mode}, {src} S0 runs, tiles {pids}")
        same = 0
        sp = {b: [] for b in BANDS}
        allpk = {b: 0 for b in BANDS}
        dqs, dds, ths = [], [], []
        for pid in pids:
            r = L.load(L.run_path(src, "S0", mode, "real", pid))
            ident = all(np.array_equal(r[f"it0_{v}"], r[f"it1_{v}"]) for v in ("q", "i", "j", "theta", "y0", "sigvec")) \
                and np.array_equal(r["spec0"], r["spec1"])
            same += ident
            S = L.szifi_cat(r, 0)
            Fc = L.finder_cat(L.prod_tile(pid), f"{mode}_", 0)
            d = L.edge_dist(S["py"], S["px"])
            # spurious: no finder detection within 1'
            tr = cKDTree(np.c_[Fc["py"], Fc["px"]])
            dist, _ = tr.query(np.c_[S["py"], S["px"]], k=1)
            spur = dist > 1.0 / L.PIXA
            for b in BANDS:
                m = (d >= b[0]) & (d < b[1]) & (S["q"] > 4.5)
                allpk[b] += int(m.sum())
                sp[b].append(S["q"][m & spur])
            # dq of finder detections (q > 4.5) by distance to the edge
            sa = Fc["q"] > 4.5
            trS = cKDTree(np.c_[S["py"], S["px"]])
            cand = trS.query_ball_point(np.c_[Fc["py"][sa], Fc["px"][sa]], r=1.0 / L.PIXA + 1e-9)
            for a, cl in zip(np.flatnonzero(sa), cand):
                if not cl:
                    continue
                cl = np.asarray(cl)
                b = cl[np.argmin((S["py"][cl] - Fc["py"][a]) ** 2 + (S["px"][cl] - Fc["px"][a]) ** 2)]
                dqs.append(S["q"][b] - Fc["q"][a])
                dds.append(L.edge_dist(Fc["py"][a], Fc["px"][a]))
                ths.append(L.TH[Fc["jt"][a]])
        print(f"    claim 2: iteration 1 identical to iteration 0 (q, pixels, theta, y0, sigma_vec, spectra): {same} of {len(pids)} tiles")
        nt = len(pids)
        for b in BANDS:
            area = ((600 - 2 * b[0]) ** 2 - (600 - 2 * b[1]) ** 2) / 3600.0 * nt
            q = np.concatenate(sp[b])
            print(f"    edge distance {b[0]:4.1f}-{b[1]:4.1f}': SZiFi q > 4.5 peaks per 100 deg^2 of the band {100 * allpk[b] / area:7.0f}; "
                  f"without a finder detection within 1': {q.size} ({100 * q.size / max(allpk[b], 1):.0f}%); highest q among them "
                  f"{(q.max() if q.size else float('nan')):.1f}")
        dqs, dds, ths = np.array(dqs), np.array(dds), np.array(ths)
        ref_m = dds >= 60
        mi, si, ni = dqs[ref_m].mean(), dqs[ref_m].std(ddof=1), ref_m.sum()
        line = "    mean dq = q_SZiFi - q_finder by edge distance: " + "; ".join(
            f"{a}-{b}' {dqs[(dds >= a) & (dds < b)].mean():+.2f} ({((dds >= a) & (dds < b)).sum()})" for a, b in BINS)
        print(line)
        for bb in (5.0, 6.0, 7.5, 8.0, 10.0):
            for lab, sel in (("all", np.ones(dqs.size, bool)), ("theta > 2'", ths > 2.0)):
                mi_, si_, ni_ = dqs[ref_m & sel].mean(), dqs[ref_m & sel].std(ddof=1), (ref_m & sel).sum()
                chi, k = 0.0, 0
                for a, b in [(bb, 10.0), (10, 12.5), (12.5, 15), (15, 20), (20, 30), (30, 60)]:
                    if a >= b:
                        continue
                    m = (dds >= a) & (dds < b) & sel
                    if m.sum() < 2:
                        continue
                    chi += (dqs[m].mean() - mi_) ** 2 / (dqs[m].var(ddof=1) / m.sum() + si_ ** 2 / ni_)
                    k += 1
                print(f"    buffer {bb:4.1f}': bins from the buffer to 60' vs >= 60' ({lab:10s}): chi^2 {chi:5.1f} for {k} bins, p = {stats.chi2.sf(chi, k):.2f}")
