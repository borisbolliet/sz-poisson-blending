"""j11: what makes SZiFi's iteration end above step 3a's periodic emulation?  A test of the build's explanation (claim
'C3f': SZiFi masks more of the tile, smaller unmasked fraction f) against a second candidate, the tile edges.

SZiFi works on the tile as on a non-periodic patch: its cluster discs, its diffusive inpainting and its cosine blend stop
at the tile edges, while its spectra (FFT) treat the tile as periodic.  A masked disc that reaches an edge is inpainted
from one side only, so the inpainted map no longer matches across the periodic boundary; the emulation (szsim.finder,
periodic) has no such boundary.  For every realistic S1/S2 run (the build's stored runs and the verifier's own), the
cluster mask of iteration 1 is rebuilt from SZiFi's iteration-0 catalogue (discs of radius 3 theta around every q > 4
detection, pixel centres within the radius, not periodic, as mmf.get_cluster_mask / maps.ps_mask do) and checked
against the unmasked fraction SZiFi itself recorded.  Per tile:
  df    = f_SZiFi - f_emulation;
  edge  = fraction of the tile's boundary pixels (outermost rows and columns) that the mask covers.
Then the per-tile excess r_SZiFi - r_emulation at 0.5' (iteration 1) is regressed on df and edge.
Usage: .venv/bin/python -I -B j11_edge_mask.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np
import jlib as L

ref = L.load(os.path.join(L.OUT, "j01_ref.npz"))
ana = L.load(os.path.join(L.OUT, "j03_analytic.npz"))
NEW = [int(p) for p in ref["new_pids"]]
SIGY = {m: ref[f"sigY_{m}"] for m in L.MODES}
N = L.N
yy = np.arange(N)


def szifi_mask(r):
    c = L.szifi_cat(r, 0, shift=0.0)
    m = np.ones((N, N), bool)
    for i, j, q, jt in zip(c["py"], c["px"], c["q"], c["jt"]):
        if q <= 4.0:
            continue
        R = 3.0 * L.TH[jt] / L.PIXA                     # pixels
        Ri = int(np.ceil(R))
        a0, a1 = max(int(i) - Ri, 0), min(int(i) + Ri + 1, N)
        b0, b1 = max(int(j) - Ri, 0), min(int(j) + Ri + 1, N)
        d2 = (yy[a0:a1, None] - i) ** 2 + (yy[None, b0:b1] - j) ** 2
        m[a0:a1, b0:b1] &= d2 > R * R
    return m


rows = []
for src in ("build", "mine"):
    for cfg, name in (("S1", "szifi"), ("S2", "szifi_l12000")):
        for mode in L.MODES:
            pids = L.run_pids(src, cfg, mode, "real")
            if src == "mine":
                pids = [p for p in pids if p in NEW]
            for pid in pids:
                r = L.load(L.run_path(src, cfg, mode, "real", pid))
                if int(r["n_it"]) < 2:
                    continue
                m = szifi_mask(r)
                f_rec = float((r["fsky_rec"] if "fsky_rec" in r else r["rec_fsky"])[0])
                bd = np.concatenate([~m[0, :], ~m[-1, :], ~m[:, 0], ~m[:, -1]])
                r1 = L.szifi_sigma(r, 1)[0] / ana[f"sigana_{mode}"][0]
                if src == "build":
                    e = L.load(os.path.join(L.D3, "arbiter", f"a01_real_{pid:03d}.npz"),
                               [f"{mode}_{name}_sig1", f"{mode}_{name}_fsky"])
                    rE, fE = e[f"{mode}_{name}_sig1"][0] / SIGY[mode][0], float(e[f"{mode}_{name}_fsky"])
                else:
                    p = f"real_{mode}_{pid:03d}_{name}_"
                    rE, fE = ref[p + "it1_sig"][0] / SIGY[mode][0], float(ref[p + "fsky"])
                rows.append((src, cfg, mode, pid, f_rec, float(m.mean()), r1 - rE, f_rec - fE, bd.mean()))

print("j11: SZiFi's iteration-1 cluster mask rebuilt from its iteration-0 catalogue")
dmax = max(abs(r[4] - r[5]) for r in rows)
print(f"  rebuilt unmasked fraction vs the one SZiFi recorded: max |diff| {dmax:.1e} over {len(rows)} runs")
for cfg in ("S1", "S2"):
    for mode in L.MODES:
        sel = [r for r in rows if r[1] == cfg and r[2] == mode]
        if len(sel) < 5:
            continue
        ex = np.array([r[6] for r in sel])
        df = np.array([r[7] for r in sel])
        ed = np.array([r[8] for r in sel])
        X1 = np.c_[np.ones_like(df), df]
        X2 = np.c_[np.ones_like(df), df, ed]
        res1 = ex - X1 @ np.linalg.lstsq(X1, ex, rcond=None)[0]
        b2 = np.linalg.lstsq(X2, ex, rcond=None)[0]
        res2 = ex - X2 @ b2
        print(f"\n  {cfg} {mode}: {len(sel)} tiles (build {sum(r[0] == 'build' for r in sel)}, verifier {sum(r[0] == 'mine' for r in sel)}); "
              f"excess r_SZiFi - r_emul at 0.5': mean {ex.mean():+.4f}, sd {ex.std(ddof=1):.4f}")
        print(f"    correlation of the excess with df {np.corrcoef(ex, df)[0, 1]:+.2f}, with the edge coverage {np.corrcoef(ex, ed)[0, 1]:+.2f}; "
              f"correlation of df with the edge coverage {np.corrcoef(df, ed)[0, 1]:+.2f}")
        cov = np.linalg.inv(X2.T @ X2) * res2.var(ddof=3)
        se = np.sqrt(np.diag(cov))
        print(f"    rms of the excess about a fit on df alone {res1.std(ddof=2):.4f}; on df and the edge coverage {res2.std(ddof=3):.4f}")
        print(f"    fit excess = a + b df + c edge: a {b2[0]:+.4f} +- {se[0]:.4f}, b {b2[1]:+.3f} +- {se[1]:.3f}, c {b2[2]:+.3f} +- {se[2]:.3f}; "
              f"at the sample means: b <df> = {b2[1] * df.mean():+.4f}, c <edge> = {b2[2] * ed.mean():+.4f} (mean excess {ex.mean():+.4f})")
        for r in sorted(sel, key=lambda r: -r[6])[:4]:
            print(f"    largest excess: {r[0]:5s} tile {r[3]:3d}: excess {r[6]:+.4f}, df {r[7]:+.3f}, edge coverage {r[8]:.3f}")
        print(f"    edge coverage: mean {ed.mean():.3f}, range {ed.min():.3f}-{ed.max():.3f}")
