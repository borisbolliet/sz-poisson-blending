"""z07: test (f) -- SZiFi's iteration 0 against the production finder's stored iteration 0 versus the distance to the
tile edge, and SZiFi's position offset; chooses the edge buffer of the peak-finding and selection masks.

Inputs: runs/S0_MODE_real_PID.npz (SZiFi itself, S0, no edge buffer: peak finding on the whole tile; 6 tiles per noise
level) and data_real/step3/tiles/tile_PID.npz (production finder, periodic, iteration 0).
Positions: a finder detection at pixel (iy, ix) sits at (iy, ix); SZiFi reports pixel (i, j) (z_szlib.cat_arrays).
(1) Offset: SZiFi detections and finder detections with q > 6, both >= 60' from the edges, nearest pair within 1.5'
    (3 pixels) without correction: mean (i - iy, j - ix).  SZiFi's position is then taken as (i - dy, j - dx).
(2) For every finder detection with q > 4.5: the SZiFi detection nearest to the corrected position within 1' (2 pixels),
    any scale (step 3a's definition) and, separately, within 1' and one scale step; dq = q_SZiFi - q_finder of the
    position match; by the distance d of the finder detection to the tile edge (d = min over the axes of
    min(i, N - 1 - i) x 0.5'), for all detections and for those with theta > 2' (wider filters).  Also SZiFi
    detections (q > 4.5) without a finder partner within 1', by distance.
(3) Buffer: the smallest b on a 0.5' grid such that, over the distance bins from b to 60', (i) the position-match
    fractions both ways are within 2 sigma of the deep interior (>= 60') in every bin, and (ii) the bin means of dq
    (all detections, and theta > 2') agree with the interior mean: chi^2 of the bin means (error = interior sd /
    sqrt(N_bin)) has p > 0.05, for both noise levels.  The analysis interior is b + 1'.
Usage: python -I -B z07_edges.py
"""
import os, sys, time
sys.dont_write_bytecode = True
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(ROOT, "data_real", "step3_szifi_real")
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2

t0 = time.time()
N, PIX = 1200, 0.5
REF = dict(np.load(os.path.join(OUT, "reference.npz")))
PIDS = [int(p) for p in REF["pids_s0"]]
MODES = ("baseline", "goal")
EDGES = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 10, 12.5, 15, 20, 30, 60, 300.1])


def dist_edge(y, x):
    return np.minimum(np.minimum(y, N - 1 - y), np.minimum(x, N - 1 - x)) * PIX


def load(mode, pid):
    s = np.load(os.path.join(OUT, "runs", f"S0_{mode}_real_{pid:03d}.npz"))
    t = np.load(os.path.join(ROOT, "data_real", "step3", "tiles", f"tile_{pid:03d}.npz"))
    S = dict(y=s["it0_i"] * 1.0, x=s["it0_j"] * 1.0, q=s["it0_q"], j=s["it0_j_theta"])
    B = dict(y=t[f"{mode}_it0_iy"] * 1.0, x=t[f"{mode}_it0_ix"] * 1.0, q=t[f"{mode}_it0_q"], j=t[f"{mode}_it0_j"])
    return S, B


def nearest(A, B, rad, dj_max=None, shift=(0.0, 0.0)):
    """For each A object, the nearest B object within rad pixels (and |dj| <= dj_max); B positions shifted."""
    out = np.full(A["q"].size, -1, np.int64)
    if A["q"].size == 0 or B["q"].size == 0:
        return out
    tr = cKDTree(np.c_[B["y"] - shift[0], B["x"] - shift[1]])
    cand = tr.query_ball_point(np.c_[A["y"], A["x"]], rad)
    for a, c in enumerate(cand):
        if not c:
            continue
        c = np.asarray(c)
        if dj_max is not None:
            c = c[np.abs(B["j"][c] - A["j"][a]) <= dj_max]
            if c.size == 0:
                continue
        d = np.hypot(B["y"][c] - shift[0] - A["y"][a], B["x"][c] - shift[1] - A["x"][a])
        out[a] = c[np.argmin(d)]
    return out


DATA = {m: {p: load(m, p) for p in PIDS} for m in MODES}
print(f"z07: SZiFi S0 (no edge buffer) iteration 0 vs the production finder's iteration 0, {len(PIDS)} tiles per noise "
      f"level {PIDS}")

# ---------------------------------------------------------------- (1) offset
print("\n(1) SZiFi's position offset: pairs q > 6 (both), >= 60' from the edges, nearest within 1.5' without correction")
OFF = {}
for m in MODES:
    dy, dx = [], []
    for p in PIDS:
        S, B = DATA[m][p]
        sa = (B["q"] > 6) & (dist_edge(B["y"], B["x"]) >= 60)
        Bs = {k: v[sa] for k, v in B.items()}
        Ss = {k: v[S["q"] > 6] for k, v in S.items()}
        k = nearest(Bs, Ss, 3.0)
        ok = k >= 0
        dy.append(Ss["y"][k[ok]] - Bs["y"][ok]); dx.append(Ss["x"][k[ok]] - Bs["x"][ok])
    dy, dx = np.concatenate(dy), np.concatenate(dx)
    OFF[m] = (dy.mean(), dx.mean())
    vals, cnt = np.unique(np.round(dy).astype(int), return_counts=True)
    print(f"  {m:8s}: {dy.size} pairs; mean (i - iy, j - ix) = ({dy.mean():+.3f} +- {dy.std() / np.sqrt(dy.size):.3f}, "
          f"{dx.mean():+.3f} +- {dx.std() / np.sqrt(dx.size):.3f}) pixels; i - iy values {dict(zip(vals.tolist(), cnt.tolist()))}")
dyx = np.mean([OFF[m][0] for m in MODES]), np.mean([OFF[m][1] for m in MODES])
shift = (0.5, 0.5)
print(f"  mean over both levels ({dyx[0]:+.3f}, {dyx[1]:+.3f}) pixels: SZiFi's reported pixel lies half a pixel after the "
      f"finder's; positions corrected by (-0.5, -0.5) pixels = (-0.25', -0.25') below")

# ---------------------------------------------------------------- (2) dq versus distance to the edge
print("\n(2) by the distance d of the object to the tile edge [arcmin]; finder q > 4.5 matched to SZiFi within 1' (position "
      "only, step 3a's definition; 'scale' = also within one scale step), corrected positions")
BUF = {}
STATS = {}
for m in MODES:
    rowsB, rowsS = [], []
    for p in PIDS:
        S, B = DATA[m][p]
        Bq = {k: v[B["q"] > 4.5] for k, v in B.items()}
        k = nearest(Bq, S, 2.0, dj_max=None, shift=shift)
        k1 = nearest(Bq, S, 2.0, dj_max=1, shift=shift)
        dB = dist_edge(Bq["y"], Bq["x"])
        dq = np.where(k >= 0, S["q"][np.maximum(k, 0)] - Bq["q"], np.nan)
        rowsB.append(np.c_[dB, dq, Bq["j"], k1 >= 0])
        Sq = {k_: v[S["q"] > 4.5] for k_, v in S.items()}
        Sq_shift = dict(Sq, y=Sq["y"] - shift[0], x=Sq["x"] - shift[1])
        kk = nearest(Sq_shift, B, 2.0, dj_max=None)
        rowsS.append(np.c_[dist_edge(Sq_shift["y"], Sq_shift["x"]), kk >= 0, Sq["q"]])
    RB = np.concatenate(rowsB); RS = np.concatenate(rowsS)
    print(f"\n  {m}: {RB.shape[0]} finder detections q > 4.5, {RS.shape[0]} SZiFi detections q > 4.5")
    print(f"  {'d [arcmin]':>13s} {'N_find':>7s} {'matched':>8s} {'+scale':>7s} {'mean dq':>8s} {'sd dq':>6s}"
          f" {'dq th<=2':>9s} {'dq th>2 (N)':>13s} {'N_SZiFi':>8s} {'SZ matched':>10s}")
    stats = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        b = (RB[:, 0] >= lo) & (RB[:, 0] < hi)
        s = (RS[:, 0] >= lo) & (RS[:, 0] < hi)
        dq = RB[b, 1]
        ok = np.isfinite(dq)
        big = RB[b, 2] > 6
        f1 = ok.mean() if b.any() else np.nan
        fsc = RB[b, 3].mean() if b.any() else np.nan
        f2 = RS[s, 1].mean() if s.any() else np.nan
        mdq = dq[ok].mean() if ok.any() else np.nan
        sdq = dq[ok].std() if ok.sum() > 1 else np.nan
        a1 = dq[ok & ~big].mean() if (ok & ~big).any() else np.nan
        a2 = dq[ok & big].mean() if (ok & big).any() else np.nan
        stats.append((lo, hi, b.sum(), f1, mdq, sdq, s.sum(), f2, ok.sum(), a2, (ok & big).sum()))
        print(f"  {lo:5.1f}-{hi:5.1f}   {b.sum():7d} {100 * f1:7.1f}% {100 * fsc:6.1f}% {mdq:+8.3f} {sdq:6.3f} "
              f"{a1:+9.3f} {a2:+8.3f} ({(ok & big).sum():3d}) {s.sum():8d} {100 * f2:9.1f}%")
    for lo, hi in ((0.0, 2.5), (2.5, 5.0), (5.0, 8.0), (8.0, 60.0)):
        sb = (RS[:, 0] >= lo) & (RS[:, 0] < hi)
        bb = (RB[:, 0] >= lo) & (RB[:, 0] < hi) & np.isfinite(RB[:, 1])
        un = sb & (RS[:, 1] == 0)
        print(f"  {lo:4.1f}-{hi:4.1f}': SZiFi q > 4.5 peaks without a finder partner {un.sum()} "
              f"({100 * un.sum() / max(sb.sum(), 1):.0f}%), highest q among them {RS[un, 2].max() if un.any() else np.nan:.1f}; "
              f"largest |dq| of a matched pair {np.abs(RB[bb, 1]).max() if bb.any() else np.nan:.2f}; SZiFi peaks per 100 deg^2 "
              f"of that band {100 * sb.sum() / (len(PIDS) * ((600 - 2 * lo) ** 2 - (600 - 2 * hi) ** 2) / 3600):.0f}")
    inn = RB[:, 0] >= 60
    dq_in = RB[inn, 1]; okin = np.isfinite(dq_in)
    ref_m, ref_sd = dq_in[okin].mean(), dq_in[okin].std()
    bigin = okin & (RB[inn, 2] > 6)
    ref_b, ref_bsd = dq_in[bigin].mean(), dq_in[bigin].std()
    f1_in = okin.mean(); s_in = RS[:, 0] >= 60; f2_in = RS[s_in, 1].mean()
    print(f"  deep interior (>= 60'): matched {100 * f1_in:.2f}% / {100 * f2_in:.2f}% (finder->SZiFi / SZiFi->finder), "
          f"{100 * RB[inn, 3].mean():.1f}% also within one scale step; dq {ref_m:+.3f} +- {ref_sd:.3f} "
          f"(theta > 2': {ref_b:+.3f} +- {ref_bsd:.3f})")
    cands = []
    for b0 in np.arange(0.0, 30.01, 0.5):
        good = True
        c_all = c_big = 0.0; n_all = n_big = 0
        for lo, hi, n, f1, mdq, sdq, ns, f2, nok, mb, nb in stats:
            if lo < b0 or lo >= 60 or n < 3:
                continue
            e_f1 = np.sqrt(max(f1_in * (1 - f1_in), 1e-3) / n)
            e_f2 = np.sqrt(max(f2_in * (1 - f2_in), 1e-3) / max(ns, 1))
            if f1 < f1_in - 2 * e_f1 or (ns >= 3 and f2 < f2_in - 2 * e_f2):
                good = False
            if nok >= 3:
                c_all += ((mdq - ref_m) / (ref_sd / np.sqrt(nok))) ** 2; n_all += 1
            if nb >= 3:
                c_big += ((mb - ref_b) / (ref_bsd / np.sqrt(nb))) ** 2; n_big += 1
        p_all = chi2.sf(c_all, max(n_all, 1)); p_big = chi2.sf(c_big, max(n_big, 1))
        if good and p_all > 0.05 and p_big > 0.05:
            cands.append(b0)
            print(f"  buffer {b0}': chi^2 of the bin means {c_all:.1f} for {n_all} bins (p = {p_all:.2f}); theta > 2' "
                  f"{c_big:.1f} for {n_big} bins (p = {p_big:.2f}); match fractions pass")
            break
    BUF[m] = cands[0] if cands else np.nan
    STATS[m] = np.array(stats, dtype=float)
    print(f"  -> smallest buffer passing the test: {BUF[m]}'")
b = float(np.nanmax([BUF[m] for m in MODES]))
print(f"\nchosen edge buffer for the peak-finding and selection masks: b = {b}' (largest of the two levels); analysis "
      f"interior: detections >= {b + 1.0}' from the edges (b + 1')")
np.savez(os.path.join(OUT, "edge_buffer.npz"), buffer_arcmin=b, interior_arcmin=b + 1.0, shift=np.array(shift),
         offset_measured=np.array(dyx), edges=EDGES,
         **{f"stats_{m}": STATS[m] for m in MODES})  # per bin: lo, hi, N, f1, mean dq, sd dq, N_SZ, f2, N_ok, mean dq th>2, N th>2
print(f"done [{time.time() - t0:.0f} s]")
