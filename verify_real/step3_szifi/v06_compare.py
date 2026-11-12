"""v06: checks (b) and (c) -- the SZiFi-convention finder (our emulation, out/v04_CONV_*.npz) against the build's stored
results on the same tiles.

Positions: a build detection at pixel (iy, ix) refers to position (iy, ix); a SZiFi-convention detection at pixel
(n0, n1) refers to (n0 - 1/2, n1 - 1/2) (the 'same'-mode crop with N even), while SZiFi would REPORT the pixel itself.
Matching: nearest detection (q > 4, any scale) within 1' (2 pixels), non-periodic distances, objects >= MARGIN from the
tile edges.  Scales: theta_est on the same 15 scales.
Usage: python -I -B v06_compare.py CONV [MARGIN_ARCMIN]
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, HERE)
import numpy as np
import vsz_lib as V

F = V.F
CONV = sys.argv[1]
MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else (2.0 if CONV == "per" else 60.0)
RAD = 2.0                                         # match radius [pixels] = 1'
TH = V.THETAS
OUT = V.OUTD
G = F.grid()
print(f"v06: SZiFi-convention finder ({'periodic' if CONV == 'per' else 'linear same-mode'} convolution) vs the build; "
      f"matching radius 1', margin {MARGIN}' from the tile edges")

# step-2 normalisations quoted from the build's log (for reference only)
t05 = open(os.path.join(V.ROOT, "analysis", "realistic", "step3", "t05_analysis.log")).read().splitlines()
step2 = {}
cur = None
for l in t05:
    m = re.match(r"=+ (baseline|goal):", l)
    if m:
        cur = m.group(1)
    for key, lab in (("i", "step 2 (i) sub-threshold confusion"), ("iii", "step 2 (iii) plain std")):
        if cur and l.strip().startswith(lab):
            step2[(cur, key)] = np.array([float(x) for x in l.split(lab)[1].split()])


def load(mode, src, pid):
    fn = os.path.join(OUT, f"v04_{CONV}_{mode}_{src}{pid:03d}.npz")
    return dict(np.load(fn)) if os.path.exists(fn) else None


def sz_det(r, it):
    k = f"det{it}"
    if f"{k}_q" not in r:
        k = "det0"
    y, x = r[f"{k}_n0"] - 0.5, r[f"{k}_n1"] - 0.5
    return dict(y=y, x=x, q=r[f"{k}_q"], j=r[f"{k}_j"], ry=r[f"{k}_n0"] * 1.0, rx=r[f"{k}_n1"] * 1.0)


def b_det(st, mode, it):
    return dict(y=st[f"{mode}_it{it}_iy"] * 1.0, x=st[f"{mode}_it{it}_ix"] * 1.0, q=st[f"{mode}_it{it}_q"],
                j=st[f"{mode}_it{it}_j"])


def compare(A, B, qcut, lab):
    """Objects of A with q > qcut (inside the margin) matched to B (q > 4) within 1'."""
    sa = (A["q"] > qcut) & V.interior(A["y"], A["x"], MARGIN)
    sb = V.interior(B["y"], B["x"], MARGIN - 1.0)
    ia = np.flatnonzero(sa); ib = np.flatnonzero(sb)
    k, d = V.match(A["y"][ia], A["x"][ia], B["y"][ib], B["x"][ib], RAD)
    ok = k >= 0
    res = dict(n=ia.size, nm=int(ok.sum()))
    if ok.any():
        a, b = ia[ok], ib[k[ok]]
        res.update(dq=B["q"][b] - A["q"][a], dj=B["j"][b] - A["j"][a], dist=d[ok] * V.PIXA,
                   dy=(B["y"][b] - A["y"][a]) * V.PIXA, dx=(B["x"][b] - A["x"][a]) * V.PIXA)
    return res


def pool_stats(rs):
    n = sum(r["n"] for r in rs); nm = sum(r["nm"] for r in rs)
    cat = lambda k: np.concatenate([r[k] for r in rs if k in r]) if any(k in r for r in rs) else np.zeros(0)
    return n, nm, cat("dq"), cat("dj"), cat("dist"), cat("dy"), cat("dx")


def line(lab, rs):
    n, nm, dq, dj, dist, dy, dx = pool_stats(rs)
    if nm == 0:
        print(f"    {lab:58s}: N {n}, matched 0")
        return
    print(f"    {lab:58s}: N {n:5d}, matched {100 * nm / n:5.1f}%; dq mean {dq.mean():+.3f} std {dq.std():.3f}; "
          f"same scale {100 * np.mean(dj == 0):4.1f}%, within 1 step {100 * np.mean(np.abs(dj) <= 1):4.1f}%; "
          f"offset median {np.median(dist):.2f}', mean (dy, dx) ({dy.mean():+.3f}, {dx.mean():+.3f})'")


TILES = [0, 33, 66, 99, 132, 165, 198, 231, 264, 297]
for mode in V.MODES:
    rs = [(p, load(mode, "tile", p), V.load_stored_tile(p)) for p in TILES]
    rs = [x for x in rs if x[1] is not None]
    if not rs:
        continue
    sY = F.Filter(G, mode=mode, analytic=True).sigma_arcmin2
    print(f"\n==================== {mode}: {len(rs)} realistic tiles ({len(rs) * 100} deg^2)")
    # ---------------- sigma / r
    r_sz0 = np.array([r["sig0"] / r["sig_ana"] for _, r, _ in rs])
    r_sz1 = np.array([r["sig1"] / r["sig_ana"] for _, r, _ in rs])
    r_b0 = np.array([st[f"{mode}_it0_sig"] / sY for _, _, st in rs])
    r_b1 = np.array([st[f"{mode}_it1_sig"] / sY for _, _, st in rs])
    print("  r(theta) = sigma / sigma(analytic CMB + noise), each with its own template; mean over tiles [tile std]")
    print("    theta [']                                  " + " ".join(f"{t:7.2f}" for t in TH))
    for lab, a in (("build, before iteration", r_b0), ("SZiFi conventions, before iteration", r_sz0),
                   ("build, after iteration (whitened mask)", r_b1),
                   ("SZiFi, after iteration (inpaint+blend+lsep)", r_sz1),
                   ("SZiFi, no inpainting (= before)", r_sz0)):
        print(f"    {lab:43s} " + " ".join(f"{x:7.4f}" for x in a.mean(0)))
        if "SZiFi, after" in lab or "build, after" in lab:
            print(f"    {'   tile std':43s} " + " ".join(f"{x:7.4f}" for x in a.std(0)))
    for key, lab in (("i", "step 2 (i), build's t05 log"), ("iii", "step 2 (iii), build's t05 log")):
        if (mode, key) in step2:
            print(f"    {lab:43s} " + " ".join(f"{x:7.4f}" for x in step2[(mode, key)]))
    if (mode, "i") in step2 and (mode, "iii") in step2:
        ri, riii = step2[(mode, "i")], step2[(mode, "iii")]
        print(f"    SZiFi after: (r-1)/(r_(i)-1)              " + " ".join(f"{x:7.2f}" for x in (r_sz1.mean(0) - 1) / (ri - 1)))
        print(f"    SZiFi after: (r-1)/(r_(iii)-1)            " + " ".join(f"{x:7.2f}" for x in (r_sz1.mean(0) - 1) / (riii - 1)))
        print(f"    build after: (r-1)/(r_(i)-1) [these tiles] " + " ".join(f"{x:7.2f}" for x in (r_b1.mean(0) - 1) / (ri - 1)))
    print(f"    sigma ratio after/before: build {np.mean(r_b1 / r_b0, 0).min():.4f}-{np.mean(r_b1 / r_b0, 0).max():.4f}; "
          f"SZiFi {np.mean(r_sz1 / r_sz0, 0).min():.4f}-{np.mean(r_sz1 / r_sz0, 0).max():.4f}")
    fs = np.array([r["fsky"] for _, r, _ in rs]); fb = np.array([st[f"{mode}_it1_fsky"] for _, _, st in rs])
    w2 = np.array([r["w2mean_hole"] for _, r, _ in rs]); w1 = np.array([r["wmean_hole"] for _, r, _ in rs])
    print(f"    unmasked fraction: SZiFi {fs.mean():.3f} [{fs.min():.3f}-{fs.max():.3f}], build {fb.mean():.3f}; "
          f"SZiFi blend weight in the holes: mean w {w1.mean():.3f}, mean w^2 {w2.mean():.3f}")
    dd = np.arange(6)
    for lo, hi in ((100, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 20000)):
        b = np.flatnonzero((V.szgrid().lc >= lo) & (V.szgrid().lc < hi))
        a1 = np.mean([np.mean(r["C1"][b][:, dd, dd] / r["C0"][b][:, dd, dd]) for _, r, _ in rs])
        a2 = np.mean([np.mean(r["C1raw"][b][:, dd, dd] / r["C0"][b][:, dd, dd]) for _, r, _ in rs])
        a3 = np.mean([np.mean(st[f"{mode}_it1_C"][b][:, dd, dd] / st[f"{mode}_it0_C"][b][:, dd, dd]) for _, _, st in rs])
        print(f"    auto spectra after/before, l {lo:5d}-{hi:5d}: SZiFi {a1:.4f} (before the lsep division {a2:.4f}); "
              f"build {a3:.4f}")
    # ---------------- counts
    print("  detections per tile (mean), q > 4 / 4.5 / 5:")
    for lab, getq in (("build it0", lambda r, st: st[f"{mode}_it0_q"]), ("build it1 (final)", lambda r, st: st[f"{mode}_it1_q"]),
                      ("SZiFi it0 (= final without inpainting)", lambda r, st: r["det0_q"]),
                      ("SZiFi it1 (final with inpainting)", lambda r, st: r["det1_q"])):
        c = np.array([[np.sum(getq(r, st) > qc) for qc in (4.0, 4.5, 5.0)] for _, r, st in rs])
        print(f"    {lab:40s}: " + " / ".join(f"{x:6.1f}" for x in c.mean(0)))
    c0 = sum(np.sum(st[f"{mode}_it0_q"] > 5) for _, _, st in rs); c1 = sum(np.sum(st[f"{mode}_it1_q"] > 5) for _, _, st in rs)
    s0 = sum(np.sum(r["det0_q"] > 5) for _, r, _ in rs); s1 = sum(np.sum(r["det1_q"] > 5) for _, r, _ in rs)
    print(f"    q > 5 before -> after the iteration: build {c0} -> {c1} ({100 * (c1 / c0 - 1):+.1f}%); "
          f"SZiFi {s0} -> {s1} ({100 * (s1 / s0 - 1):+.1f}%)")
    # ---------------- catalogue matching
    print(f"  catalogue matching (objects >= {MARGIN}' from the edges):")
    for qc in (4.5, 5.0):
        print(f"   q > {qc}:")
        pairs = (("build it0 -> SZiFi it0 (same iteration)", 0, 0, "b"), ("SZiFi it0 -> build it0", 0, 0, "s"),
                 ("build it1 -> SZiFi it1 (finals, SZiFi inpainting)", 1, 1, "b"),
                 ("SZiFi it1 -> build it1", 1, 1, "s"),
                 ("build it1 -> SZiFi it0 (finals, SZiFi without inpainting)", 1, 0, "b"),
                 ("SZiFi it0 -> build it1", 1, 0, "s"))
        for lab, ib, isz, direction in pairs:
            out = []
            for _, r, st in rs:
                Bd, Sd = b_det(st, mode, ib), sz_det(r, isz)
                out.append(compare(Bd, Sd, qc, lab) if direction == "b" else compare(Sd, Bd, qc, lab))
            line(lab, out)
    # SZiFi's reported pixel (no half-pixel correction) vs the build, finals at iteration 0
    out = []
    for _, r, st in rs:
        Bd, Sd = b_det(st, mode, 0), sz_det(r, 0)
        Sd = dict(Sd, y=Sd["ry"], x=Sd["rx"])
        out.append(compare(Bd, Sd, 5.0, ""))
    line("build it0 -> SZiFi it0, SZiFi's reported pixel (no -1/2), q>5", out)

# ---------------- noise-only tiles (check c)
NOISE = [0, 20, 40, 60, 80, 99]
for mode in V.MODES:
    rs = [(p, load(mode, "noise", p), V.load_npz(os.path.join(V.DATA3, "noise_only", f"noise_{p:03d}.npz"))) for p in NOISE]
    rs = [x for x in rs if x[1] is not None]
    if not rs:
        continue
    area = len(rs) * 100.0
    sA = F.Filter(G, mode=mode, analytic=True).sigma_arcmin2
    print(f"\n==================== {mode}: {len(rs)} noise-only tiles ({area:.0f} deg^2)")
    a0 = np.array([r["sig0"] / r["sig_ana"] for _, r, _ in rs]); a1 = np.array([r["sig1"] / r["sig_ana"] for _, r, _ in rs])
    b0 = np.array([st[f"{mode}_whitened_it0_sig"] / sA for _, _, st in rs])
    b1 = np.array([st[f"{mode}_whitened_it1_sig"] / sA for _, _, st in rs])
    print("    sigma / analytic, theta [']               " + " ".join(f"{t:7.2f}" for t in TH[::2]))
    for lab, a in (("build it0", b0), ("build it1 (whitened)", b1), ("SZiFi it0", a0), ("SZiFi it1 (inpaint)", a1)):
        print(f"    {lab:40s}  " + " ".join(f"{x:7.4f}" for x in a.mean(0)[::2]))
    fs = np.array([r["fsky"] for _, r, _ in rs])
    print(f"    SZiFi unmasked fraction {fs.mean():.3f}; build {np.mean([st[f'{mode}_whitened_it1_fsky'] for _, _, st in rs]):.3f}")
    print("    false detections per 100 deg^2 (no merge), q > 4 / 4.5 / 5 [counts]:")
    for lab, getq in (("build it0", lambda r, st: st[f"{mode}_whitened_it0_q"]),
                      ("build it1 (final)", lambda r, st: st[f"{mode}_whitened_it1_q"]),
                      ("SZiFi it0 (= final without inpainting)", lambda r, st: r["det0_q"]),
                      ("SZiFi it1 (final with inpainting)", lambda r, st: r["det1_q"] if "det1_q" in r else r["det0_q"])):
        c = np.array([sum(np.sum(getq(r, st) > qc) for _, r, st in rs) for qc in (4.0, 4.5, 5.0)])
        print(f"    {lab:40s}: " + " / ".join(f"{100 * x / area:6.2f}" for x in c) + f"   {c.tolist()}")

# ---------------- injection tiles (check c)
INJ = [0, 30, 60, 90, 120, 180]
for mode in V.MODES:
    print(f"\n==================== {mode}: injection tiles {INJ} (one per size)")
    for k in INJ:
        r = load(mode, "inj", k)
        if r is None:
            continue
        inj = V.load_npz(os.path.join(V.DATA3, "injections", f"inj_{mode}_{k:03d}.npz"))
        th = float(inj["theta_inj"])
        yi, xi, qi = inj["y"] / V.PIXA, inj["x"] / V.PIXA, inj["q_inj"]
        keep = V.interior(yi, xi, MARGIN + 1.0)
        res = {}
        for lab, D in (("build final", dict(y=inj["d_iy"] * 1.0, x=inj["d_ix"] * 1.0, q=inj["d_q"], j=inj["d_j"])),
                       ("SZiFi it1", sz_det(r, 1)), ("SZiFi it0", sz_det(r, 0)),
                       ("SZiFi it1, reported pixel", dict(sz_det(r, 1), y=sz_det(r, 1)["ry"], x=sz_det(r, 1)["rx"]))):
            # highest-q detection within 1' of each injected cluster
            qobs = np.full(qi.size, -np.inf); off = np.full(qi.size, np.nan); dyx = np.full((qi.size, 2), np.nan)
            jj = np.full(qi.size, -1)
            for n in np.flatnonzero(keep):
                dd = np.hypot(D["y"] - yi[n], D["x"] - xi[n])
                m = np.flatnonzero(dd <= RAD)
                if m.size:
                    b = m[np.argmax(D["q"][m])]
                    qobs[n] = D["q"][b]; off[n] = dd[b] * V.PIXA; jj[n] = D["j"][b]
                    dyx[n] = ((D["y"][b] - yi[n]) * V.PIXA, (D["x"][b] - xi[n]) * V.PIXA)
            res[lab] = (qobs, off, dyx, jj)
        sel = keep & (qi >= 6) & (qi <= 9.5)
        s5 = keep & (qi >= 4.5) & (qi < 6)
        msg = []
        for lab, (qobs, off, dyx, jj) in res.items():
            det = sel & np.isfinite(qobs)
            dq = qobs[det] - qi[det]
            msg.append(f"{lab}: detected {det.sum()}/{sel.sum()}, q_obs - q_inj mean {dq.mean():+.2f} std {dq.std():.2f}, "
                       f"offset median {np.nanmedian(off[det]):.2f}' mean (dy,dx) ({np.nanmean(dyx[det, 0]):+.2f},"
                       f"{np.nanmean(dyx[det, 1]):+.2f})'; P(q_obs>5 | q_inj 4.5-6) {np.mean(qobs[s5] > 5):.2f}")
        print(f"  k = {k:3d}, theta_inj = {th}' ({keep.sum()} clusters >= margin): q_inj 6-9.5:")
        for m_ in msg:
            print(f"      {m_}")
        a, b = res["build final"][0], res["SZiFi it1"][0]
        both = keep & np.isfinite(a) & np.isfinite(b) & (qi >= 4)
        print(f"      same clusters, q_inj >= 4 found by both: N {both.sum()}, q_SZiFi it1 - q_build mean "
              f"{np.mean(b[both] - a[both]):+.3f}, std {np.std(b[both] - a[both]):.3f}")
print("\ndone")
