"""c04 (tests a and d): the verifier's own SZiFi runs against the build's stored runs, and how sensitive SZiFi's outputs
are to numerically irrelevant changes of the configuration.

(1) Parameter dictionaries: ours (vc_szifi.make_params, 'g' variants = the build's scale grid) against the dictionary
    stored with the build's runs (data_real/step3_szifi_real/runs/*.npz, params_json), all keys except paths.
(2) Bit-for-bit comparison of our runs with the build's runs of the same configuration and tile (catalogues of both
    iterations: q, y0, pixel, scale; sigma_vec; SZiFi's recorded spectra; unmasked fraction).
(3) Scale-grid representation: our 'S1' uses theta = exp(linspace(log 0.5, log 15, 15)) (SZiFi's own default form),
    'S1g' uses np.geomspace(0.5, 15, 15) (the build's); 6 of the 15 values differ by <= 5.6e-16 (relative).  Effect on
    iteration-0 q (common entries), scale assignment, the unmasked fraction f, sigma after the iteration and counts.
(4) Diagnosis: the same pair with the l range [100, 30600] (every mode in band, no empty spectrum bin), iteration 0;
    and SZiFi with map_dtype float64 (iteration 0).
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B c04_compare.py
"""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                   # noqa: E402
import vc_szifi as V                                 # noqa: E402  (only make_params and paths; SZiFi is not run here)

BR = "/Users/boris/GitHub/sz-poisson-blending/data_real/step3_szifi_real/runs"
MR = V.RUND


def load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def jl(v):
    return json.loads(json.dumps(v.tolist() if isinstance(v, np.ndarray) else
                                 (v.__name__ if isinstance(v, type) else
                                  (str(v) if isinstance(v, float) and not np.isfinite(v) else v))))


print("(1) parameter dictionaries: ours vs the build's (stored with its runs); paths excluded")
PAIRS_P = (("S0g", "S0_baseline_real_000"), ("S1g", "S1_baseline_real_000"), ("S2g", "S2_baseline_real_000"),
           ("S3g", "smokeS3_baseline_real_000"), ("S3ig", "S3i_baseline_real_000"), ("S3it0g", "S3it0_baseline_real_033"))
for mine, theirs in PAIRS_P:
    fn = os.path.join(BR, theirs + ".npz")
    if not os.path.exists(fn):
        print(f"  {mine}: build file {theirs} missing")
        continue
    pb = json.loads(str(load(fn)["params_json"]))
    pm = {k: jl(v) for k, v in V.make_params(mine).items()}
    diff = []
    for k in sorted(set(pb) | set(pm)):
        if k in ("path_data", "survey_file", "a_matrix", "cmmf_type"):
            continue
        a, b = pm.get(k, "(absent)"), pb.get(k, "(absent)")
        if a != b:
            diff.append(f"{k}: ours {a} | build {b}")
    print(f"  {mine:7s} vs {theirs:26s}: {len(pb)} keys; differences: {diff if diff else 'none'}")
    if "a_matrix" in pb:
        print(f"           build a_matrix (tSZ SED it used) = {np.ravel(pb['a_matrix']).round(6).tolist()}")


def cat_keys(r, it, build):
    sk = "j_theta" if build else "jt"
    return {(int(i), int(j), int(t)): (float(q), float(y)) for i, j, t, q, y in
            zip(r[f"it{it}_i"], r[f"it{it}_j"], r[f"it{it}_{sk}"], r[f"it{it}_q"], r[f"it{it}_y0"])}


def bitcompare(mine, theirs):
    a, b = load(os.path.join(MR, mine + ".npz")), load(os.path.join(BR, theirs + ".npz"))
    out = []
    for it in range(int(b["n_it"])):
        same = True
        for km, kb in (("q", "q"), ("y0", "y0"), ("i", "i"), ("j", "j"), ("jt", "j_theta"), ("sigvec", "sigvec")):
            x, y = a.get(f"it{it}_{km}"), b.get(f"it{it}_{kb}")
            same &= x is not None and y is not None and x.shape == y.shape and np.array_equal(x, y)
        out.append(f"it{it} catalogue+sigma {'identical' if same else 'DIFFERENT'} ({a[f'it{it}_q'].size} / {b[f'it{it}_q'].size} det)")
    for k in range(int(b.get("n_spec", 0))):
        out.append(f"spec{k} {'identical' if np.array_equal(a[f'spec{k}'], b[f'spec{k}']) else 'DIFFERENT (max rel %.1e)' % np.abs(a[f'spec{k}'] / b[f'spec{k}'] - 1).max()}")
    if "rec_fsky" in b and b["rec_fsky"].size:
        out.append(f"f {a['fsky'].tolist()} / {b['rec_fsky'].tolist()}")
    return "; ".join(out)


print("\n(2) our runs (own loader + driver) vs the build's stored runs, bit for bit")
PAIRS = [("S1g_real_baseline_000", "S1_baseline_real_000"), ("S1g_real_baseline_033", "S1_baseline_real_033"),
         ("S1g_real_goal_000", "S1_goal_real_000"), ("S1g_real_goal_033", "S1_goal_real_033"),
         ("S2g_real_baseline_000", "S2_baseline_real_000"), ("S2g_real_baseline_033", "S2_baseline_real_033"),
         ("S0g_real_baseline_000_b0", "S0_baseline_real_000"), ("S0g_real_baseline_033_b0", "S0_baseline_real_033"),
         ("S3it0g_real_baseline_033", "S3it0_baseline_real_033"),
         ("S3ig_real_baseline_000_s1000003", "S3i_baseline_real_000")]
for mine, theirs in PAIRS:
    if not os.path.exists(os.path.join(MR, mine + ".npz")):
        print(f"  {mine}: not run")
        continue
    print(f"  {mine:32s} vs build {theirs:22s}: {bitcompare(mine, theirs)}")

print("\n(3) scale grid exp(linspace) ('S1') vs geomspace ('S1g' = build); values differ by <= 5.6e-16")
th_a = V.M.THETAS                                     # exp(linspace(log 0.5, log 15, 15)), our default form
th_b = np.geomspace(0.5, 15.0, 15)
print(f"  scales whose value differs in the last bit: {np.flatnonzero(th_a != th_b).tolist()}")
rows = []
for tile in ("real_baseline_000", "real_baseline_033", "real_goal_000", "real_goal_033"):
    fa, fb = os.path.join(MR, f"S1_{tile}.npz"), os.path.join(MR, f"S1g_{tile}.npz")
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print(f"  {tile}: pair not complete")
        continue
    a, b = load(fa), load(fb)
    ka, kb = cat_keys(a, 0, False), cat_keys(b, 0, False)
    com = sorted(set(ka) & set(kb))
    dq = np.array([ka[k][0] - kb[k][0] for k in com])
    sc = np.array([k[2] for k in com])
    changed = sorted({s for s, d in zip(sc, dq) if d != 0})
    pa = {(k[0], k[1]): k[2] for k in ka}
    pb = {(k[0], k[1]): k[2] for k in kb}
    cp = set(pa) & set(pb)
    nflip = sum(pa[p] != pb[p] for p in cp)
    dd = dq[np.isin(sc, changed)] if changed else np.zeros(0)
    q5a, q5b = (a["it1_q"] > 5).sum(), (b["it1_q"] > 5).sum()
    rs = a["it1_sigvec"][1:] / b["it1_sigvec"][1:]
    print(f"  {tile}: it0 det {a['it0_q'].size} / {b['it0_q'].size}; q of common entries identical except at scales "
          f"{changed}: there dq sd {dd.std() if dd.size else 0:.4f}, max |dq| {np.abs(dd).max() if dd.size else 0:.4f} ({dd.size} entries); "
          f"same pixel, other scale: {nflip} of {len(cp)}; f {a['fsky'][0]:.4f} / {b['fsky'][0]:.4f}; "
          f"sigma_it1 ratio at 0.64'/4.45'/15' {rs[0]:.4f}/{rs[8]:.4f}/{rs[13]:.4f}; final q>5 {q5a} / {q5b}")
    rows.append((a["fsky"][0] - b["fsky"][0], rs[0] - 1, rs[8] - 1, rs[13] - 1, q5a / q5b - 1))
if rows:
    R = np.array(rows)
    print(f"  over {len(rows)} tiles: f(S1) - f(S1g) = " + " ".join(f"{x:+.4f}" for x in R[:, 0])
          + f"; sigma_it1(S1)/sigma_it1(S1g) - 1 at 0.64' = " + " ".join(f"{x:+.4f}" for x in R[:, 1])
          + f"; at 15' = " + " ".join(f"{x:+.4f}" for x in R[:, 3])
          + f"; final q>5 count ratio - 1 = " + " ".join(f"{x:+.3f}" for x in R[:, 4]))
    print(f"  rms over tiles: |df| {np.sqrt((R[:, 0] ** 2).mean()):.4f}; sigma_it1 at 0.64' {np.sqrt((R[:, 1] ** 2).mean()):.4f}, "
          f"4.45' {np.sqrt((R[:, 2] ** 2).mean()):.4f}, 15' {np.sqrt((R[:, 3] ** 2).mean()):.4f}")

print("\n(4) diagnosis")
fa, fb = os.path.join(MR, "S1wit0_real_baseline_000.npz"), os.path.join(MR, "S1wit0g_real_baseline_000.npz")
if os.path.exists(fa) and os.path.exists(fb):
    a, b = load(fa), load(fb)
    ka, kb = cat_keys(a, 0, False), cat_keys(b, 0, False)
    com = sorted(set(ka) & set(kb))
    dq = np.array([ka[k][0] - kb[k][0] for k in com])
    print(f"  l range [100, 30600] (no empty bins), iteration 0, tile 0 baseline: exp(linspace) vs geomspace: "
          f"{a['it0_q'].size} / {b['it0_q'].size} detections, {len(com)} common (pixel, scale); max |dq| {np.abs(dq).max():.2e}; "
          f"catalogues identical: {set(ka) == set(kb)}; sigma_vec max rel diff "
          f"{np.abs(a['it0_sigvec'][1:] / b['it0_sigvec'][1:] - 1).max():.1e}")
    s = a["spec0"]
    e = a["spec0_ell"]
    print(f"  spectra of that run at l = 20052 / 21564 (now in band): C_00 = {s[np.argmin(abs(e - 20052)), 0, 0]:.3e} / "
          f"{s[-1, 0, 0]:.3e} uK^2 sr")
for tag in ("S1g_real_baseline_000", "S1it0gf64_real_baseline_000"):
    fn = os.path.join(MR, tag + ".npz")
    if os.path.exists(fn):
        r = load(fn)
        e, s = r["spec0_ell"], r["spec0"]
        hi = e > 20016
        lo = (e > 3000) & (e < 20000)
        ratio = np.median(np.array([s[k].diagonal() for k in np.flatnonzero(hi)]), 0) / \
            np.median(np.array([s[k].diagonal() for k in np.flatnonzero(lo)]), 0)
        q = r["it0_q"]
        print(f"  {tag}: {hi.sum()} bins above l = 20016 hold only round-off: median auto spectrum there / median at "
              f"3000-20000 = " + " ".join(f"{x:.1e}" for x in ratio)
              + f"; iteration 0: {q.size} detections, median q {np.median(q):.3g}, sigma_vec[14] {r['it0_sigvec'][14]:.4e}")
ell = np.sqrt((np.fft.fftfreq(1200, V.M.DX) * 2 * np.pi)[:, None] ** 2 + (np.fft.fftfreq(1200, V.M.DX) * 2 * np.pi)[None, :] ** 2)
print(f"  modes of the 1200^2 grid with l > 20000 (outside SZiFi's l range, covariance from empty bins or the last bin "
      f"by extrapolation): {(ell > 20000).mean():.3f} of all modes")

print("\n(5) is SZiFi's larger iteration-1 inflation a property of the method or of the empty bins above l = 20000?")
print("    sigma_it1/sigma_it0 at 0.64' / 4.45' / 15', unmasked fraction f and iteration-0 detections at theta >= 7.24'")
print("    (q > 4, all of the tile inside the 8' buffer) for: S1g (= build), S1 (exp-linspace scales), S1wg (l range")
print("    [100, 30600]: no empty bins; the extra modes carry noise only and no template power at large theta), against")
print("    the arbiter's a01 'szifi' emulation of the same tile (stored, finder template, periodic tile) and the production")
print("    finder's iteration-0 catalogue (stored, periodic tile, whole tile).")
A01 = "/Users/boris/GitHub/sz-poisson-blending/data_real/step3/arbiter"
TIL = "/Users/boris/GitHub/sz-poisson-blending/data_real/step3/tiles"
print("    (S1g = our run where present, else the build's stored S1 run of that tile, which our S1g reproduces bit for bit)")
summ = {"baseline": [], "goal": []}
for mode in ("baseline", "goal"):
    for pid in range(0, 324, 11):
        fw = os.path.join(MR, f"S1wg_real_{mode}_{pid:03d}.npz")
        if not os.path.exists(fw):
            continue
        a01 = load(os.path.join(A01, f"a01_real_{pid:03d}.npz"))
        tl = load(os.path.join(TIL, f"tile_{pid:03d}.npz"))
        r01 = a01[f"{mode}_szifi_sig1"] / a01[f"{mode}_szifi_sig0"]
        nfin = int(((tl[f"{mode}_it0_q"] > 4) & (tl[f"{mode}_it0_j"] >= 11)).sum())
        line = [f"a01: {r01[1]:.4f}/{r01[9]:.4f}/{r01[14]:.4f}, f {float(a01[f'{mode}_szifi_fsky']):.4f}; finder theta>=7.24': {nfin}"]
        row = [r01[1], r01[9], r01[14], float(a01[f"{mode}_szifi_fsky"]), nfin]
        for cfg in ("S1g", "S1", "S1wg"):
            fn = os.path.join(MR, f"{cfg}_real_{mode}_{pid:03d}.npz")
            build = False
            if not os.path.exists(fn) and cfg == "S1g":
                fn, build = os.path.join(BR, f"S1_{mode}_real_{pid:03d}.npz"), True
            if not os.path.exists(fn):
                line.append(f"{cfg}: not run")
                row += [np.nan] * 5
                continue
            r = load(fn)
            sv0 = r["it0_sigvec"] if not build else r["it0_sigvec"]
            rr = r["it1_sigvec"] / sv0
            jt = r["it0_jt"] if not build else r["it0_j_theta"]
            fs = r["fsky"][0] if not build else r["rec_fsky"][0]
            nl = int(((r["it0_q"] > 4) & (jt >= 11)).sum())
            line.append(f"{cfg}{'(build)' if build else ''}: {rr[1]:.4f}/{rr[9]:.4f}/{rr[14]:.4f}, f {fs:.4f}, theta>=7.24': {nl}")
            row += [rr[1], rr[9], rr[14], fs, nl]
        summ[mode].append(row)
        print(f"  {mode:8s} tile {pid:3d}: " + "; ".join(line))
for mode in ("baseline", "goal"):
    if not summ[mode]:
        continue
    A = np.array(summ[mode], float)
    n = A.shape[0]
    def ms(c):
        x = A[:, c]
        x = x[np.isfinite(x)]
        return f"{x.mean():.4f} (+- {x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else 0:.4f}, n={x.size})"
    print(f"  {mode} means over {n} tiles: sigma_it1/sigma_it0 at 0.64': a01 {ms(0)}, S1g {ms(5)}, S1wg {ms(15)}; at 4.45': a01 {ms(1)}, "
          f"S1g {ms(6)}, S1wg {ms(16)}; at 15': a01 {ms(2)}, S1g {ms(7)}, S1wg {ms(17)}")
    print(f"  {mode} means: f a01 {ms(3)}, S1g {ms(8)}, S1wg {ms(18)}; iteration-0 q > 4 detections at theta >= 7.24': finder {ms(4)}, "
          f"S1g {ms(9)}, S1wg {ms(19)}")
    d = A[:, 5] - A[:, 15]
    print(f"  {mode} paired S1g - S1wg in sigma_it1/sigma_it0 at 0.64': " + " ".join(f"{x:+.4f}" for x in d)
          + f"; at 4.45': " + " ".join(f"{x:+.4f}" for x in A[:, 6] - A[:, 16]))

print("\n(6) where the extra large-scale detections come from, and the final counts (interior: >= 9' from the edges)")
print("    'stacked' = an iteration-0 detection at theta >= 7.24' with another detection of the same run within 2 pixels at a")
print("    smaller scale (SZiFi keeps every local maximum along the scale axis, so one object can give two entries)")
for mode in ("baseline", "goal"):
    rows = []
    for pid in range(0, 324, 11):
        fw = os.path.join(MR, f"S1wg_real_{mode}_{pid:03d}.npz")
        if not os.path.exists(fw):
            continue
        tl = load(os.path.join(TIL, f"tile_{pid:03d}.npz"))
        inn = lambda i, j: (i >= 18) & (i <= 1181) & (j >= 18) & (j <= 1181)          # noqa: E731
        pf = (tl[f"{mode}_it1_q"] > 5) & inn(tl[f"{mode}_it1_iy"], tl[f"{mode}_it1_ix"])
        row = [int(pf.sum())]
        for cfg, fn, build in (("S1g", os.path.join(MR, f"S1g_real_{mode}_{pid:03d}.npz"), False),
                               ("S1wg", fw, False)):
            if not os.path.exists(fn):
                fn, build = os.path.join(BR, f"S1_{mode}_real_{pid:03d}.npz"), True
            r = load(fn)
            jt = r["it0_j_theta"] if build else r["it0_jt"]
            i0, j0, q0 = r["it0_i"], r["it0_j"], r["it0_q"]
            big = np.flatnonzero((q0 > 4) & (jt >= 11))
            stacked = sum(int(np.any((np.abs(i0 - i0[b]) <= 2) & (np.abs(j0 - j0[b]) <= 2) & (jt < jt[b]))) for b in big)
            n0 = int(((q0 > 5) & inn(i0, j0)).sum())
            n1 = int(((r["it1_q"] > 5) & inn(r["it1_i"], r["it1_j"])).sum())
            row += [len(big), stacked, n0, n1]
        rows.append(row)
    A = np.array(rows, float)
    print(f"  {mode} ({len(rows)} tiles, sums): theta >= 7.24' detections at iteration 0: S1g {A[:, 1].sum():.0f} of which stacked "
          f"{A[:, 2].sum():.0f}; S1wg {A[:, 5].sum():.0f} of which stacked {A[:, 6].sum():.0f}")
    print(f"  {mode}: q > 5 counts, iteration 0 -> 1: S1g {A[:, 3].sum():.0f} -> {A[:, 4].sum():.0f} ({A[:, 4].sum() / A[:, 3].sum() - 1:+.3f}); "
          f"S1wg {A[:, 7].sum():.0f} -> {A[:, 8].sum():.0f} ({A[:, 8].sum() / A[:, 7].sum() - 1:+.3f}); final count / production finder's: "
          f"S1g {A[:, 4].sum() / A[:, 0].sum():.3f}, S1wg {A[:, 8].sum() / A[:, 0].sum():.3f} (production {A[:, 0].sum():.0f})")

print("\n(7) SZiFi's own auto spectra after / before the iteration (mean over the six channels and the bins), same tiles")
for mode in ("baseline", "goal"):
    out = {"S1g": [], "S1wg": []}
    for pid in range(0, 324, 11):
        fw = os.path.join(MR, f"S1wg_real_{mode}_{pid:03d}.npz")
        if not os.path.exists(fw):
            continue
        for cfg in ("S1g", "S1wg"):
            fn = os.path.join(MR, f"{cfg}_real_{mode}_{pid:03d}.npz")
            if not os.path.exists(fn):
                fn = os.path.join(BR, f"S1_{mode}_real_{pid:03d}.npz")
            r = load(fn)
            e = r["spec1_ell"]
            rr = np.array([r["spec1"][k].diagonal() / r["spec0"][k].diagonal() for k in range(len(e))])
            out[cfg].append([rr[(e > lo) & (e <= hi)].mean() for lo, hi in ((100, 3000), (3000, 20000))])
    for cfg in out:
        A = np.array(out[cfg])
        print(f"  {mode:8s} {cfg:5s} ({len(A)} tiles): l 100-3000 {A[:, 0].mean():.3f}; l 3000-20000 {A[:, 1].mean():.3f} "
              f"(tile range {A[:, 1].min():.3f}-{A[:, 1].max():.3f})")
