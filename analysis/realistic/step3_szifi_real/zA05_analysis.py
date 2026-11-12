"""zA05 (arbiter): why SZiFi's default iteration (S1) ends above step 3a's emulation (a01), from SZiFi's own runs.

Runs (data_real/step3_szifi_real/arbiter/runs/, zA03 with zA_lib; 12 realistic tiles per noise level, pid 0, 11, 22,
33, 44, 55, 88, 132, 176, 220, 264, 308, and 5 noise-only tiles, pid 0, 22, 44, 66, 88):
  S1   : the build's stored runs (as configured); one arbiter rerun (tile 0, baseline) checks the reproduction;
  S1z  : S1 with SZiFi's inverse covariance set to 0 outside lrange (removes the amplified round-off; zA_lib);
  S1p  : S1 with SZiFi's inpainting and blend done on the periodically padded tile (zA_lib);
  S1zp : both;
  S1zL : S1z with the scale grid built as exp(linspace) (last-bit differences; tile 0 baseline).
  S1wg : the config check's runs with lrange [100, 30600] (no empty bins; tiles 0-55), read from
         verify_real/step3_szifi_real_config/out/runs/ (only sigma ratios, f and counts are used: their band differs).
Comparisons: step 3a's arbiter emulation a01 ('szifi': finder template, periodic tile) and the production finder on the
same tiles.  Conventions as z08 (arbiter-fixed): r = sigma / sigma_analytic of each filter's own template (z04 for
SZiFi, sigma_Y for a01 and the finder); interior = equal-area window >= 9' from the edges; SZiFi positions - 1/2 pixel.
Errors: error of the mean over tiles (paired where stated) or 2000 tile-bootstrap resamples for count ratios.
Usage: python -I -B zA05_analysis.py
"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial import cKDTree

t0 = time.time()
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(ROOT, "data_real", "step3_szifi_real")
ARUN, BRUN = os.path.join(OUT, "arbiter", "runs"), os.path.join(OUT, "runs")
CRUN = os.path.join(ROOT, "verify_real", "step3_szifi_real_config", "out", "runs")
A01 = os.path.join(ROOT, "data_real", "step3", "arbiter")
N, PIX, SHIFT = 1200, 0.5, 0.5
MODES = ("baseline", "goal")
REF = dict(np.load(os.path.join(OUT, "reference.npz")))
ANA = dict(np.load(os.path.join(OUT, "analytic_sigma.npz")))
EB = dict(np.load(os.path.join(OUT, "edge_buffer.npz")))
M_INT = float(EB["interior_arcmin"])
AREA = ((N * PIX - 2 * M_INT) / 60.0) ** 2
TH = REF["thetas"]
PR = [0, 11, 22, 33, 44, 55, 88, 132, 176, 220, 264, 308]
PN = [0, 22, 44, 66, 88]
J3 = [0, 9, 14]                    # 0.5', 4.45', 15'
rng = np.random.default_rng(20261112)
SUM = {}


def load(path):
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


_C = {}


def run(var, mode, kind, pid):
    key = (var, mode, kind, pid)
    if key not in _C:
        if var == "S1":
            _C[key] = load(os.path.join(BRUN, f"S1_{mode}_{kind}_{pid:03d}.npz"))
        elif var == "S1wg":
            r = load(os.path.join(CRUN, f"S1wg_{kind}_{mode}_{pid:03d}.npz"))
            if r is not None:
                for it in (0, 1):
                    r[f"it{it}_j_theta"] = r[f"it{it}_jt"]
                    r[f"it{it}_sig"] = r[f"it{it}_sigvec"].copy()
                r["rec_fsky"] = r["fsky"]
            _C[key] = r
        else:
            _C[key] = load(os.path.join(ARUN, f"{var}_{mode}_{kind}_{pid:03d}.npz"))
    return _C[key]


_T = {}


def tile(pid):
    if pid not in _T:
        _T[pid] = load(os.path.join(ROOT, "data_real", "step3", "tiles", f"tile_{pid:03d}.npz"))
    return _T[pid]


def a01(kind, pid):
    return load(os.path.join(A01, f"a01_{kind}_{pid:03d}.npz"))


FN = load(os.path.join(OUT, "finder_noise.npz"))


def finder(mode, kind, pid, it):
    if kind == "real":
        t = tile(pid)
        g = lambda k: t[f"{mode}_it{it}_{k}"]
    else:
        g = lambda k: FN[f"{mode}_{pid:03d}_it{it}_{k}"]
    return dict(y=g("iy") * 1.0, x=g("ix") * 1.0, q=g("q"), j=g("j"))


def sz(r, it):
    return dict(y=r[f"it{it}_i"] - SHIFT, x=r[f"it{it}_j"] - SHIFT, q=r[f"it{it}_q"], j=r[f"it{it}_j_theta"])


CATK = ("y", "x", "q", "j")


def inner(c, m=M_INT):
    lo, hi = m / PIX - 0.25, N - 1 - m / PIX + 0.75
    s = (c["y"] >= lo) & (c["y"] < hi) & (c["x"] >= lo) & (c["x"] < hi)
    return {k: (v[s] if k in CATK else v) for k, v in c.items()}


def count(c, qc):
    return int(np.sum(inner(c)["q"] > qc))


def match(A, B, qc, rad=2.0, dj=None):
    a = inner(A); a = {k: v[a["q"] > qc] for k, v in a.items()}
    b = inner(B, M_INT - 1.0); b = {k: v[b["q"] > 4] for k, v in b.items()}
    n = a["q"].size
    k = np.full(n, -1)
    if n and b["q"].size:
        tr = cKDTree(np.c_[b["y"], b["x"]])
        for i_, c in enumerate(tr.query_ball_point(np.c_[a["y"], a["x"]], rad + 1e-9)):
            c = np.asarray(c, dtype=np.int64)
            if dj is not None and c.size:
                c = c[np.abs(b["j"][c] - a["j"][i_]) <= dj]
            if c.size:
                k[i_] = c[np.argmin(np.hypot(b["y"][c] - a["y"][i_], b["x"][c] - a["x"][i_]))]
    ok = k >= 0
    return n, int(ok.sum()), b["q"][k[ok]] - a["q"][ok]


def boot_ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    b = rng.integers(0, num.size, (2000, num.size))
    return num.sum() / den.sum(), (num[b].sum(1) / den[b].sum(1)).std()


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.4f} +- {x.std(ddof=1) / np.sqrt(x.size):.4f}"


def big(r, it=0):
    return int(np.sum((r[f"it{it}_q"] > 4) & (r[f"it{it}_j_theta"] >= 11)))


def stacked(r, it=0):
    i, j, q, t = r[f"it{it}_i"], r[f"it{it}_j"], r[f"it{it}_q"], r[f"it{it}_j_theta"]
    n = 0
    for k in np.flatnonzero((t >= 11) & (q > 4)):
        n += bool(np.any((np.abs(i - i[k]) <= 2) & (np.abs(j - j[k]) <= 2) & (t < t[k])))
    return n


def sratio(r, lo, hi):
    e0 = r["spec0_ell"]
    b = (e0 >= lo) & (e0 < hi)
    return np.mean([np.mean(r["spec1"][b, i, i] / r["spec0"][b, i, i]) for i in range(6)])


def rsz(r, it, mode):
    return r[f"it{it}_sig"] / ANA[f"{mode}_S0"]


print(f"zA05: arbiter runs of SZiFi itself; interior {AREA:.2f} deg^2 per tile; tiles {PR} (real), {PN} (noise-only)")
# ------------------------------------------------------------------------------------------------ (0) reproduction
print("\n(0) REPRODUCTION: the arbiter's S1 rerun of tile 0 (baseline) vs the build's stored S1 run")
a, b = load(os.path.join(ARUN, "S1_baseline_real_000.npz")), run("S1", "baseline", "real", 0)
keys = [k for k in b if k.startswith(("it0_", "it1_", "spec"))] + ["rec_fsky"]
same = [k for k in keys if np.array_equal(a[k], b[k])]
print(f"    {len(same)} of {len(keys)} arrays identical (catalogues, sigma, SZiFi's spectra of both iterations, f); "
      f"differing: {sorted(set(keys) - set(same))}")
SUM["repro"] = (len(same), len(keys))
for va, vb in (("S1p", "S1"), ("S1zp", "S1z")):
    n_ok = n_all = 0
    for mode in MODES:
        for p in PR:
            ra, rb = run(va, mode, "real", p), run(vb, mode, "real", p)
            if ra is None or rb is None:
                continue
            n_all += 1
            n_ok += all(np.array_equal(ra[f"it0_{k}"], rb[f"it0_{k}"]) for k in ("q", "i", "j", "j_theta", "sig")) and \
                np.array_equal(ra["spec0"], rb["spec0"]) and np.array_equal(ra["rec_fsky"], rb["rec_fsky"])
    print(f"    {va} iteration 0 and cluster mask identical to {vb}'s (the periodic wrappers act only at iteration 1): "
          f"{n_ok} of {n_all} tile runs")

# ------------------------------------------------------------------------------------------------ (1) iteration 0
print("\n(1) ITERATION 0: SZiFi as configured (S1) vs with its inverse covariance zeroed outside lrange (S1z)")
for mode in MODES:
    s1 = {p: run("S1", mode, "real", p) for p in PR}
    sz_ = {p: run("S1z", mode, "real", p) for p in PR}
    if any(v is None for v in sz_.values()):
        print(f"  {mode}: S1z runs missing"); continue
    nd = np.array([[s1[p]["it0_q"].size, sz_[p]["it0_q"].size] for p in PR])
    com, tot, dqs = 0, 0, []
    for p in PR:
        ka = {(i, j, t): q for i, j, t, q in zip(s1[p]["it0_i"], s1[p]["it0_j"], s1[p]["it0_j_theta"], s1[p]["it0_q"])}
        kb = {(i, j, t): q for i, j, t, q in zip(sz_[p]["it0_i"], sz_[p]["it0_j"], sz_[p]["it0_j_theta"], sz_[p]["it0_q"])}
        c = set(ka) & set(kb)
        com += len(c); tot += len(set(ka) | set(kb))
        dqs += [kb[k] - ka[k] for k in c]
    dqs = np.array(dqs)
    sr = max(np.abs(sz_[p]["it0_sig"] / s1[p]["it0_sig"] - 1).max() for p in PR)
    bg = np.array([[big(s1[p]), big(sz_[p]), stacked(s1[p]), stacked(sz_[p])] for p in PR])
    fb = np.array([np.sum((tile(p)[f"{mode}_it0_q"] > 4) & (tile(p)[f"{mode}_it0_j"] >= 11)) for p in PR])
    print(f"  {mode}: q > 4 detections per tile (inside the 8' buffer) S1 {nd[:, 0].mean():.1f}, S1z {nd[:, 1].mean():.1f}; "
          f"same (pixel, scale) entries {com} of {tot} ({100 * com / tot:.1f}%); on those dq sd {dqs.std():.4f}, max |dq| "
          f"{np.abs(dqs).max():.3f}; sigma S1z / S1 - 1: max |.| {sr:.1e}")
    print(f"     q > 4 detections at theta >= 7.24' per tile: S1 {bg[:, 0].mean():.1f} (of which 'stacked' within 2 px of a "
          f"smaller-scale entry {bg[:, 2].mean():.1f}), S1z {bg[:, 1].mean():.1f} (stacked {bg[:, 3].mean():.1f}), "
          f"production finder {fb.mean():.1f}")
    SUM[f"it0_{mode}"] = dict(nd=nd.mean(0).tolist(), common=com / tot, dq_sd=dqs.std(), dq_max=np.abs(dqs).max(),
                              big=bg.mean(0).tolist(), big_finder=fb.mean())
    for var, R in (("S1", s1), ("S1z", sz_)):
        parts = []
        for qc in (4.5, 5.0):
            for dj in (None, 1):
                m1 = [match(finder(mode, "real", p, 0), sz(R[p], 0), qc, dj=dj) for p in PR]
                m2 = [match(sz(R[p], 0), finder(mode, "real", p, 0), qc, dj=dj) for p in PR]
                n1, k1 = sum(m[0] for m in m1), sum(m[1] for m in m1)
                n2, k2 = sum(m[0] for m in m2), sum(m[1] for m in m2)
                dq = np.concatenate([m[2] for m in m1])
                tag = "any scale" if dj is None else "one scale step"
                parts.append(f"q>{qc} {tag}: {100 * k1 / n1:.2f}/{100 * k2 / n2:.2f}%" +
                             (f", dq {dq.mean():+.3f}+-{dq.std():.3f}" if dj is None else ""))
                SUM[f"it0match_{var}_{mode}_{qc}_{dj}"] = (100 * k1 / n1, 100 * k2 / n2, dq.mean(), dq.std())
        print(f"     {var:4s} vs production finder (finder->SZiFi / SZiFi->finder): " + "; ".join(parts))
    sY = REF[f"{mode}_sigY"]
    d0 = np.array([rsz(sz_[p], 0, mode) - tile(p)[f"{mode}_it0_sig"] / sY for p in PR])
    print(f"     r_it0(S1z) - r_it0(finder) at 0.5' / 4.45' / 15': {d0.mean(0)[0]:+.4f} / {d0.mean(0)[9]:+.4f} / "
          f"{d0.mean(0)[14]:+.4f}")

# ------------------------------------------------------------------------------------------------ (2) scale grid
print("\n(2) LAST-BIT FRAGILITY: scale grid exp(linspace) (S1zL) vs geomspace (S1z), tile 0 baseline, both iterations")
a, b = run("S1zL", "baseline", "real", 0), run("S1z", "baseline", "real", 0)
if a is not None and b is not None:
    for it in (0, 1):
        ka = {(i, j, t): q for i, j, t, q in zip(a[f"it{it}_i"], a[f"it{it}_j"], a[f"it{it}_j_theta"], a[f"it{it}_q"])}
        kb = {(i, j, t): q for i, j, t, q in zip(b[f"it{it}_i"], b[f"it{it}_j"], b[f"it{it}_j_theta"], b[f"it{it}_q"])}
        c = set(ka) & set(kb)
        dq = np.array([ka[k] - kb[k] for k in c])
        print(f"    iteration {it}: {len(ka)} / {len(kb)} detections, {len(c)} common (pixel, scale) entries, max |dq| "
              f"{np.abs(dq).max():.1e}; sigma max rel. diff {np.abs(a[f'it{it}_sig'] / b[f'it{it}_sig'] - 1).max():.1e}")
    print(f"    f {a['rec_fsky'][0]:.6f} / {b['rec_fsky'][0]:.6f}")
    SUM["lastbit"] = float(np.abs(dq).max()) if dq.size else 0.0

# ------------------------------------------------------------------------------------------------ (3) iteration 1
print("\n(3) ITERATION 1 (final catalogue), 12 realistic tiles per noise level")
VARS = ("S1", "S1z", "S1p", "S1zp")
LR5 = ((100, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 20000))
CL = {"baseline": (1.135, 1.119, 1.050), "goal": (1.181, 1.172, 1.078)}
R1 = {}
for mode in MODES:
    sY = REF[f"{mode}_sigY"]
    A = {p: a01("real", p) for p in PR}
    a1 = np.array([A[p][f"{mode}_szifi_sig1"] / sY for p in PR])
    a0 = np.array([A[p][f"{mode}_szifi_sig0"] / sY for p in PR])
    af = np.array([A[p][f"{mode}_szifi_fsky"] for p in PR])
    print(f"\n  ---- {mode}: a01 emulation on these tiles: r after {a1.mean(0)[0]:.4f} / {a1.mean(0)[9]:.4f} / "
          f"{a1.mean(0)[14]:.4f}; it1/it0 at 0.64' {np.mean(a1[:, 1] / a0[:, 1]):.4f}; f {af.mean():.4f} "
          f"(step 3a's claim, 30 tiles: {CL[mode]})")
    R1[(mode, "a01")] = a1
    SUM[f"a01_{mode}"] = dict(r1=a1.mean(0).tolist(), f=af.mean(), infl=np.mean(a1[:, 1] / a0[:, 1]))
    pd = np.array([count(finder(mode, "real", p, 1), 5.0) for p in PR])
    for var in VARS:
        R = {p: run(var, mode, "real", p) for p in PR}
        if any(v is None for v in R.values()):
            print(f"    {var}: runs missing ({sum(v is None for v in R.values())})"); continue
        r0 = np.array([rsz(R[p], 0, mode) for p in PR]); r1 = np.array([rsz(R[p], 1, mode) for p in PR])
        R1[(mode, var)] = r1
        fs = np.array([R[p]["rec_fsky"][0] for p in PR])
        ec_src = {"S1": "S1p", "S1z": "S1zp"}.get(var, var)       # S1p / S1zp have S1's / S1z's cluster mask
        ec = np.array([run(ec_src, mode, "real", p)["edge_cov"] if run(ec_src, mode, "real", p) is not None else np.nan
                       for p in PR])
        infl = r1[:, 1] / r0[:, 1]
        sp = np.array([[sratio(R[p], 100, 3000), sratio(R[p], 3000, 20000)] for p in PR])
        n0 = np.array([count(sz(R[p], 0), 5.0) for p in PR]); n1 = np.array([count(sz(R[p], 1), 5.0) for p in PR])
        rc, ec_ = boot_ratio(n1, n0)
        rp, ep = boot_ratio(n1, pd)
        bg = np.array([big(R[p]) for p in PR])
        print(f"    {var:5s}: r after {r1.mean(0)[0]:.4f} / {r1.mean(0)[9]:.4f} / {r1.mean(0)[14]:.4f} (+- "
              f"{r1.std(0, ddof=1)[0] / np.sqrt(12):.4f} / {r1.std(0, ddof=1)[9] / np.sqrt(12):.4f} / "
              f"{r1.std(0, ddof=1)[14] / np.sqrt(12):.4f}); paired - a01: {ms(r1[:, 0] - a1[:, 0])} / "
              f"{ms(r1[:, 9] - a1[:, 9])} / {ms(r1[:, 14] - a1[:, 14])}")
        print(f"           it1/it0 at 0.64' {infl.mean():.4f} (paired - a01 {ms(infl - a1[:, 1] / a0[:, 1])}); f {fs.mean():.4f} "
              f"(paired - a01 {ms(fs - af)}); boundary pixels in the mask {np.nanmean(ec):.3f}; theta >= 7.24' it0 "
              f"{bg.mean():.1f} per tile")
        s5 = np.array([[sratio(R[p], lo, hi) for lo, hi in LR5] for p in PR])
        print(f"           spectra it1/it0 in l {LR5}: " + " ".join(f"{v:.4f}" for v in s5.mean(0)) +
              "  (a01, same tiles: " + " ".join(f"{v:.4f}" for v in np.mean([A[p][f'{mode}_szifi_spec'] for p in PR], 0)) + ")")
        print(f"           spectra it1/it0, l 100-3000 / 3000-20000: {sp.mean(0)[0]:.4f} / {sp.mean(0)[1]:.4f}; q > 5 "
              f"per tile it0 -> it1 {n0.mean():.1f} -> {n1.mean():.1f} ({100 * (rc - 1):+.1f} +- {100 * ec_:.1f}%); final "
              f"q > 5 / production finder {rp:.4f} +- {ep:.4f} ({n1.sum()} / {pd.sum()})")
        SUM[f"{var}_{mode}"] = dict(r1=r1.mean(0).tolist(), r1_sem=(r1.std(0, ddof=1) / np.sqrt(12)).tolist(),
                                    d_a01=(r1 - a1).mean(0).tolist(), d_a01_sem=((r1 - a1).std(0, ddof=1) / np.sqrt(12)).tolist(),
                                    infl=infl.mean(), f=fs.mean(), edge=np.nanmean(ec), spec=sp.mean(0).tolist(),
                                    n0=n0.mean(), n1=n1.mean(), dn=rc - 1, dn_err=ec_, prod=rp, prod_err=ep, big=bg.mean())
    an0 = np.array([A[p][f"{mode}_szifi_n0"][2] for p in PR]); an1 = np.array([A[p][f"{mode}_szifi_n1"][2] for p in PR])
    rc, ec_ = boot_ratio(an1, an0)
    print(f"    a01 q > 5 per tile (whole periodic tile) it0 -> it1: {an0.mean():.1f} -> {an1.mean():.1f} "
          f"({100 * (rc - 1):+.1f} +- {100 * ec_:.1f}%)")
    SUM[f"a01counts_{mode}"] = (rc - 1, ec_)

# ------------------------------------------------------------------------------------------------ (4) decomposition
print("\n(4) DECOMPOSITION of SZiFi's excess over a01 (paired over the 12 tiles; r units at 0.5' / 4.45' / 15')")
for mode in MODES:
    if not all((mode, v) in R1 for v in VARS):
        continue
    g = lambda v: R1[(mode, v)]
    sY = REF[f"{mode}_sigY"]
    r0s = np.array([rsz(run("S1zp", mode, "real", p), 0, mode) for p in PR])
    r0a = np.array([a01("real", p)[f"{mode}_szifi_sig0"] / sY for p in PR])
    rows = (("total: S1 - a01", g("S1") - g("a01")),
            ("artifact, bare tile: S1 - S1z", g("S1") - g("S1z")),
            ("artifact, periodic: S1p - S1zp", g("S1p") - g("S1zp")),
            ("edges, with the artifact: S1 - S1p", g("S1") - g("S1p")),
            ("edges, without it: S1z - S1zp", g("S1z") - g("S1zp")),
            ("rest: S1zp - a01", g("S1zp") - g("a01")),
            ("  of which iteration 0 (template): r0(SZiFi) - r0(a01)", r0s - r0a),
            ("  rest minus that: (S1zp - a01) - (r0 diff)", (g("S1zp") - g("a01")) - (r0s - r0a)))
    print(f"  ---- {mode}")
    for lab, d in rows:
        print(f"    {lab:58s} {ms(d[:, 0])} / {ms(d[:, 9])} / {ms(d[:, 14])}")
        SUM[f"dec_{mode}_{lab.split(':')[0].strip()}"] = [d[:, j].mean() for j in J3] + \
            [d[:, j].std(ddof=1) / np.sqrt(d.shape[0]) for j in J3]
    tot = (g("S1") - g("a01"))[:, 0].mean()
    print(f"    shares of the total at 0.5': artifact (bare) {(g('S1') - g('S1z'))[:, 0].mean() / tot:.2f}, edges (with "
          f"artifact) {(g('S1') - g('S1p'))[:, 0].mean() / tot:.2f}, edges (without) {(g('S1z') - g('S1zp'))[:, 0].mean() / tot:.2f}, "
          f"rest {(g('S1zp') - g('a01'))[:, 0].mean() / tot:.2f}")
    ec = np.array([run("S1zp", mode, "real", p)["edge_cov"] for p in PR])
    e1 = np.array([run("S1p", mode, "real", p)["edge_cov"] for p in PR])
    print(f"    edge term vs boundary coverage of the mask: correlation {np.corrcoef((g('S1z') - g('S1zp'))[:, 0], ec)[0, 1]:+.2f} "
          f"(without artifact), {np.corrcoef((g('S1') - g('S1p'))[:, 0], e1)[0, 1]:+.2f} (with); largest edge term "
          f"{(g('S1') - g('S1p'))[:, 0].max():+.4f} (tile {PR[int(np.argmax((g('S1') - g('S1p'))[:, 0]))]}, coverage "
          f"{e1[int(np.argmax((g('S1') - g('S1p'))[:, 0]))]:.3f})")

# ------------------------------------------------------------------------------------------------ (4b) S2 variants
P2 = [0, 11, 22, 33, 44, 55, 88]
print(f"\n(4b) l_sep = 12000 (S2): as configured (build), S2z, S2zp vs a01 'szifi_l12000' on tiles {P2}")
for mode in MODES:
    sY = REF[f"{mode}_sigY"]
    a1 = np.array([a01("real", p)[f"{mode}_szifi_l12000_sig1"] / sY for p in P2])
    af = np.array([a01("real", p)[f"{mode}_szifi_l12000_fsky"] for p in P2])
    pd = np.array([count(finder(mode, "real", p, 1), 5.0) for p in P2])
    print(f"  ---- {mode}: a01 r after {a1.mean(0)[0]:.4f} / {a1.mean(0)[9]:.4f} / {a1.mean(0)[14]:.4f}; f {af.mean():.4f}")
    for var in ("S2", "S2z", "S2zp"):
        R = {p: (load(os.path.join(BRUN, f"S2_{mode}_real_{p:03d}.npz")) if var == "S2" else run(var, mode, "real", p))
             for p in P2}
        if any(v is None for v in R.values()):
            print(f"    {var}: runs missing"); continue
        r1 = np.array([rsz(R[p], 1, mode) for p in P2])
        fs = np.array([R[p]["rec_fsky"][0] for p in P2])
        n1 = np.array([count(sz(R[p], 1), 5.0) for p in P2])
        rp, ep = boot_ratio(n1, pd)
        print(f"    {var:5s}: r after {r1.mean(0)[0]:.4f} / {r1.mean(0)[9]:.4f} / {r1.mean(0)[14]:.4f}; paired - a01 "
              f"{ms(r1[:, 0] - a1[:, 0])} / {ms(r1[:, 9] - a1[:, 9])} / {ms(r1[:, 14] - a1[:, 14])}; f {fs.mean():.4f}; "
              f"final q > 5 / production {rp:.4f} +- {ep:.4f}")
        SUM[f"{var}_{mode}"] = dict(r1=r1.mean(0).tolist(), d_a01=(r1 - a1).mean(0).tolist(),
                                    d_a01_sem=((r1 - a1).std(0, ddof=1) / np.sqrt(len(P2))).tolist(), f=fs.mean(),
                                    prod=rp, prod_err=ep)

# ------------------------------------------------------------------------------------------------ (5) config's S1wg
print("\n(5) THE CONFIG CHECK'S lrange [100, 30600] RUNS (S1wg) vs S1z on the same tiles (0-55)")
for mode in MODES:
    pw = [p for p in PR if run("S1wg", mode, "real", p) is not None and run("S1z", mode, "real", p) is not None]
    if not pw:
        continue
    out = []
    for var in ("S1", "S1z", "S1wg"):
        R = {p: run(var, mode, "real", p) for p in pw}
        infl = np.array([R[p]["it1_sig"][[1, 9, 14]] / R[p]["it0_sig"][[1, 9, 14]] for p in pw])
        fs = np.array([R[p]["rec_fsky"][0] for p in pw])
        bg = np.array([big(R[p]) for p in pw])
        n0 = np.array([count(sz(R[p], 0), 5.0) for p in pw]); n1 = np.array([count(sz(R[p], 1), 5.0) for p in pw])
        out.append(f"{var}: it1/it0 {infl.mean(0)[0]:.4f}/{infl.mean(0)[1]:.4f}/{infl.mean(0)[2]:.4f}, f {fs.mean():.4f}, "
                   f"theta>=7.24' {bg.mean():.1f}, q>5 it0->it1 {100 * (n1.sum() / n0.sum() - 1):+.1f}%")
    A = {p: a01("real", p) for p in pw}
    ai = np.array([A[p][f"{mode}_szifi_sig1"][[1, 9, 14]] / A[p][f"{mode}_szifi_sig0"][[1, 9, 14]] for p in pw])
    print(f"  {mode} ({len(pw)} tiles; it1/it0 at 0.64' / 4.45' / 15'): " + "; ".join(out) +
          f"; a01: it1/it0 {ai.mean(0)[0]:.4f}/{ai.mean(0)[1]:.4f}/{ai.mean(0)[2]:.4f}")

# ------------------------------------------------------------------------------------------------ (6) noise-only
print("\n(6) NOISE-ONLY TILES (5 per level): sigma / analytic after the iteration, and false detections")
for mode in MODES:
    A = {p: a01("noise", p) for p in PN}
    for var in ("S1", "S1z", "S1zp"):
        R = {p: run(var, mode, "noise", p) for p in PN}
        if any(v is None for v in R.values()):
            continue
        s1 = np.array([rsz(R[p], 1, mode) for p in PN])
        fs = np.array([R[p]["rec_fsky"][0] for p in PN])
        cnt = np.array([[count(sz(R[p], 1), qc) for qc in (4.0, 4.5, 5.0)] for p in PN]).sum(0)
        bg = np.mean([big(R[p]) for p in PN])
        print(f"  {mode:8s} {var:4s}: sigma/analytic at 0.5' / 4.45' / 15' {np.nanmean(s1, 0)[0]:.4f} / "
              f"{np.nanmean(s1, 0)[9]:.4f} / {np.nanmean(s1, 0)[14]:.4f}; f {fs.mean():.4f}; theta >= 7.24' it0 {bg:.1f} "
              f"per tile; false q > 4 / 4.5 / 5 per 100 deg^2: {100 * cnt[0] / (AREA * 5):.1f} / {100 * cnt[1] / (AREA * 5):.2f} / "
              f"{100 * cnt[2] / (AREA * 5):.2f} {cnt.tolist()}")
        SUM[f"noise_{var}_{mode}"] = dict(s1=np.nanmean(s1, 0).tolist(), f=fs.mean(), false=cnt.tolist())
    sa = np.array([A[p][f"{mode}_szifi_sig1"] / A[p][f"{mode}_sig_ana"] for p in PN])
    print(f"  {mode:8s} a01 : sigma/analytic {sa.mean(0)[0]:.4f} / {sa.mean(0)[9]:.4f} / {sa.mean(0)[14]:.4f}")
    SUM[f"noise_a01_{mode}"] = sa.mean(0).tolist()

# ------------------------------------------------------------------------------------------------ (7) compute
print("\n(7) COMPUTE of the arbiter's SZiFi runs")
files = sorted(f for f in os.listdir(ARUN) if f.endswith(".npz"))
tt = {}
for f in files:
    r = load(os.path.join(ARUN, f))
    tt.setdefault(str(r["variant"]), []).append((float(r["t_run"]), float(r["peak_rss_gb"])))
for v, x in sorted(tt.items()):
    x = np.array(x)
    print(f"  {v:5s}: {len(x):3d} runs, wall per run {x[:, 0].mean():.0f} s (max {x[:, 0].max():.0f}), process peak RSS "
          f"up to {x[:, 1].max():.2f} GB")
print(f"  total SZiFi wall time (sum over runs) {sum(sum(a for a, _ in x) for x in tt.values()) / 3600:.2f} h")
json.dump(SUM, open(os.path.join(OUT, "arbiter", "zA05_summary.json"), "w"), default=lambda o: np.asarray(o).tolist())
print(f"\ndone [{time.time() - t0:.0f} s]")
