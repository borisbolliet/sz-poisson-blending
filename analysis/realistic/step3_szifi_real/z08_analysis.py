"""z08: comparisons (a)-(d) and (g) -- SZiFi itself (runs/CFG_MODE_KIND_PID.npz) against step 3a's claims, the
production finder's stored outputs and the step-3a emulations, on the same tiles.

Conventions
  r(theta) = sigma / sigma_analytic, each filter with its own template (as the step-3a SZiFi check defined it):
    SZiFi: sigma from its run (sigma_vec; at 0.5' from its catalogue, z_szlib) / sigma of SZiFi's own filter with the
           analytic CMB + noise covariance (z04, setups 'S0' for S0-S2 and 'S3' for S3, S3i);
    production finder and the arbiter's a01 emulation: sigma / sigma_Y (s3lib.sigma_Y, step 1's MMF), as in step 3a;
    SZiFi check's emulation (verify_real/step3_szifi/out/v04_*): sig / sig_ana (its own template).
  Positions: finder (iy, ix); SZiFi reports (i, j) and sits at (i - 1/2, j - 1/2) (z07).  Interior: objects at least
  m = b + 1' from the tile edges (b = the edge buffer of SZiFi's peak-finding and selection masks, z07).
  Matching: an A object (q > qc, interior) is matched if a B object (q > 4, at least m - 1' from the edges) lies within
  1' (2 pixels), at any scale (step 3a's definition) or, where stated, also within one scale step; dq = q_B - q_A.
  Counts per tile inside the interior; rates per 100 deg^2 of interior.  Errors: tile bootstrap (2000 resamples).
Arbiter rerun (2026-11-12): interior made equal-area for both position grids (inner()); summary written to
data_real/step3_szifi_real/arbiter/z08_summary.json.  The build's original output is z08_analysis_build_original.log.
Usage: python -I -B z08_analysis.py
"""
import os, sys, time, json
sys.dont_write_bytecode = True
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(ROOT, "data_real", "step3_szifi_real")
RUND = os.path.join(OUT, "runs")
import numpy as np
from scipy.spatial import cKDTree

t0 = time.time()
rng = np.random.default_rng(20261112)
N, PIX = 1200, 0.5
MODES = ("baseline", "goal")
REF = dict(np.load(os.path.join(OUT, "reference.npz")))
ANA = dict(np.load(os.path.join(OUT, "analytic_sigma.npz")))
EB = dict(np.load(os.path.join(OUT, "edge_buffer.npz")))
B_BUF, M_INT = float(EB["buffer_arcmin"]), float(EB["interior_arcmin"])
SHIFT = 0.5
TH = REF["thetas"]
PR = [int(p) for p in REF["pids_real"]]
PN = [int(p) for p in REF["pids_noise"]]
V04 = os.path.join(ROOT, "verify_real", "step3_szifi", "out")
A01 = os.path.join(ROOT, "data_real", "step3", "arbiter")
JJ = [0, 3, 6, 9, 12, 14]                       # 0.50' 1.04' 2.15' 4.45' 9.23' 15'
AREA = ((N * PIX - 2 * M_INT) / 60.0) ** 2      # deg^2 of interior per tile
SETUP = {"S0": "S0", "S1": "S0", "S2": "S0", "S3": "S3", "S3i": "S3", "S3it0": "S3"}
NB = 2000


def fmt(v, f="{:7.4f}"):
    return " ".join(f.format(x) for x in v)


def load(path):
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def run(cfg, mode, kind, pid):
    return load(os.path.join(RUND, f"{cfg}_{mode}_{kind}_{pid:03d}.npz"))


_TILE = {}


def tile(pid):
    if pid not in _TILE:
        _TILE[pid] = load(os.path.join(ROOT, "data_real", "step3", "tiles", f"tile_{pid:03d}.npz"))
    return _TILE[pid]


FN = load(os.path.join(OUT, "finder_noise.npz"))


def finder(mode, kind, pid, it):
    if kind == "real":
        t = tile(pid)
        g = lambda k: t[f"{mode}_it{it}_{k}"]
    else:
        g = lambda k: FN[f"{mode}_{pid:03d}_it{it}_{k}"]
    return dict(y=g("iy") * 1.0, x=g("ix") * 1.0, q=g("q"), j=g("j"), sig=g("sig"))


def sz(r, it):
    return dict(y=r[f"it{it}_i"] - SHIFT, x=r[f"it{it}_j"] - SHIFT, q=r[f"it{it}_q"], j=r[f"it{it}_j_theta"],
                sig=r[f"it{it}_sig"])


def emul(r, it):
    """step-3a emulation (v04 / tiles_szifi): v04 reports SZiFi's pixel convention, tiles_szifi the finder's."""
    return dict(y=r[f"det{it}_n0"] - SHIFT, x=r[f"det{it}_n1"] - SHIFT, q=r[f"det{it}_q"], j=r[f"det{it}_j"])


def dist_edge(c):
    return np.minimum(np.minimum(c["y"], N - 1 - c["y"]), np.minimum(c["x"], N - 1 - c["x"])) * PIX


CATK = ("y", "x", "q", "j")


def inner(c, m=M_INT):
    """Objects in the interior: positions p (pixel units, both axes) with lo <= p < hi, lo = m/PIX - 1/4,
    hi = N - 1 - m/PIX + 3/4.  The half-open window holds 1164 positions per axis (m = 9') on the finder's integer grid
    and on SZiFi's half-integer grid alike (arbiter fix: the build's rule dist_edge >= m kept 1164 finder positions but
    only 1163 SZiFi positions per axis, which biased SZiFi / finder count ratios low by about 0.17%)."""
    lo, hi = m / PIX - 0.25, N - 1 - m / PIX + 0.75
    s = (c["y"] >= lo) & (c["y"] < hi) & (c["x"] >= lo) & (c["x"] < hi)
    return {k: (v[s] if k in CATK else v) for k, v in c.items()}


def match(A, B, qc, rad=2.0, dj=1):
    """A objects (q > qc, interior) -> nearest B (q > 4, >= m - 1' from the edges) within rad px and |dj| <= dj."""
    a = inner(A); a = {k: (v[a["q"] > qc] if k in CATK else v) for k, v in a.items()}
    b = inner(B, M_INT - 1.0); b = {k: (v[b["q"] > 4] if k in CATK else v) for k, v in b.items()}
    n = a["q"].size
    k = np.full(n, -1)
    if n and b["q"].size:
        tr = cKDTree(np.c_[b["y"], b["x"]])
        for i_, c in enumerate(tr.query_ball_point(np.c_[a["y"], a["x"]], rad + 1e-9)):
            if not c:
                continue
            c = np.asarray(c)
            if dj is not None:
                c = c[np.abs(b["j"][c] - a["j"][i_]) <= dj]
            if c.size:
                d = np.hypot(b["y"][c] - a["y"][i_], b["x"][c] - a["x"][i_])
                k[i_] = c[np.argmin(d)]
    ok = k >= 0
    return dict(n=n, nm=int(ok.sum()), dq=b["q"][k[ok]] - a["q"][ok], dj=b["j"][k[ok]] - a["j"][ok],
                dy=b["y"][k[ok]] - a["y"][ok], dx=b["x"][k[ok]] - a["x"][ok])


def pool(ms):
    n = sum(m["n"] for m in ms); nm = sum(m["nm"] for m in ms)
    dq = np.concatenate([m["dq"] for m in ms]) if ms else np.zeros(0)
    dj = np.concatenate([m["dj"] for m in ms]) if ms else np.zeros(0)
    return n, nm, dq, dj


def match_line(lab, ms):
    n, nm, dq, dj = pool(ms)
    if n == 0:
        return f"    {lab:58s}: N 0"
    return (f"    {lab:58s}: N {n:5d}, matched {100 * nm / n:6.2f}%; dq {dq.mean():+.3f} +- {dq.std():.3f}; same scale "
            f"{100 * np.mean(dj == 0):5.1f}%")


def count(c, qc):
    return int(np.sum(inner(c)["q"] > qc))


def boot_ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    b = rng.integers(0, num.size, (NB, num.size))
    return num.sum() / den.sum(), (num[b].sum(1) / den[b].sum(1)).std()


def r_szifi(r, it, cfg, mode):
    return r[f"it{it}_sig"] / ANA[f"{mode}_{SETUP[cfg]}"]


print(f"z08: SZiFi itself vs step 3a.  Edge buffer b = {B_BUF}', analysis interior >= {M_INT}' from the edges "
      f"({AREA:.2f} deg^2 per tile); SZiFi positions shifted by -{SHIFT} pixel (z07)")
avail = {}
for cfg in ("S0", "S1", "S2", "S3i", "S3it0", "S3"):
    for kind in ("real", "noise"):
        for mode in MODES:
            avail[(cfg, kind, mode)] = [p for p in (PR if kind == "real" else PN) if os.path.exists(
                os.path.join(RUND, f"{cfg}_{mode}_{kind}_{p:03d}.npz"))]
print("runs available (tiles per noise level, baseline / goal):")
for cfg in ("S0", "S1", "S2", "S3i", "S3it0", "S3"):
    print(f"  {cfg:6s} real {len(avail[(cfg, 'real', 'baseline')]):2d} / {len(avail[(cfg, 'real', 'goal')]):2d}, "
          f"noise-only {len(avail[(cfg, 'noise', 'baseline')]):2d} / {len(avail[(cfg, 'noise', 'goal')]):2d}")
SUM = {}

# =============================================================================================== (a) iteration 0
print("\n" + "=" * 110 + "\n(a) ITERATION 0: SZiFi itself vs the production finder (stored) and the step-3a emulation")
for mode in MODES:
    pids = avail[("S1", "real", mode)]
    if not pids:
        continue
    sY = REF[f"{mode}_sigY"]
    R = {p: run("S1", mode, "real", p) for p in pids}
    rs = np.array([r_szifi(R[p], 0, "S1", mode) for p in pids])
    rf = np.array([tile(p)[f"{mode}_it0_sig"] / sY for p in pids])
    raw = np.array([R[p]["it0_sig"] / tile(p)[f"{mode}_it0_sig"] for p in pids])
    print(f"\n  {mode}: {len(pids)} realistic tiles (S1's iteration 0 = S0's to 2e-8, z02)")
    print(f"    theta [']                                   {fmt(TH, '{:7.2f}')}")
    print(f"    r SZiFi it0 (own template, own analytic)   {fmt(rs.mean(0))}")
    print(f"       tile sd                                  {fmt(rs.std(0))}")
    print(f"    r finder it0 (stored sigma / sigma_Y)       {fmt(rf.mean(0))}")
    print(f"    r_SZiFi - r_finder, mean                    {fmt((rs - rf).mean(0), '{:+7.4f}')}")
    print(f"       tile sd                                  {fmt((rs - rf).std(0))}")
    print(f"    (r_SZiFi / r_finder - 1), mean              {fmt((rs / rf - 1).mean(0), '{:+7.4f}')}")
    rr = raw / raw.mean(0)
    print(f"    sigma_SZiFi / sigma_finder [y0 per arcmin^2 of Y500]: mean {fmt(raw.mean(0), '{:7.2f}')}")
    print(f"       relative tile sd                         {fmt(rr.std(0), '{:7.4f}')}")
    SUM[f"a_dr_{mode}"] = (rs - rf).mean(0)
    SUM[f"a_rs0_{mode}"] = rs.mean(0)
    for qc in (4.5, 5.0):
        for dj, lab in ((None, "1', any scale (step 3a)"), (1, "1' and one scale step")):
            m1 = [match(finder(mode, "real", p, 0), sz(R[p], 0), qc, dj=dj) for p in pids]
            m2 = [match(sz(R[p], 0), finder(mode, "real", p, 0), qc, dj=dj) for p in pids]
            print(match_line(f"q > {qc}, {lab}: finder it0 -> SZiFi it0", m1))
            print(match_line(f"q > {qc}, {lab}: SZiFi it0 -> finder it0", m2))
            n1, nm1, dq1, dj1 = pool(m1); n2, nm2, dq2, _ = pool(m2)
            SUM[f"a_match_{mode}_{qc}_{dj}"] = (100 * nm1 / n1, 100 * nm2 / n2, dq1.mean(), dq1.std(), n1, n2,
                                                100 * np.mean(np.abs(dj1) <= 1))
        m1 = [match(finder(mode, "real", p, 0), sz(R[p], 0), qc, dj=None) for p in pids]
        dy = np.concatenate([m["dy"] for m in m1]); dx = np.concatenate([m["dx"] for m in m1])
        print(f"    q > {qc}: matched SZiFi position (corrected by -1/2 pixel) minus finder position: mean "
              f"({dy.mean():+.3f}, {dx.mean():+.3f}) pixels")
    # unmatched SZiFi objects at q > 5: how far are they from a finder detection?
    # emulation (SZiFi check, v04_lin: SZiFi's linear convolution, own template, no buffer), same tiles
    vp = [p for p in pids if os.path.exists(os.path.join(V04, f"v04_lin_{mode}_tile{p:03d}.npz"))]
    if vp:
        E = {p: load(os.path.join(V04, f"v04_lin_{mode}_tile{p:03d}.npz")) for p in vp}
        se = np.array([R[p]["it0_sig"] / E[p]["sig0"] - 1 for p in vp])
        sa = ANA[f"{mode}_S0"] / np.array([E[p]["sig_ana"] for p in vp]).mean(0) - 1
        print(f"    vs the SZiFi check's emulation (v04_lin, {len(vp)} tiles): sigma_SZiFi / sigma_emul - 1, max |.| over "
              f"tiles and scales {np.abs(se).max():.1e}; analytic sigma (own templates) / emulation's - 1: "
              f"{fmt(sa[JJ], '{:+.1e}')} at {fmt(TH[JJ], '{:.2f}')}'")
        for qc in (4.5, 5.0):
            m1 = [match(emul(E[p], 0), sz(R[p], 0), qc, dj=None) for p in vp]
            m2 = [match(sz(R[p], 0), emul(E[p], 0), qc, dj=None) for p in vp]
            print(match_line(f"q > {qc}, 1' any scale: emulation it0 -> SZiFi it0", m1))
            print(match_line(f"q > {qc}, 1' any scale: SZiFi it0 -> emulation it0", m2))
        # exact identity in the interior: same pixel, same scale, |dq|
        nid = ntot = 0; dqm = 0.0
        for p in vp:
            a = inner(sz(R[p], 0)); e = inner(emul(E[p], 0))
            ka = set(zip(np.rint(a["y"] + SHIFT).astype(int), np.rint(a["x"] + SHIFT).astype(int), a["j"]))
            ke = {(int(round(y + SHIFT)), int(round(x + SHIFT)), int(j)): q for y, x, j, q in zip(e["y"], e["x"], e["j"], e["q"])}
            ntot += len(ka | set(ke)); nid += len(ka & set(ke))
            qa = {(int(round(y + SHIFT)), int(round(x + SHIFT)), int(j)): q for y, x, j, q in zip(a["y"], a["x"], a["j"], a["q"])}
            for k_ in ka & set(ke):
                dqm = max(dqm, abs(qa[k_] - ke[k_]))
        print(f"    interior catalogues SZiFi it0 vs emulation it0 (q > 4): {nid} of {ntot} (pixel, scale) entries in both; "
              f"max |dq| over the common entries {dqm:.1e}")
        SUM[f"a_emul_{mode}"] = (np.abs(se).max(), nid, ntot, dqm)

# S0 runs: claim 2 (no inpainting -> iteration 1 = iteration 0) and the whole-tile comparison with v04_lin
print("\n  S0 (no edge buffer): iteration 1 against iteration 0, and the whole tile against the emulation")
for mode in MODES:
    pids = avail[("S0", "real", mode)]
    same = 0
    for p in pids:
        r = run("S0", mode, "real", p)
        ok = all(np.array_equal(r[f"it0_{k}"], r[f"it1_{k}"]) for k in ("q", "i", "j", "j_theta", "sigvec")) \
            and np.array_equal(r["spec0"], r["spec1"])
        same += ok
    print(f"    {mode}: iteration 1 identical to iteration 0 (spectra, sigma, catalogue, bit for bit) in {same} of {len(pids)} tiles")
    SUM[f"claim2_{mode}"] = (same, len(pids))
    vp = [p for p in pids if os.path.exists(os.path.join(V04, f"v04_lin_{mode}_tile{p:03d}.npz"))]
    nid = ntot = 0; n_s = n_e = 0
    for p in vp:
        r = run("S0", mode, "real", p); e = load(os.path.join(V04, f"v04_lin_{mode}_tile{p:03d}.npz"))
        ka = set(zip(r["it0_i"].tolist(), r["it0_j"].tolist(), r["it0_j_theta"].tolist()))
        ke = set(zip(e["det0_n0"].tolist(), e["det0_n1"].tolist(), e["det0_j"].tolist()))
        nid += len(ka & ke); ntot += len(ka | ke); n_s += len(ka); n_e += len(ke)
    if vp:
        print(f"    {mode}: whole tile (edges included), SZiFi it0 {n_s} detections vs emulation {n_e}: {nid} (pixel, scale) "
              f"entries in both, {ntot - nid} in only one")

# =============================================================================================== (b) iteration 1
print("\n" + "=" * 110 + "\n(b) ITERATION 1 (final catalogue), per configuration")
CL = {"baseline": {"S1": (1.135, 1.119, 1.050), "S2": (1.019, 1.044, 1.026), "prod": (1.039, 1.050, 1.024)},
      "goal": {"S1": (1.181, 1.172, 1.078), "S2": (1.038, 1.075, 1.046), "prod": (1.053, 1.067, 1.035)}}
A01V = {"S1": "szifi", "S2": "szifi_l12000"}
LR5 = ((100, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 20000))
for mode in MODES:
    sY = REF[f"{mode}_sigY"]
    print(f"\n  ---- {mode}")
    print(f"    theta [']                                   {fmt(TH, '{:7.2f}')}")
    for key in ("r_i", "r_ii", "r_iii"):
        print(f"    step 2 {key:4s}                                {fmt(REF[f'{mode}_{key}'])}")
    pr = PR
    rp1 = np.array([tile(p)[f"{mode}_it1_sig"] / sY for p in pr])
    print(f"    production finder it1 (stored, 30 tiles)    {fmt(rp1.mean(0))}")
    for cfg in ("S1", "S2", "S3i"):
        pids = avail[(cfg, "real", mode)]
        if not pids:
            continue
        R = {p: run(cfg, mode, "real", p) for p in pids}
        r0 = np.array([r_szifi(R[p], 0, cfg, mode) for p in pids])
        r1 = np.array([r_szifi(R[p], 1, cfg, mode) for p in pids])
        print(f"    {cfg} it0 ({len(pids)} tiles)                       {fmt(r0.mean(0))}")
        print(f"    {cfg} it1                                   {fmt(r1.mean(0))}")
        print(f"       tile sd                                  {fmt(r1.std(0))}")
        SUM[f"b_r1_{cfg}_{mode}"] = (r1.mean(0), r1.std(0), len(pids))
        SUM[f"b_r0_{cfg}_{mode}"] = r0.mean(0)
        if cfg in CL[mode]:
            c = CL[mode][cfg]
            print(f"       claim (step 3a) at 0.5' / 4.45' / 15': {c}; SZiFi: {r1.mean(0)[0]:.3f} / {r1.mean(0)[9]:.3f} / "
                  f"{r1.mean(0)[14]:.3f} (+- {r1.std(0)[0] / np.sqrt(len(pids)):.3f} / {r1.std(0)[9] / np.sqrt(len(pids)):.3f} / "
                  f"{r1.std(0)[14] / np.sqrt(len(pids)):.3f})")
        if cfg in A01V:
            v = A01V[cfg]
            ar = {p: load(os.path.join(A01, f"a01_real_{p:03d}.npz")) for p in pids}
            a0 = np.array([ar[p][f"{mode}_{v}_sig0"] / sY for p in pids])
            a1 = np.array([ar[p][f"{mode}_{v}_sig1"] / sY for p in pids])
            print(f"    a01 emulation '{v}' it1, same tiles      {fmt(a1.mean(0))}")
            print(f"    SZiFi it1 - a01 it1 (paired)                {fmt((r1 - a1).mean(0), '{:+7.4f}')}")
            print(f"       tile sd of the difference                {fmt((r1 - a1).std(0))}")
            print(f"    SZiFi (it1/it0) / a01 (it1/it0) - 1         {fmt(((r1 / r0) / (a1 / a0) - 1).mean(0), '{:+7.4f}')}")
            SUM[f"b_a01_{cfg}_{mode}"] = a1.mean(0)
            SUM[f"b_dra01_{cfg}_{mode}"] = (r1 - a1).mean(0)
            # SZiFi check's emulation, own template, periodic and linear convolution (10 tiles)
            for conv in ("per", "lin"):
                if cfg == "S2":
                    fn_ = lambda p: os.path.join(V04.replace("/out", "/out"), f"v04_{conv}_lsep12000_{mode}_tile{p:03d}.npz")
                else:
                    fn_ = lambda p: os.path.join(V04, f"v04_{conv}_{mode}_tile{p:03d}.npz")
                vp = [p for p in pids if os.path.exists(fn_(p))]
                if vp:
                    E = {p: load(fn_(p)) for p in vp}
                    re1 = np.array([E[p]["sig1"] / E[p]["sig_ana"] for p in vp])
                    rsz = np.array([r_szifi(R[p], 1, cfg, mode) for p in vp])
                    print(f"    SZiFi check emulation v04_{conv} ({len(vp)} tiles) it1 {fmt(re1.mean(0))}")
                    print(f"       SZiFi it1 - emulation it1 (paired)       {fmt((rsz - re1).mean(0), '{:+7.4f}')}")
                    SUM[f"b_v04_{conv}_{cfg}_{mode}"] = ((rsz - re1).mean(0), len(vp))
        # spectra
        e0 = R[pids[0]]["spec0_ell"]
        dd = np.arange(6)

        def sratio(lo, hi, r):
            b = (e0 >= lo) & (e0 < hi)
            return np.mean([np.mean(r["spec1"][b, i, i] / r["spec0"][b, i, i]) for i in range(6)])
        s2 = np.array([[sratio(lo, hi, R[p]) for lo, hi in ((100, 3000), (3000, 20000))] for p in pids])
        s5 = np.array([[sratio(lo, hi, R[p]) for lo, hi in LR5] for p in pids])
        print(f"    auto spectra it1/it0 (mean of 6 channels and the bins), l 100-3000 / 3000-20000: "
              f"{s2.mean(0)[0]:.4f} / {s2.mean(0)[1]:.4f} (tile range {s2.min(0)[0]:.3f}-{s2.max(0)[0]:.3f} / "
              f"{s2.min(0)[1]:.3f}-{s2.max(0)[1]:.3f})")
        print(f"       in a01's five ranges {LR5}: {fmt(s5.mean(0))}")
        SUM[f"b_spec_{cfg}_{mode}"] = s2.mean(0)
        if cfg in A01V:
            sa = np.array([load(os.path.join(A01, f"a01_real_{p:03d}.npz"))[f"{mode}_{A01V[cfg]}_spec"] for p in pids])
            print(f"       a01 '{A01V[cfg]}', same tiles:            {fmt(sa.mean(0))}")
        fs = np.array([R[p]["rec_fsky"][0] for p in pids])
        print(f"    cluster mask: unmasked fraction {fs.mean():.3f} (range {fs.min():.3f}-{fs.max():.3f}), masked detections "
              f"per tile {np.mean([R[p]['rec_n_masked'][0] for p in pids]):.1f}")
        nb_s = np.array([np.sum((R[p]["it0_q"] > 4) & (R[p]["it0_j_theta"] >= 11)) for p in pids])
        nb_f = np.array([np.sum((tile(p)[f"{mode}_it0_q"] > 4) & (tile(p)[f"{mode}_it0_j"] >= 11)) for p in pids])
        print(f"       q > 4 detections with theta >= 7.2' per tile (they mask discs of radius >= 21.7'): SZiFi it0 "
              f"{nb_s.mean():.1f}, production finder it0 {nb_f.mean():.1f}")
        SUM[f"b_f_{cfg}_{mode}"] = (fs.mean(), nb_s.mean(), nb_f.mean())
        if cfg in A01V:
            af = np.array([load(os.path.join(A01, f"a01_real_{p:03d}.npz"))[f"{mode}_{A01V[cfg]}_fsky"] for p in pids])
            print(f"       a01 '{A01V[cfg]}' unmasked fraction, same tiles: {af.mean():.3f}; paired SZiFi - a01: "
                  f"{(fs - af).mean():+.3f} +- {(fs - af).std():.3f}; correlation of (r1 - a1) at 0.5' with (f - f_a01): "
                  f"{np.corrcoef((r1 - a1)[:, 0], fs - af)[0, 1] if len(pids) > 2 else np.nan:+.2f}")
            SUM[f"b_fa01_{cfg}_{mode}"] = af.mean()
        # counts before -> after
        n0 = np.array([count(sz(R[p], 0), 5.0) for p in pids]); n1 = np.array([count(sz(R[p], 1), 5.0) for p in pids])
        rat, err = boot_ratio(n1, n0)
        print(f"    q > 5 detections per tile in the interior, it0 -> it1: {n0.mean():.1f} -> {n1.mean():.1f} "
              f"({100 * (rat - 1):+.1f} +- {100 * err:.1f}%); per 100 deg^2 {100 * n0.mean() / AREA:.1f} -> {100 * n1.mean() / AREA:.1f}")
        SUM[f"b_counts_{cfg}_{mode}"] = (n0.mean(), n1.mean(), rat - 1, err, 100 * n0.mean() / AREA, 100 * n1.mean() / AREA)
        if cfg in A01V:
            an0 = np.array([load(os.path.join(A01, f"a01_real_{p:03d}.npz"))[f"{mode}_{A01V[cfg]}_n0"][2] for p in pids])
            an1 = np.array([load(os.path.join(A01, f"a01_real_{p:03d}.npz"))[f"{mode}_{A01V[cfg]}_n1"][2] for p in pids])
            ar_, ae_ = boot_ratio(an1, an0)
            print(f"       a01 '{A01V[cfg]}' (whole periodic tile, 100 deg^2): {an0.mean():.1f} -> {an1.mean():.1f} "
                  f"({100 * (ar_ - 1):+.1f} +- {100 * ae_:.1f}%)")
            SUM[f"b_a01counts_{cfg}_{mode}"] = (an0.mean(), an1.mean(), ar_ - 1)
        # S1 vs the emulated SZiFi-default catalogue (tiles_szifi, t04 with masked_cov 'szifi'), tiles 0-99
        if cfg == "S1":
            tp = [p for p in pids if p < 100]
            if tp:
                def tsz(p, it):
                    t = load(os.path.join(ROOT, "data_real", "step3", "tiles_szifi", f"tile_{p:03d}.npz"))
                    return dict(y=t[f"{mode}_it{it}_iy"] * 1.0, x=t[f"{mode}_it{it}_ix"] * 1.0, q=t[f"{mode}_it{it}_q"],
                                j=t[f"{mode}_it{it}_j"])
                for qc in (4.5, 5.0):
                    m1 = [match(tsz(p, 1), sz(R[p], 1), qc, dj=None) for p in tp]
                    m2 = [match(sz(R[p], 1), tsz(p, 1), qc, dj=None) for p in tp]
                    print(match_line(f"q > {qc}, 1' any scale: tiles_szifi it1 -> S1 it1 ({len(tp)} tiles)", m1))
                    print(match_line(f"q > {qc}, 1' any scale: S1 it1 -> tiles_szifi it1", m2))
                    n1_, nm1, dq1, _ = pool(m1); n2_, nm2, dq2, _ = pool(m2)
                    SUM[f"b_tsz_{mode}_{qc}"] = (100 * nm1 / n1_, 100 * nm2 / n2_, -dq1.mean(), dq1.std())
    # S3 (SZiFi's SO example as configured): iteration 0 only, and the one full smoke run
    pids = avail[("S3it0", "real", mode)]
    if pids:
        n = np.array([[count(sz(run("S3it0", mode, "real", p), 0), qc) for qc in (4.0, 5.0)] for p in pids])
        med = np.median(np.concatenate([inner(sz(run("S3it0", mode, "real", p), 0))["q"] for p in pids]))
        print(f"    S3 as configured (mixed_gaussian), iteration 0, {len(pids)} tiles: q > 4 / 5 detections per tile in the "
              f"interior {n[:, 0].mean():.0f} / {n[:, 1].mean():.0f}; median q of the detections {med:.1f}")
        SUM[f"b_S3it0_{mode}"] = (n[:, 0].mean(), n[:, 1].mean(), med, len(pids))

# =============================================================================================== (c) noise-only tiles
print("\n" + "=" * 110 + "\n(c) NOISE-ONLY TILES: sigma / analytic, and false detections per 100 deg^2 of interior")
for mode in MODES:
    print(f"\n  ---- {mode}; scales {fmt(TH[JJ], '{:.2f}')}'")
    for cfg in ("S1", "S2", "S3i"):
        pids = avail[(cfg, "noise", mode)]
        if not pids:
            continue
        R = {p: run(cfg, mode, "noise", p) for p in pids}
        a0 = np.array([r_szifi(R[p], 0, cfg, mode) for p in pids]); a1 = np.array([r_szifi(R[p], 1, cfg, mode) for p in pids])
        print(f"    {cfg:4s} ({len(pids)} tiles) sigma/analytic it0: {fmt(np.nanmean(a0, 0)[JJ])};  it1: {fmt(np.nanmean(a1, 0)[JJ])}  "
              f"(tile sd it1 {fmt(np.nanstd(a1, 0)[JJ])}; sigma(0.5') from the catalogue in {np.isfinite(a1[:, 0]).sum()} tiles)")
        SUM[f"c_s1_{cfg}_{mode}"] = (np.nanmean(a1, 0), np.nanstd(a1, 0), len(pids))
        SUM[f"c_s0_{cfg}_{mode}"] = np.nanmean(a0, 0)
        cnt = np.array([[count(sz(R[p], 1), qc) for qc in (4.0, 4.5, 5.0)] for p in pids])
        cnt0 = np.array([[count(sz(R[p], 0), qc) for qc in (4.0, 4.5, 5.0)] for p in pids])
        area = AREA * len(pids)
        print(f"         false detections per 100 deg^2 q > 4 / 4.5 / 5: it1 {100 * cnt.sum(0)[0] / area:.1f} / "
              f"{100 * cnt.sum(0)[1] / area:.2f} / {100 * cnt.sum(0)[2] / area:.2f} {cnt.sum(0).tolist()}; it0 "
              f"{100 * cnt0.sum(0)[0] / area:.1f} / {100 * cnt0.sum(0)[1] / area:.2f} / {100 * cnt0.sum(0)[2] / area:.2f}; "
              f"interior {area:.0f} deg^2")
        SUM[f"c_false_{cfg}_{mode}"] = (100 * cnt.sum(0) / area, cnt.sum(0), area)
        fs = np.array([R[p]["rec_fsky"][0] for p in pids])
        print(f"         unmasked fraction {fs.mean():.3f}")
    pids3 = avail[("S3it0", "noise", mode)]
    if pids3:
        n3 = np.array([[count(sz(run("S3it0", mode, "noise", p), 0), qc) for qc in (4.0, 5.0)] for p in pids3])
        med = np.median(np.concatenate([inner(sz(run("S3it0", mode, "noise", p), 0))["q"] for p in pids3]))
        print(f"    S3 as configured (mixed_gaussian), iteration 0, {len(pids3)} noise-only tiles: q > 4 / 5 detections per tile "
              f"in the interior {n3[:, 0].mean():.0f} / {n3[:, 1].mean():.0f}; median q {med:.1f}")
        SUM[f"c_S3it0_{mode}"] = (n3[:, 0].mean(), n3[:, 1].mean(), med, len(pids3))
    pids = avail[("S1", "noise", mode)] or PN
    fcnt = np.array([[count(finder(mode, "noise", p, 1), qc) for qc in (4.0, 4.5, 5.0)] for p in pids])
    fs1 = np.array([finder(mode, "noise", p, 1)["sig"] for p in pids])
    na = [load(os.path.join(A01, f"a01_noise_{p:03d}.npz")) for p in pids]
    sa = np.array([n_[f"{mode}_sig_ana"] for n_ in na])
    print(f"    production finder, same {len(pids)} tiles and interior: sigma/analytic it1 {fmt((fs1 / sa).mean(0)[JJ])}; false "
          f"q > 4 / 4.5 / 5: {100 * fcnt.sum(0)[0] / (AREA * len(pids)):.1f} / {100 * fcnt.sum(0)[1] / (AREA * len(pids)):.2f} / "
          f"{100 * fcnt.sum(0)[2] / (AREA * len(pids)):.2f} {fcnt.sum(0).tolist()}")
    for v in ("szifi", "szifi_l12000"):
        s1 = np.array([n_[f"{mode}_{v}_sig1"] / n_[f"{mode}_sig_ana"] for n_ in na])
        nf = np.sum([n_[f"{mode}_{v}_n1"] for n_ in na], 0)
        print(f"    a01 '{v}' on the same tiles (whole tile): sigma/analytic it1 {fmt(s1.mean(0)[JJ])}; false "
              f"{100 * nf[0] / (100 * len(pids)):.1f} / {100 * nf[1] / (100 * len(pids)):.2f} / {100 * nf[2] / (100 * len(pids)):.2f}")
        SUM[f"c_a01_{v}_{mode}"] = s1.mean(0)

# =============================================================================================== (d) paired counts
print("\n" + "=" * 110 + "\n(d) PAIRED COUNTS: N(q > 5, final catalogue, interior) of each configuration / the production finder's")
for mode in MODES:
    for cfg in ("S1", "S2", "S3i"):
        pids = avail[(cfg, "real", mode)]
        if not pids:
            continue
        nc = np.array([count(sz(run(cfg, mode, "real", p), 1), 5.0) for p in pids])
        npd = np.array([count(finder(mode, "real", p, 1), 5.0) for p in pids])
        rat, err = boot_ratio(nc, npd)
        print(f"  {mode:8s} {cfg:4s} / production ({len(pids)} tiles): {rat:.4f} +- {err:.4f}  ({nc.sum()} / {npd.sum()})")
        SUM[f"d_{cfg}_{mode}"] = (rat, err, len(pids))
    for a, b in (("S3i", "S2"), ("S2", "S1")):
        pids = sorted(set(avail[(a, "real", mode)]) & set(avail[(b, "real", mode)]))
        if not pids:
            continue
        na_ = np.array([count(sz(run(a, mode, "real", p), 1), 5.0) for p in pids])
        nb_ = np.array([count(sz(run(b, mode, "real", p), 1), 5.0) for p in pids])
        rat, err = boot_ratio(na_, nb_)
        print(f"  {mode:8s} {a} / {b} ({len(pids)} tiles): {rat:.4f} +- {err:.4f}")
        SUM[f"d_{a}_{b}_{mode}"] = (rat, err, len(pids))

# =============================================================================================== (g) compute
print("\n" + "=" * 110 + "\n(g) COMPUTE per tile run (SZiFi itself, two noise iterations unless stated)")
for cfg in ("S0", "S1", "S2", "S3i", "S3it0"):
    for kind in ("real", "noise"):
        rs_ = [run(cfg, m, kind, p) for m in MODES for p in avail[(cfg, kind, m)]]
        if not rs_:
            continue
        t = np.array([r["t_run"] for r in rs_]); m = np.array([r["peak_rss_gb"] for r in rs_])
        tf = np.array([np.sum(r["rec_t_filter"]) for r in rs_])
        print(f"  {cfg:5s} {kind:5s}: {len(rs_):3d} runs; wall per run {t.mean():6.1f} s (min {t.min():.0f}, max {t.max():.0f}); "
              f"of which filtering + peak finding {tf.mean():.1f} s; process peak RSS up to {m.max():.2f} GB")
json.dump({k: (np.asarray(v).tolist() if not isinstance(v, tuple) else [np.asarray(x).tolist() for x in v]) for k, v in SUM.items()},
          open(os.path.join(OUT, "arbiter", "z08_summary.json"), "w"))   # arbiter rerun: new file; the build's is kept
print(f"\ndone [{time.time() - t0:.0f} s]")
