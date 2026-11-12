"""For PROGRESS.md: independent spot-check of the real-SZiFi arbiter's new numbers (own code; reads only stored run files).

Checks, on the arbiter's 12 realistic tiles per level and 5 noise-only tiles:
  1. r after the iteration at 0.5', 4.45', 15' for S1 (build), S1z, S1p, S1zp (arbiter) and a01 (step 3a emulation);
     sigma(0.5') recovered from y0/q myself (E1), sigma_Y from step 1's table by my own spline.
  2. the decomposition of S1 - a01 at 0.5' into artifact (S1 - S1z), edges (S1z - S1zp) and rest (S1zp - a01).
  3. final q > 5 counts, SZiFi / production finder, interior >= 9' from the edges, SZiFi shifted by -1/2 pixel;
     also for the build's 30 S1 tiles (as configured).
  4. noise-only sigma/analytic after, S1 / S1z / S1zp and a01 (5 tiles).
  5. E4: SZiFi's recorded iteration-0 spectra above l = 20016 against the in-band level (tile 0 baseline, S1).
  6. a second way of removing the artifact: the config check's S1wg runs (l range [100, 30600], every mode in band)
     against the arbiter's S1z on the 6 shared tiles per level.
  7. paired q > 5 count ratios on the same tiles for step 3a's emulation (a01 'szifi' / a01 'whitened', whole tile).
  8. the per-tile edge term at 0.5', with and without the artifact.
  9. the unmasked fraction f and the iteration-0 entries at theta >= 7.24' (S1, S1z, a01, production).
 10. SZiFi's template offset before the iteration, and S1zp - a01 after it with that offset removed.
 11. l_sep = 12000: S2, S2z, S2zp against a01, the q > 5 count ratios (also a01's), and S2 against the production finder's r.
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RB = f"{ROOT}/data_real/step3_szifi_real/runs"
RA = f"{ROOT}/data_real/step3_szifi_real/arbiter/runs"
A01 = f"{ROOT}/data_real/step3/arbiter"
TIL = f"{ROOT}/data_real/step3/tiles"
ANA = np.load(f"{ROOT}/data_real/step3_szifi_real/analytic_sigma.npz")
SY = np.load(f"{ROOT}/data_real/sigmaY_mmf.npz")
TH = ANA["thetas"]
J = [0, 9, 14]
PIDS = [0, 11, 22, 33, 44, 55, 88, 132, 176, 220, 264, 308]
PIDS30 = list(range(0, 320, 11))
NOISE = [0, 22, 44, 66, 88]
NPIX, PIX = 1200, 0.5
EDGE = int(round(9.0 / PIX))  # 18 pixels
rng = np.random.default_rng(12345)


def sigY(mode):
    cs = CubicSpline(np.log(SY["theta500"]), np.log(SY[f"{mode}_l100"]))
    s = np.exp(cs(np.log(TH)))
    s[0] = SY[f"exact_{mode}_l100"][1]  # exact value at 0.5'
    return s


def sz_sigma(d, it):
    s = d[f"it{it}_sigvec"].astype(float).copy()
    th = d[f"it{it}_theta"]
    m = np.isclose(th, TH[0])
    s[0] = np.median(d[f"it{it}_y0"][m] / d[f"it{it}_q"][m])
    return s


def load(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def run(tag, mode, kind, pid):
    p = f"{RA}/{tag}_{mode}_{kind}_{pid:03d}.npz"
    if not os.path.exists(p):
        p = f"{RB}/{tag}_{mode}_{kind}_{pid:03d}.npz"
    return load(p)


def n_sz(d, it, qc=5.0):
    q = d[f"it{it}_q"]
    i = d[f"it{it}_i"] - 0.5
    j = d[f"it{it}_j"] - 0.5
    m = (q > qc) & (i >= EDGE) & (i < NPIX - EDGE) & (j >= EDGE) & (j < NPIX - EDGE)
    return int(m.sum())


def n_prod(t, mode, qc=5.0):
    q = t[f"{mode}_it1_q"]
    y = t[f"{mode}_it1_iy"]
    x = t[f"{mode}_it1_ix"]
    m = (q > qc) & (y >= EDGE) & (y < NPIX - EDGE) & (x >= EDGE) & (x < NPIX - EDGE)
    return int(m.sum())


def boot_ratio(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, a.size, (n, a.size))
    return a.sum() / b.sum(), (a[idx].sum(1) / b[idx].sum(1)).std()


def fmt(v):
    return " / ".join(f"{x:.3f}" for x in v)


for mode in ["baseline", "goal"]:
    sa = ANA[f"{mode}_S0"]
    sy = sigY(mode)
    ref = np.load(f"{ROOT}/data_real/step3_szifi_real/reference.npz")[f"{mode}_sigY"]
    print(f"===== {mode}: my sigma_Y vs reference.npz: max rel diff {np.abs(sy / ref - 1).max():.2e}")
    R = {}
    for tag in ["S1", "S1p", "S1z", "S1zp"]:
        rr, chk = [], []
        for p in PIDS:
            d = run(tag, mode, "real", p)
            s1 = sz_sigma(d, 1)
            chk.append(abs(s1[0] / d["it1_sig"][0] - 1))
            rr.append(s1 / sa)
        R[tag] = np.array(rr)
        print(f"  {tag:5s} r after at 0.5'/4.45'/15': {fmt(R[tag].mean(0)[J])}   "
              f"(my sigma(0.5') vs stored it1_sig[0]: max rel diff {max(chk):.1e})")
    ra = np.array([load(f"{A01}/a01_real_{p:03d}.npz")[f"{mode}_szifi_sig1"] / sy for p in PIDS])
    R["a01"] = ra
    print(f"  a01   r after at 0.5'/4.45'/15': {fmt(ra.mean(0)[J])}")
    n = len(PIDS)
    tot = R["S1"][:, 0] - ra[:, 0]
    art = R["S1"][:, 0] - R["S1z"][:, 0]
    edg = R["S1z"][:, 0] - R["S1zp"][:, 0]
    rest = R["S1zp"][:, 0] - ra[:, 0]
    for name, v in [("total S1-a01", tot), ("artifact S1-S1z", art), ("edges S1z-S1zp", edg),
                    ("rest S1zp-a01", rest), ("edges w/ artifact S1-S1p", R["S1"][:, 0] - R["S1p"][:, 0])]:
        print(f"  {name:26s} at 0.5': {v.mean():+.4f} +- {v.std(ddof=1) / np.sqrt(n):.4f}"
              f"   share of total {v.mean() / tot.mean() * 100:5.1f}%")
    # counts, 12 tiles
    prod = [n_prod(load(f"{TIL}/tile_{p:03d}.npz"), mode) for p in PIDS]
    for tag in ["S1", "S1z", "S1zp"]:
        nz = [n_sz(run(tag, mode, "real", p), 1) for p in PIDS]
        r_, e_ = boot_ratio(nz, prod)
        n0 = np.array([n_sz(run(tag, mode, "real", p), 0) for p in PIDS], float)
        n1 = np.array(nz, float)
        ch = n1 / n0 - 1
        print(f"  {tag:5s} final q>5 / production (12 tiles): {r_:.3f} +- {e_:.3f}   "
              f"q>5 through the iteration: {ch.mean() * 100:+.1f} +- {ch.std(ddof=1) / np.sqrt(n) * 100:.1f}%")
    # counts, the build's 30 tiles, S1 as configured
    prod30 = [n_prod(load(f"{TIL}/tile_{p:03d}.npz"), mode) for p in PIDS30]
    nz30 = [n_sz(load(f"{RB}/S1_{mode}_real_{p:03d}.npz"), 1) for p in PIDS30]
    r_, e_ = boot_ratio(nz30, prod30)
    print(f"  S1 as configured, 30 tiles: final q>5 / production {r_:.3f} +- {e_:.3f}  ({sum(nz30)} / {sum(prod30)})")
    r30 = np.array([sz_sigma(load(f"{RB}/S1_{mode}_real_{p:03d}.npz"), 1) / sa for p in PIDS30])
    print(f"  S1 as configured, 30 tiles: r after {fmt(r30.mean(0)[J])}")
    # noise-only
    for tag in ["S1", "S1z", "S1zp"]:
        rr = np.array([sz_sigma(run(tag, mode, "noise", p), 1) / sa for p in NOISE])
        print(f"  noise-only {tag:5s} sigma/analytic after 0.5'/4.45'/15': {fmt(rr.mean(0)[J])}")
    rn = np.array([load(f"{A01}/a01_noise_{p:03d}.npz")[f"{mode}_szifi_sig1"] / sy for p in NOISE])
    print(f"  noise-only a01   sigma/analytic after 0.5'/4.45'/15': {fmt(rn.mean(0)[J])}")

# E4: spectra above the band
d = load(f"{RB}/S1_baseline_real_000.npz")
ell, sp = d["spec0_ell"], d["spec0"]
auto = np.array([sp[:, c, c] for c in range(6)])  # (6, nbin)
inb = (ell > 15000) & (ell < 19900)
oob = ell > 20100
for c in [2, 3]:
    print(f"E4 channel {c}: out-of-band bins {oob.sum()}, |spec| out/in-band median ratio "
          f"{np.median(np.abs(auto[c, oob])) / np.median(auto[c, inb]):.1e} (max {np.abs(auto[c, oob]).max() / np.median(auto[c, inb]):.1e});"
          f" highest bin l = {ell.max():.0f}")

print("\n(6) second way of removing the artifact: config check S1wg vs arbiter S1z")
CW = f"{ROOT}/verify_real/step3_szifi_real_config/out/runs"
E, N = EDGE, NPIX

def sig(d, it):  # same rule as sz_sigma, kept separate for the check's run files
    s = d[f"it{it}_sigvec"].astype(float).copy()
    m = np.isclose(d[f"it{it}_theta"], TH[0])
    s[0] = np.median(d[f"it{it}_y0"][m] / d[f"it{it}_q"][m])
    return s

def nq(d, it):
    i, j, q = d[f"it{it}_i"] - 0.5, d[f"it{it}_j"] - 0.5, d[f"it{it}_q"]
    return int(((q > 5) & (i >= E) & (i < N - E) & (j >= E) & (j < N - E)).sum())

for mode in ["baseline", "goal"]:
    sa = ANA[f"{mode}_S0"]
    a, b, c, na, nb, nc, fa, fb = [], [], [], [], [], [], [], []
    for p in [0, 11, 22, 33, 44, 55]:
        w = np.load(f"{CW}/S1wg_real_{mode}_{p:03d}.npz", allow_pickle=True)
        z = np.load(f"{RA}/S1z_{mode}_real_{p:03d}.npz")
        s = np.load(f"{RB}/S1_{mode}_real_{p:03d}.npz")
        a.append(sig(w, 1) / sa); b.append(sig(z, 1) / sa); c.append(sig(s, 1) / sa)
        na.append(nq(w, 1)); nb.append(nq(z, 1)); nc.append(nq(s, 1))
        fa.append(float(w["fsky"][0])); fb.append(float(z["rec_fsky"][0]))
    a, b, c = map(np.array, (a, b, c))
    print(f"{mode}: r after 0.5'/4.45'/15'  S1wg {a.mean(0)[J].round(4)}  S1z {b.mean(0)[J].round(4)}  S1 {c.mean(0)[J].round(4)}")
    print(f"   max |S1wg - S1z| per tile at the three scales: {np.abs(a - b)[:, J].max(0).round(4)};"
          f" f: S1wg {np.mean(fa):.4f} S1z {np.mean(fb):.4f}")
    print(f"   q>5 interior, it1, summed over 6 tiles: S1wg {sum(na)}  S1z {sum(nb)}  S1 {sum(nc)}")

print("\n(7) step 3a emulation / production, q > 5, whole periodic tile")
P12 = [0, 11, 22, 33, 44, 55, 88, 132, 176, 220, 264, 308]
for mode in ["baseline", "goal"]:
    for name, pids in [("12 arbiter tiles", P12), ("30 build tiles", list(range(0, 320, 11)))]:
        e = np.array([np.load(f"{A01}/a01_real_{p:03d}.npz")[f"{mode}_szifi_n1"][2] for p in pids])
        w = np.array([np.load(f"{A01}/a01_real_{p:03d}.npz")[f"{mode}_whitened_n1"][2] for p in pids])
        t = np.array([int((np.load(f"{TIL}/tile_{p:03d}.npz")[f"{mode}_it1_q"] > 5).sum()) for p in pids])
        print(f"{mode:8s} {name}: emulation/production q>5 (whole tile) {e.sum()/w.sum():.3f};  "
              f"production rerun == stored catalogue count: {np.array_equal(w, t)}")

print("\n(8) per-tile edge term at 0.5' (r without periodic filling minus r with it), 12 tiles")
for mode in ["baseline", "goal"]:
    sa = ANA[f"{mode}_S0"][0]
    rr = {t: np.array([sz_sigma(run(t, mode, "real", p), 1)[0] / sa for p in PIDS]) for t in ["S1", "S1p", "S1z", "S1zp"]}
    wo, wi = rr["S1z"] - rr["S1zp"], rr["S1"] - rr["S1p"]
    print(f"  {mode:8s} without the artifact (S1z - S1zp): mean {wo.mean():+.4f}, largest {wo.max():+.4f} (tile {PIDS[int(wo.argmax())]});"
          f"  with it (S1 - S1p): mean {wi.mean():+.4f}, largest {wi.max():+.4f} (tile {PIDS[int(wi.argmax())]})")

TH7 = int(np.argmin(np.abs(TH - 7.237)))  # scale index of 7.24'


def n_large_sz(d):
    q, th = d["it0_q"], d["it0_theta"]
    i, j = d["it0_i"] - 0.5, d["it0_j"] - 0.5
    inside = (i >= EDGE) & (i < NPIX - EDGE) & (j >= EDGE) & (j < NPIX - EDGE)
    big = (q > 4) & (th >= TH[TH7] - 1e-9)
    return int((big & inside).sum()), int(big.sum())


def n_large_prod(t, mode):
    q, js = t[f"{mode}_it0_q"], t[f"{mode}_it0_j"]
    y, x = t[f"{mode}_it0_iy"], t[f"{mode}_it0_ix"]
    inside = (y >= EDGE) & (y < NPIX - EDGE) & (x >= EDGE) & (x < NPIX - EDGE)
    big = (q > 4) & (js >= TH7)
    return int((big & inside).sum()), int(big.sum())


print("\n(9) unmasked fraction f and iteration-0 entries at theta >= 7.24' (q > 4), 12 tiles; interior / whole tile")
for mode in ["baseline", "goal"]:
    for tag in ["S1", "S1z"]:
        ds = [run(tag, mode, "real", p) for p in PIDS]
        f = np.mean([float(d["rec_fsky"][0]) for d in ds])
        nl = np.array([n_large_sz(d) for d in ds]).mean(0)
        print(f"  {mode:8s} {tag:4s}: f {f:.3f}; entries at >= 7.24' per tile {nl[0]:.1f} (interior) / {nl[1]:.1f} (whole)")
    fa = np.mean([float(load(f"{A01}/a01_real_{p:03d}.npz")[f"{mode}_szifi_fsky"]) for p in PIDS])
    npl = np.array([n_large_prod(load(f"{TIL}/tile_{p:03d}.npz"), mode) for p in PIDS]).mean(0)
    print(f"  {mode:8s} a01 f {fa:.3f}; production finder entries at >= 7.24' per tile {npl[0]:.1f} (interior) / {npl[1]:.1f} (whole)")

print("\n(10) SZiFi's template offset before the iteration, and S1zp - a01 after it with that offset removed (12 tiles)")
for mode in ["baseline", "goal"]:
    sa, sy = ANA[f"{mode}_S0"], sigY(mode)
    r0s = np.array([sz_sigma(run("S1zp", mode, "real", p), 0) / sa for p in PIDS])
    r1s = np.array([sz_sigma(run("S1zp", mode, "real", p), 1) / sa for p in PIDS])
    r0a = np.array([load(f"{A01}/a01_real_{p:03d}.npz")[f"{mode}_szifi_sig0"] / sy for p in PIDS])
    r1a = np.array([load(f"{A01}/a01_real_{p:03d}.npz")[f"{mode}_szifi_sig1"] / sy for p in PIDS])
    off = (r0s - r0a)[:, J]
    rest = ((r1s - r1a) - (r0s - r0a))[:, J]
    n = len(PIDS)
    print(f"  {mode:8s} offset r0(SZiFi) - r0(a01) at 0.5'/4.45'/15': " + " / ".join(f"{v:+.4f}" for v in off.mean(0)) +
          ";  (S1zp - a01) - offset: " + " / ".join(f"{m:+.4f} +- {s:.4f}" for m, s in
                                                   zip(rest.mean(0), rest.std(0, ddof=1) / np.sqrt(n))))

print("\n(11) l_sep = 12000: S2 (build), S2z, S2zp against a01 'szifi_l12000', and q > 5 counts against production")
P7 = [0, 11, 22, 33, 44, 55, 88]
P10 = list(range(0, 100, 11))
for mode in ["baseline", "goal"]:
    sa, sy = ANA[f"{mode}_S0"], sigY(mode)
    ra = np.array([load(f"{A01}/a01_real_{p:03d}.npz")[f"{mode}_szifi_l12000_sig1"] / sy for p in P7])
    prod7 = [n_prod(load(f"{TIL}/tile_{p:03d}.npz"), mode) for p in P7]
    print(f"  {mode:8s} a01   r after {fmt(ra.mean(0)[J])}")
    for tag in ["S2", "S2z", "S2zp"]:
        rr = np.array([sz_sigma(run(tag, mode, "real", p), 1) / sa for p in P7])
        d_ = (rr - ra)[:, J]
        nz = [n_sz(run(tag, mode, "real", p), 1) for p in P7]
        r_, e_ = boot_ratio(nz, prod7)
        print(f"  {mode:8s} {tag:5s} r after {fmt(rr.mean(0)[J])}; paired - a01 " +
              " / ".join(f"{v:+.4f}" for v in d_.mean(0)) + f"; final q>5 / production {r_:.3f} +- {e_:.3f}")
    for name, pids in [("7 S2zp tiles", P7), ("10 S2 tiles", P10)]:
        e = sum(int(load(f"{A01}/a01_real_{p:03d}.npz")[f"{mode}_szifi_l12000_n1"][2]) for p in pids)
        w = sum(int(load(f"{A01}/a01_real_{p:03d}.npz")[f"{mode}_whitened_n1"][2]) for p in pids)
        print(f"  {mode:8s} a01 l12000 / production q>5 (whole tile), {name}: {e / w:.3f}")
    nz10 = [n_sz(load(f"{RB}/S2_{mode}_real_{p:03d}.npz"), 1) for p in P10]
    prod10 = [n_prod(load(f"{TIL}/tile_{p:03d}.npz"), mode) for p in P10]
    print(f"  {mode:8s} S2 / production q>5 (interior), 10 S2 tiles: {sum(nz10) / sum(prod10):.3f}")
    rp = np.array([load(f"{TIL}/tile_{p:03d}.npz")[f"{mode}_it1_sig"] / sy for p in P10])
    rs2 = np.array([sz_sigma(load(f"{RB}/S2_{mode}_real_{p:03d}.npz"), 1) / sa for p in P10])
    print(f"  {mode:8s} 10 S2 tiles: S2 r after {fmt(rs2.mean(0)[J])}; production finder r after {fmt(rp.mean(0)[J])}; "
          f"max |S2 - production| of the means {np.abs(rs2.mean(0) - rp.mean(0))[J].max():.3f}")
