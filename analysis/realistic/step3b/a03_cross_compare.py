"""Step 3b arbitration / a03: (1) the build's pipeline vs the sims verifier's pipeline on the SAME maps (a02 outputs
against the verifier's stored per-halo values); (2) per-tile distributions of the paired loss in the build's tiles and
the verifier's tiles (is the baseline tension a tail effect or a shift?); (3) its dependence on the tile's content.

(1) per tile of a02: rows, q', sigma; credited halos (merge none, closest 1', fof 10') and their q_obs; credited
    isolated copies and q_iso; at q > 5 the count of halos credited by one pipeline only.
(2) N_iso - N_det at q > 5 per tile (credited copies minus credited halos): mean, std, skewness, kurtosis, quantiles;
    the std ratio with a bootstrap error; the same for N_det and N_iso.
(3) per-tile N_iso - N_det against the number of halos with q' > 20 and with theta500 > 8' (q' > 4), with a linear
    fit per data set.
Usage: python a03_cross_compare.py
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arblib as A
A.threads(1)
import numpy as np
from scipy import stats as SS

E = 5.0


def cross(mode):
    fs = sorted(glob.glob(os.path.join(A.ARB, "xpipe", f"x_{mode}_*.npz")))
    if not fs:
        return
    tot = dict(rows_eq=0, ntile=0, dq_max=0.0, dsig=0.0, dqp=0.0)
    cnt = {k: np.zeros(4, np.int64) for k in ("none", "closest1", "fof10", "iso")}     # both, build only, sims only, total sims
    dqo, dqi = [], []
    for fn in fs:
        t = int(os.path.basename(fn)[-8:-4])
        a = np.load(fn)
        with np.load(os.path.join(A.SIMS, "tiles_whitened", f"t_{t:04d}.npz")) as s:
            srows = s[mode + "_rows"]; sqp = s[mode + "_qp"]
            sq = {m: s[f"{mode}_qobs_{m}"] for m in A.MERGES}; siso = s[mode + "_qiso"]
            sr1 = s[mode + "_r1"]
        tot["ntile"] += 1
        same = np.array_equal(a["rows"], srows)
        tot["rows_eq"] += int(same)
        if not same:
            print(f"  tile {t}: rows differ ({a['rows'].size} vs {srows.size})", flush=True)
            continue
        tot["dqp"] = max(tot["dqp"], float(np.max(np.abs(a["qp"] / sqp - 1))))
        import s3blib as B
        rb = a["sig1"] / B.sigma_Y(mode, B.F.THETAS15)
        tot["dsig"] = max(tot["dsig"], float(np.max(np.abs(rb / sr1 - 1))))
        for m in A.MERGES:
            x = a[f"qobs_{m}"].astype(float); y = sq[m]
            bx, by = x > E, y > E
            cnt[m] += np.array([np.sum(bx & by), np.sum(bx & ~by), np.sum(~bx & by), np.sum(by)])
            both = np.isfinite(x) & np.isfinite(y)
            if m == "none" and both.any():
                dqo.append(x[both] - y[both])
        x = a["qiso"].astype(float); y = siso
        bx, by = x > E, y > E
        cnt["iso"] += np.array([np.sum(bx & by), np.sum(bx & ~by), np.sum(~bx & by), np.sum(by)])
        both = np.isfinite(x) & np.isfinite(y)
        dqi.append(x[both] - y[both])
    dqo = np.concatenate(dqo); dqi = np.concatenate(dqi)
    print(f"(1) {mode}: {tot['ntile']} of the verifier's tiles re-created and run through the build's code; identical halo rows "
          f"(q' >= 1.5) in {tot['rows_eq']}; max |q'_build / q'_sims - 1| = {tot['dqp']:.1e}; max |r_build / r_sims - 1| = "
          f"{tot['dsig']:.1e}", flush=True)
    for m in ("none", "closest1", "fof10", "iso"):
        b, bo, so, n = cnt[m]
        lab = "isolated copies" if m == "iso" else f"credited halos, merge {m}"
        print(f"    {lab:28s} q > 5: both {b}, build pipeline only {bo}, verifier only {so} (of {n})", flush=True)
    print(f"    q_obs (credited by both, merge none): max |difference| {np.abs(dqo).max():.2e}; q_iso: max |difference| "
          f"{np.abs(dqi).max():.2e}, mean {dqi.mean():+.2e}, rms {dqi.std():.2e} ({dqi.size} copies)", flush=True)


def dist(label, D):
    nt = D["ntiles"]
    nd = A.per_tile(D["t"], (D["qo_none"] > E).astype(float), nt)
    ni = A.per_tile(D["t"], (D["qiso"] > E).astype(float), nt)
    x = ni - nd
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    rng = np.random.default_rng(3)
    sb = np.array([x[rng.integers(0, nt, nt)].std(ddof=1) for _ in range(2000)])
    print(f"  {label:16s} tiles {nt:5d}: N_iso - N_det mean {x.mean():5.2f} +- {x.std(ddof=1) / np.sqrt(nt):.2f}, std "
          f"{x.std(ddof=1):.2f} +- {sb.std():.2f}, skew {SS.skew(x):+.2f}, excess kurtosis {SS.kurtosis(x):+.2f}; percentiles "
          f"1/5/25/50/75/95/99: {', '.join(f'{v:g}' for v in q)}; std of N_det {nd.std(ddof=1):.2f}, of N_iso {ni.std(ddof=1):.2f}",
          flush=True)
    bright = A.per_tile(D["t"], (D["qp"] > 20).astype(float), nt)
    big = A.per_tile(D["t"], ((D["th"] > 8) & (D["qp"] > 4)).astype(float), nt)
    X = np.stack([np.ones(nt), bright - bright.mean(), big - big.mean()], 1)
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    res = x - X @ coef
    cov = np.linalg.inv(X.T @ X) * res.var(ddof=3)
    print(f"      per tile: halos with q' > 20 {bright.mean():.2f}, with theta500 > 8' and q' > 4 {big.mean():.2f}; fit N_iso - N_det = "
          f"{coef[0]:.2f} + {coef[1]:.3f} +- {np.sqrt(cov[1, 1]):.3f} x (bright - mean) + {coef[2]:.3f} +- {np.sqrt(cov[2, 2]):.3f} x (big - mean)",
          flush=True)
    return x


if __name__ == "__main__":
    t00 = time.time()
    for mode in ("baseline", "goal"):
        print(f"=================== {mode} ===================", flush=True)
        cross(mode)
        print(f"(2), (3) per-tile paired loss at q > {E:.0f}", flush=True)
        xs = {}
        for lab, D in (("build Q2", A.load_build(mode, "q2")), ("build new tiles", A.load_build(mode, "new")),
                       ("sims verifier", A.load_sims(mode))):
            xs[lab] = dist(lab, D)
        xb = np.concatenate([xs["build Q2"], xs["build new tiles"]])
        u = SS.mannwhitneyu(xs["sims verifier"], xb)
        ks = SS.ks_2samp(xs["sims verifier"], xb)
        lv = SS.levene(xs["sims verifier"], xb)
        d = xs["sims verifier"].mean() - xb.mean()
        e = np.sqrt(xs["sims verifier"].var(ddof=1) / xs["sims verifier"].size + xb.var(ddof=1) / xb.size)
        print(f"  sims - build (all 1,548 tiles): mean {d:+.2f} +- {e:.2f} per tile ({d / e:+.2f} sigma); Mann-Whitney p = {u.pvalue:.3f}; "
              f"KS p = {ks.pvalue:.3f}; Levene (equal variance) p = {lv.pvalue:.3f}", flush=True)
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {A.peak_rss_gb():.2f} GB", flush=True)
