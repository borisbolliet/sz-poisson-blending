"""Step 3b arbitration / a09: the claims no verifier checked or checked only in part, with small targeted reruns.

(1) F1: SZiFi's defaults rerun on tile 0 (szsim.finder.run_finder, mask_method 'szifi', l_sep 3000, blend, 100 sweeps; the
    tile painted from step 3a's truth table, CMB / noise from its seeds) against step 3a's data_real/step3/tiles_szifi
    file: final catalogue (scale, pixel, q) and sigma after the iteration.
(2) S6 for l_sep 12000: the finder on the halo-free versions of tiles 0-7 (both levels) -> r = sigma_finder / sigma_Y at
    0.5', against the stored per-tile r (b03) and the claimed mean 0.976 / 0.974.
(3) S3: b03's accuracy check of the virtual response (virtual vs exactly filtered painted cluster, 2 clusters per kernel
    node, with and without the lattice harmonics) rerun on a different halo-free tile (pid 7) and new random positions.
(4) N4: count changes at q > 5 from the stored per-halo tables for the setups the verifiers did not recompute (l_sep
    12000; closest 1'; + 'no real cluster' entries), as R_p = N_det / N_iso - 1; fof 10' removal of credited halos in the
    Q2 tiles and in the new tiles.
(5) C1, C2: the /usr/bin/time blocks of the 25 build logs re-read (sum of wall, CPU, largest RSS).
Usage: python a09_misc.py
"""
import os, sys, re, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
import arblib as A                                 # noqa: E402
F = B.F


def f1():
    G = F.grid()
    bas, tables = B.basis()
    d = {k: v.astype(np.float64) for k, v in np.load(os.path.join(B.D3, "truth", "truth_000.npz")).items()}
    Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], tables, G), G)
    ref = np.load(os.path.join(B.D3, "tiles_szifi", "tile_000.npz"))
    for mode in B.MODES:
        X = Xt + F.sim_cmb(B.SEED_CMB, G) + F.sim_noise(B.SEED_NOISE, mode, G)
        its = F.run_finder(X, G, mask_method="szifi", mask_kw=B.VARIANTS["szifi3000"]["mask_kw"])
        det, sig = its[-1]["det"], its[-1]["filt"].sigma_arcmin2
        same = all(np.array_equal(det[k], ref[f"{mode}_it1_{k}"]) for k in ("j", "iy", "ix"))
        dq = np.abs(det["q"] - ref[f"{mode}_it1_q"]).max() if same else np.nan
        ds = np.abs(sig / ref[f"{mode}_it1_sig"] - 1).max()
        print(f"(1) tile 0 {mode}: SZiFi-defaults rerun vs step 3a's file: {det['q'].size} final detections, same scales and pixels: {same}; "
              f"max |dq| {dq:.1e}; max |sigma ratio - 1| {ds:.1e}; q > 5: {int((det['q'] > 5).sum())} vs {int((ref[f'{mode}_it1_q'] > 5).sum())}", flush=True)
        del its, X


def f2():
    G = F.grid()
    for mode in B.MODES:
        r, rs = [], []
        for pid in range(8):
            its, _ = B.noise_only_filter(pid, mode, "szifi12000", G)
            r.append(B.r_of(its[-1]["filt"].sigma_arcmin2, mode)[0])
            with np.load(os.path.join(B.OUT, "virtual", "szifi12000", f"v_{mode}_{pid:03d}.npz")) as v:
                rs.append(v["r"][0])
            del its
        r, rs = np.array(r), np.array(rs)
        allr = np.array([np.load(os.path.join(B.OUT, "virtual", "szifi12000", f"v_{mode}_{p:03d}.npz"))["r"][0] for p in range(324)])
        print(f"(2) l_sep 12000, halo-free tiles 0-7, {mode}: r(0.5') rerun {r.mean():.4f} +- {r.std(ddof=1) / np.sqrt(r.size):.4f}; max |rerun / "
              f"stored - 1| {np.abs(r / rs - 1).max():.1e}; stored mean over all 324 tiles {allr.mean():.4f} (claim {'0.976' if mode == 'baseline' else '0.974'})",
              flush=True)


def f3():
    import b03_virtual as B3
    G = F.grid()
    bas, tables = B.basis()
    for mi, mode in enumerate(B.MODES):
        its, _ = B.noise_only_filter(7, mode, "prod", G)
        filt = its[-1]["filt"]; sig = filt.sigma_arcmin2.copy()
        tabs = B.filter_tables(filt, bas, G)
        err = B3.accuracy_check(filt, sig, tabs, mode, np.random.default_rng(9090 + mi), G)
        print(f"(3) tile 7 {mode}: virtual vs exact own S/N over +-2W, max / peak: with the lattice harmonics {err[:, :, 1].max():.1e} "
              f"(worst node {B.TH_K[np.unravel_index(err[:, :, 1].argmax(), err[:, :, 1].shape)[0]]:.2f}'); angle-averaged tables alone "
              f"{err[:, :, 0].max():.1e}; theta <= 2.4': {err[:10, :, 1].max():.1e} / {err[:10, :, 0].max():.1e}", flush=True)
        del its, filt, tabs


def f4():
    for mode in B.MODES:
        W324 = A.boot_w(324, 2000, 41)
        for variant in ("szifi12000", "szifi3000", "prod"):
            D = A.load_build(mode, "q2", variant)
            nd = A.per_tile(D["t"], (D["qo_none"] > 5).astype(float), 324); ni = A.per_tile(D["t"], (D["qiso"] > 5).astype(float), 324)
            v, e, _ = A.ratio_boot(nd, ni, W324)
            print(f"(4) {mode} {variant:10s} (324 Q2 tiles): R_p(q > 5) = {A.fmt(v, e)}", flush=True)
        D = A.load_build(mode, "all")
        nt = D["ntiles"]
        W = A.boot_w(nt, 2000, 42)
        ni = A.per_tile(D["t"], (D["qiso"] > 5).astype(float), nt)
        nd = A.per_tile(D["t"], (D["qo_none"] > 5).astype(float), nt)
        for m in ("closest1", "fof10"):
            x = A.per_tile(D["t"], (D["qo_" + m] > 5).astype(float), nt)
            v, e, _ = A.ratio_boot(x, ni, W)
            q2 = D["t"] < 324
            rq = 1 - np.sum(D["qo_" + m][q2] > 5) / np.sum(D["qo_none"][q2] > 5)
            rn = 1 - np.sum(D["qo_" + m][~q2] > 5) / np.sum(D["qo_none"][~q2] > 5)
            print(f"(4) {mode} production {m} (1,548 tiles): R_p(q > 5) = {A.fmt(v, e)}; credited halos removed by the merge: Q2 tiles "
                  f"{100 * rq:.2f}%, new tiles {100 * rn:.2f}%, all {100 * (1 - x.sum() / nd.sum()):.2f}%", flush=True)
        # 'no real cluster' entries at q > 5 (their primary's z): Q2 from the b02 table, new tiles from b15
        V = np.load(os.path.join(B.OUT, f"halos_prod_{mode}.npz"))
        nr = list(V["nr_q"][V["nr_q"] > 5].astype(float))
        nnr = np.bincount(V["nr_pid"][V["nr_q"] > 5].astype(np.int64), minlength=324).astype(float)
        fs = sorted(f for f in os.listdir(os.path.join(B.OUT, "newtiles")) if f.endswith(".npz"))
        extra = []
        for f in fs:
            with np.load(os.path.join(B.OUT, "newtiles", f)) as d:
                extra.append(float(np.sum(d[mode + "_nr_q"] > 5)))
        nnr = np.concatenate([nnr, np.array(extra)])
        v, e, _ = A.ratio_boot(nd + nnr, ni, W)
        print(f"(4) {mode} production + 'no real cluster' entries (1,548 tiles): R_p(q > 5) = {A.fmt(v, e)}; such entries "
              f"{100 * nnr.sum() / (nd.sum() + nnr.sum()):.2f}% of credited + no-real", flush=True)


def f5():
    tot_w, tot_c, mx = 0.0, 0.0, 0.0
    logs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "b*.log")))
    logs = [l for l in logs if os.path.basename(l) != "b12_compute.log"]
    n = 0
    for l in logs:
        s = open(l).read()
        m = re.findall(r"\n\s+([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys\n\s+(\d+)\s+maximum resident set size", s)
        if not m:
            continue
        r, u, sy, rss = m[-1]
        tot_w += float(r); tot_c += float(u) + float(sy); mx = max(mx, int(rss) / 1e9); n += 1
    print(f"(5) {n} build logs with a /usr/bin/time block: sum of wall {tot_w:.0f} s ({tot_w / 3600:.2f} h), CPU (user + sys) {tot_c:.0f} s "
          f"({tot_c / 3600:.2f} CPU-h); largest maximum RSS {mx:.2f} GB", flush=True)


if __name__ == "__main__":
    t00 = time.time()
    for f in (f5, f4, f1, f2, f3):
        t0 = time.time()
        f()
        print(f"    [{time.time() - t0:.0f} s; peak RSS {B.peak_rss_gb():.2f} GB]", flush=True)
    print(f"done [{time.time() - t00:.0f} s]", flush=True)
