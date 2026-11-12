"""Step 3b / b01: the finder with SZiFi's own masked covariance on the 324 realistic tiles per noise level.

Variants (s3blib.VARIANTS): 'szifi3000' = SZiFi's defaults (diffusive inpainting with 100 sweeps, cosine blend, binned
spectra above l_sep = 3000 divided by the unmasked fraction); 'szifi12000' = the same with l_sep = 12000 (SZiFi's SO
example script).  Step 3a ran 'szifi3000' on tiles 0-99 (data_real/step3/tiles_szifi, t04 with MASK = szifi); those files
are reused, and tile 0 is rerun here as a regeneration check.
Per tile pid (as step 3a's t04): paint the halos once (szsim.finder.paint_y; truth from step 3a's truth tables); per
noise level: CMB (seed 300000 + pid) + noise (seed 400000 + pid); per variant: run_finder (estimated covariance, q > 4
masked with 3 theta discs, rerun once), then the exact decomposition of every final detection (s3lib.decompose: table
contributions of all halos, exact lattice values of the 4 largest contributors for q >= 4.5, component values),
with forced = False (forced photometry and the isolated search only for the detections' primaries; the covariance
diagnostics of t04 are skipped).  Output keys follow t04 ({mode}_it{k}_sig/_q/_j/_iy/_ix/_C, {mode}_d_top_idx,
{mode}_d_top_ex, ...).
Outputs: data_real/step3b/tiles_szifi3000/tile_PID.npz (pids 100-323, and 0 for the check),
         data_real/step3b/tiles_szifi12000/tile_PID.npz (pids 0-323).
Usage: python b01_tiles_szifi.py PID0 PID1 NPROC
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
from multiprocessing import Pool                   # noqa: E402
F, S3 = B.F, B.S3
CHECK = (0,)                                       # pids where szifi3000 is rerun and compared with step 3a's file


def variants_for(pid):
    return (["szifi3000"] if (pid >= 100 or pid in CHECK) else []) + ["szifi12000"]


def work(pid):
    todo = [v for v in variants_for(pid) if not os.path.exists(os.path.join(B.OUT, f"tiles_{v}", f"tile_{pid:03d}.npz"))]
    if not todo:
        return pid, 0.0, B.peak_rss_gb(), None
    t0 = time.time()
    G = F.grid()
    bas, tables = B.basis()
    d = dict(np.load(os.path.join(B.D3, "truth", f"truth_{pid:03d}.npz")))
    d = {k: d[k].astype(np.float64) for k in d}
    t1 = time.time()
    Xt = F.tsz_channels(F.paint_y(d["y"], d["x"], d["Y500"], d["theta500"], tables, G), G)
    t_paint = time.time() - t1
    res = {v: dict(pid=pid, thetas=F.THETAS15, seed_cmb=B.SEED_CMB + pid, seed_noise=B.SEED_NOISE + pid, variant=v,
                   mask_kw=str(B.VARIANTS[v]["mask_kw"])) for v in todo}
    tim = {}
    for mode in B.MODES:
        Xc = F.sim_cmb(B.SEED_CMB + pid, G)
        Xn = F.sim_noise(B.SEED_NOISE + pid, mode, G)
        X = Xt + Xc + Xn
        for v in todo:
            t1 = time.time()
            its = F.run_finder(X, G, mask_method=B.VARIANTS[v]["mask_method"], mask_kw=B.VARIANTS[v]["mask_kw"])
            t_find = time.time() - t1
            res[v].update(S3.finder_outputs(its, prefix=mode + "_"))
            t1 = time.time()
            filt = its[-1]["filt"]
            cxs = dict(tsz=filt.CX(Xt), cmb=filt.CX(Xc), noise=filt.CX(Xn))
            dec = S3.decompose(its[-1], cxs, d, bas, "q_" + mode, forced=False)
            t_dec = time.time() - t1
            res[v].update({mode + "_" + k: val for k, val in dec.items()})
            res[v][mode + "_timing"] = np.array([t_find, t_dec])
            tim[(mode, v)] = (t_find, t_dec)
            del its, cxs, filt
        del X, Xc, Xn
    for v in todo:
        res[v]["t_paint"] = t_paint
        res[v]["t_total"] = time.time() - t0
        res[v]["peak_rss_gb"] = B.peak_rss_gb()
        np.savez(os.path.join(B.OUT, f"tiles_{v}", f"tile_{pid:03d}.npz"), **res[v])
    return pid, time.time() - t0, B.peak_rss_gb(), (t_paint, tim)


if __name__ == "__main__":
    P0, P1, NPROC = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    for v in ("szifi3000", "szifi12000"):
        os.makedirs(os.path.join(B.OUT, f"tiles_{v}"), exist_ok=True)
    t00 = time.time()
    pids = list(range(P0, P1))
    print(f"b01: tiles {P0}..{P1 - 1}, {NPROC} processes; variants per tile: szifi3000 for pid >= 100 (and {CHECK} as a check), "
          f"szifi12000 for all", flush=True)
    rss_max = 0.0
    with Pool(NPROC, maxtasksperchild=20) as pool:
        for pid, tt, rss, tim in pool.imap_unordered(work, pids):
            rss_max = max(rss_max, rss)
            if tim is None:
                print(f"  tile {pid:3d}: exists", flush=True)
                continue
            tp, tm = tim
            print(f"  tile {pid:3d}: {tt:5.1f} s (paint {tp:.1f}; " + "; ".join(f"{m} {v}: find {a:.1f} decomp {b:.1f}" for (m, v), (a, b) in tm.items())
                  + f"); worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]", flush=True)
    print(f"b01 runs done: {len(pids)} tiles in {time.time() - t00:.0f} s wall; largest worker peak RSS {rss_max:.2f} GB", flush=True)
    # ---- regeneration check: szifi3000 on the CHECK tiles vs step 3a's tiles_szifi
    for pid in CHECK:
        fn = os.path.join(B.OUT, "tiles_szifi3000", f"tile_{pid:03d}.npz")
        if pid not in pids or not os.path.exists(fn):
            continue
        a = np.load(fn); b = np.load(os.path.join(B.D3, "tiles_szifi", f"tile_{pid:03d}.npz"))
        for mode in B.MODES:
            same = all(np.array_equal(a[f"{mode}_it1_{k}"], b[f"{mode}_it1_{k}"]) for k in ("j", "iy", "ix"))
            dq = np.abs(a[f"{mode}_it1_q"] - b[f"{mode}_it1_q"]).max() if same else np.nan
            ds = np.abs(a[f"{mode}_it1_sig"] / b[f"{mode}_it1_sig"] - 1).max()
            de = np.nanmax(np.abs(a[f"{mode}_d_top_ex"] - b[f"{mode}_d_top_ex"]))
            print(f"  regeneration check tile {pid} {mode}: szifi3000 catalogue identical to step 3a's tiles_szifi: {same}; "
                  f"max |dq| {dq:.1e}; max |sigma ratio - 1| {ds:.1e}; max |d top_ex| {de:.1e} arcmin^2; same top_idx: "
                  f"{np.array_equal(a[f'{mode}_d_top_idx'], b[f'{mode}_d_top_idx'])}", flush=True)
    print(f"b01 done [{time.time() - t00:.0f} s]", flush=True)
