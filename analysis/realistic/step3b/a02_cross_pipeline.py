"""Step 3b arbitration / a02: the build's pipeline on the sims verifier's own skies (claims N1, N3; a01's 3.5 sigma
tension of the baseline paired count change).

For the sims verifier's tiles t (verify_real/step3b_sims/y02_tiles.py), the SAME input maps are re-created with the
verifier's own generators (vlib3b: draw_tile_halos(9_310_000 + t), its NUFFT painter 'lag4' with eps 1e-9, cmb_mine
(seed 9_320_000 + t), noise_mine (seed 9_330_000 + t)), converted from its compact in-band mode list to the build's rfft
layout (out-of-band modes are zero; the finder uses in-band modes only), and put through the BUILD's code, exactly as
b15_newtiles.py: szsim.finder.run_finder (production, whitened mask), b15.credit (table contributions + exact lattice
values of the 4 largest; primary = largest exact contributor, real if >= 2 sigma), credited q_obs per halo (merge none,
closest 1', fof 10'); the halo-free tile (CMB + noise) through the finder, its S/N cube and response tables with the
lattice harmonics, and every halo with q' >= 1.9 inserted alone with the same q' (s3blib.run_virtual, b03 window).
Output per tile: data_real/step3b/arbiter/xpipe/x_MODE_TTTT.npz (halo index, q', q_obs per merge, q_iso, sigma's).
The comparison with the verifier's stored per-halo values is a03_cross_compare.py.
Usage: python a02_cross_pipeline.py MODE T0 T1 NPROC
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
from multiprocessing import Pool                   # noqa: E402
import b03_virtual as B3                           # noqa: E402
import b15_newtiles as B15                         # noqa: E402
F = B.F
SIMS_DIR = os.path.join(B.ROOT, "verify_real", "step3b_sims")
SEED_H, SEED_C, SEED_N = 9_310_000, 9_320_000, 9_330_000
OUTD = os.path.join(B.OUT, "arbiter", "xpipe")
_S = {}


def sims_lib():
    if "V" not in _S:
        sys.path.insert(0, SIMS_DIR)
        import vlib3b as V                         # the verifier's generators (read only)
        _S["V"] = V
        _S["VG"] = V.grid()
        _S["P"] = V.Painter(_S["VG"], eps=1e-9, scheme="lag4")
    return _S["V"], _S["VG"], _S["P"]


def to_rfft(Xc, VG):
    """Compact in-band modes (nm, 6) in numpy full-fft order -> the build's rfft layout (6, N, N/2 + 1)."""
    X = np.zeros((6, F.N, F.N // 2 + 1), complex)
    m = VG.mcol <= F.N // 2
    X[:, VG.mrow[m], VG.mcol[m]] = Xc[m].T
    return X


def work(job):
    mode, t = job
    fn = os.path.join(OUTD, f"x_{mode}_{t:04d}.npz")
    if os.path.exists(fn):
        return job, 0.0, B.peak_rss_gb(), None
    t0 = time.time()
    V, VG, P = sims_lib()
    G = F.grid()
    bas, tables = B.basis()
    h = V.draw_tile_halos(SEED_H + t)
    hh = {k: np.asarray(h[k], np.float64) for k in ("y", "x", "z", "theta500", "Y500")}
    Xt = to_rfft(V.tsz_channels(P.T(hh["y"], hh["x"], hh["Y500"], hh["theta500"]), VG), VG)
    Xcn = to_rfft(V.cmb_mine(np.random.default_rng(SEED_C + t), VG) + V.noise_mine(np.random.default_rng(SEED_N + t), mode, VG), VG)
    t_gen = time.time() - t0
    t1 = time.time()
    its = F.run_finder(Xt + Xcn, G)
    filt, det = its[-1]["filt"], its[-1]["det"]
    sig = filt.sigma_arcmin2.copy()
    sig0 = its[0]["filt"].sigma_arcmin2.copy() if hasattr(its[0]["filt"], "sigma_arcmin2") else None
    sel, prim, cps = B15.credit(filt, det, hh, bas)
    del its, filt
    real = cps >= B.CRED_MIN
    qp = hh["Y500"] / B.sigma_finder(sig, mode, hh["theta500"])
    rows = np.flatnonzero(qp >= 1.5)
    dsel = dict(q=det["q"], y=det["y"], x=det["x"])
    idx_of = np.full(det["q"].size, -1, np.int64); idx_of[sel] = np.arange(sel.size)
    res = dict(tile=t, mode=mode, rows=rows.astype(np.int32), qp=qp[rows].astype(np.float32), sig1=sig,
               n5=int(sel.size), nreal=int(real.sum()))
    for m in B.MERGES:
        k = F.apply_merge(dsel, m, 5.0)
        a = idx_of[k]
        a = a[(a >= 0) & real[a]]
        qo = np.full(hh["z"].size, -np.inf)
        np.maximum.at(qo, prim[a], det["q"][sel[a]])
        res[f"qobs_{m}"] = qo[rows].astype(np.float32)
        res[f"ncred_{m}"] = int(np.isfinite(qo).sum())
        res[f"ncred_out_{m}"] = int(np.sum(np.isfinite(qo) & (qp < 1.5)))
    res["det5_q"] = det["q"][sel].astype(np.float32); res["det5_prim"] = prim.astype(np.int32); res["det5_cp"] = cps.astype(np.float32)
    t_real = time.time() - t1
    t1 = time.time()
    itsn, _ = B.noise_only_filter(t, mode, "prod", G, X=Xcn)
    fn_, qn = itsn[-1]["filt"], itsn[-1]["q"]
    sign = fn_.sigma_arcmin2.copy()
    tabs = B.filter_tables(fn_, bas, G)
    iso = np.full(rows.size, -np.inf)
    m19 = qp[rows] >= 1.9
    rr = rows[m19]
    Yi = qp[rr] * B.sigma_finder(sign, mode, hh["theta500"][rr])
    oi = B.run_virtual(qn, sign, tabs, hh["y"][rr], hh["x"][rr], Yi, hh["theta500"][rr], B.window_pix(hh["theta500"][rr], B.TH_K, B3.RW))
    iso[m19] = oi["q"]
    res["qiso"] = iso.astype(np.float32); res["sig_free"] = sign
    del itsn, qn, tabs, fn_
    t_iso = time.time() - t1
    res["timing"] = np.array([t_gen, t_real, t_iso])
    os.makedirs(OUTD, exist_ok=True)
    np.savez(fn, **res)
    return job, time.time() - t0, B.peak_rss_gb(), (t_gen, t_real, t_iso, rows.size)


if __name__ == "__main__":
    mode, T0, T1, NPROC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    jobs = [(mode, t) for t in range(T0, T1)]
    t00 = time.time()
    print(f"a02: the build's pipeline on the sims verifier's tiles {T0}..{T1 - 1}, {mode}, {NPROC} processes", flush=True)
    rss_max = 0.0
    it = map(work, jobs) if NPROC == 1 else Pool(NPROC, maxtasksperchild=25).imap_unordered(work, jobs)
    for i, (job, tt, rss, tim) in enumerate(it):
        rss_max = max(rss_max, rss)
        if tim is not None and (i % 10 == 0 or i == len(jobs) - 1):
            print(f"  tile {job[1]}: {tt:.1f} s (maps {tim[0]:.1f}, realistic {tim[1]:.1f}, copies {tim[2]:.1f}; {tim[3]} rows); "
                  f"worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]", flush=True)
    print(f"a02 done: {len(jobs)} tiles in {time.time() - t00:.0f} s wall; largest worker peak RSS {rss_max:.2f} GB", flush=True)
