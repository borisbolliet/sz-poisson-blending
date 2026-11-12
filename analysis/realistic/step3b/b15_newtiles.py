"""Step 3b / b15: the production finder on the new tiles of b14, with the credit logic of step 3a and the isolated
copies of b03 -- what the headline data vector needs, nothing else.

Per new tile pid (truth from b14) and noise level, with CMB seed 300000 + pid and noise seed 400000 + pid (step 3a's
convention; no overlap with any earlier seed):
  realistic: paint the halos (szsim.finder.paint_y), CMB + noise, the production finder (run_finder, 'whitened' mask);
     for every final detection with q > 5: table contributions of all halos and exact lattice values of the 4 largest
     (szsim.finder.table_contributions, exact_contrib; the credit part of s3lib.decompose, restricted to q > 5);
     primary = the largest exact contributor, 'real' when its contribution is >= 2 sigma; credited q_obs per halo for
     merge none, closest 1', fof 10' (as b02); the 'no real cluster' entries with their primary.
  halo-free: the same CMB + noise without halos, the production finder, its final filter's S/N cube and response tables
     (with the lattice harmonics); every halo with q' >= 1.9 inserted alone at its position with the same q' (b03 'iso').
Outputs: data_real/step3b/newtiles/n_PID.npz (per noise level: rows of the halos with q_true >= 1.5 (truth row hidx),
q', q_obs per merge rule, q_iso; sigma_finder at the 15 scales; the 'no real cluster' entries).
Usage: python b15_newtiles.py PID0 PID1 NPROC
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
from multiprocessing import Pool                   # noqa: E402
import b03_virtual as B3                           # noqa: E402
F = B.F
K_TOP = 4


def credit(filt, det, h, bas):
    """Primary (truth row) and its exact contribution / sigma for the detections with q > 5."""
    sel = np.flatnonzero(det["q"] > 5.0)
    nd = sel.size
    prim = np.full(nd, -1, np.int64); cps = np.full(nd, np.nan)
    if nd == 0:
        return sel, prim, cps
    Rn, Rf = F.response_tables(filt, bas)
    targs = (Rn, Rf, bas.lnth0, bas.dln, F.R_SWITCH, bas.rn[1] - bas.rn[0], F.R_FAR_MIN, bas.rf[1] - bas.rf[0], F.LBOX)
    hl = np.log(h["theta500"])
    s_tab = np.zeros(nd); ti = np.full((nd, K_TOP), -1, np.int64); tv = np.zeros((nd, K_TOP)); nc = np.zeros(nd, np.int64)
    F.table_contributions(det["j"][sel], det["y"][sel], det["x"][sel], h["y"], h["x"], h["Y500"], hl, *targs, K_TOP,
                          s_tab, ti, tv, nc)
    sig = filt.sigma_arcmin2
    for a, d in enumerate(sel):
        best, bh = -np.inf, -1
        for k in range(K_TOP):
            hh = ti[a, k]
            if hh < 0:
                continue
            dy = (det["y"][d] - h["y"][hh] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
            dx = (det["x"][d] - h["x"][hh] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
            v = F.exact_contrib(filt, bas, det["j"][d], h["Y500"][hh], h["theta500"][hh], dy, dx)
            if v > best:
                best, bh = v, hh
        prim[a] = bh; cps[a] = best / sig[det["j"][d]]
    return sel, prim, cps


def work(pid):
    fn = os.path.join(B.OUT, "newtiles", f"n_{pid:04d}.npz")
    if os.path.exists(fn):
        return pid, 0.0, B.peak_rss_gb(), None
    t0 = time.time()
    G = F.grid()
    bas, tables = B.basis()
    with np.load(os.path.join(B.OUT, "truth_new", f"truth_{pid:04d}.npz")) as t:
        h = {k: t[k].astype(np.float64) for k in ("x", "y", "z", "theta500", "Y500", "q_baseline", "q_goal")}
    Xt = F.tsz_channels(F.paint_y(h["y"], h["x"], h["Y500"], h["theta500"], tables, G), G)
    res = dict(pid=pid)
    tim = []
    for mode in B.MODES:
        t1 = time.time()
        Xcn = F.sim_cmb(B.SEED_CMB + pid, G) + F.sim_noise(B.SEED_NOISE + pid, mode, G)
        its = F.run_finder(Xt + Xcn, G)
        filt, det = its[-1]["filt"], its[-1]["det"]
        sig = filt.sigma_arcmin2.copy()
        sel, prim, cps = credit(filt, det, h, bas)
        del its, filt
        real = cps >= B.CRED_MIN
        rows = np.flatnonzero(h["q_" + mode] >= 1.5)
        pos = np.full(h["z"].size, -1, np.int64); pos[rows] = np.arange(rows.size)
        qp = h["Y500"][rows] / B.sigma_finder(sig, mode, h["theta500"][rows])
        dsel = dict(q=det["q"], y=det["y"], x=det["x"])
        idx_of = np.full(det["q"].size, -1, np.int64); idx_of[sel] = np.arange(sel.size)
        for m in B.MERGES:
            k = F.apply_merge(dsel, m, 5.0)
            a = idx_of[k]
            a = a[(a >= 0) & real[a]]
            qo = np.full(h["z"].size, -np.inf)
            np.maximum.at(qo, prim[a], det["q"][sel[a]])
            assert np.all(pos[np.flatnonzero(np.isfinite(qo))] >= 0)
            res[f"{mode}_qobs_{m}"] = qo[rows].astype(np.float32)
        nr = np.flatnonzero(~real)
        res.update({f"{mode}_rows": rows.astype(np.int32), f"{mode}_qp": qp.astype(np.float32), f"{mode}_sig1": sig,
                    f"{mode}_nr_q": det["q"][sel[nr]].astype(np.float32), f"{mode}_nr_hidx": prim[nr].astype(np.int32),
                    f"{mode}_nr_cp_sig": cps[nr].astype(np.float32), f"{mode}_n5": int(sel.size)})
        t2 = time.time()
        # halo-free tile: isolated copies
        itsn, _ = B.noise_only_filter(pid, mode, "prod", G, X=Xcn)
        fn_, qn = itsn[-1]["filt"], itsn[-1]["q"]
        sign = fn_.sigma_arcmin2.copy()
        tabs = B.filter_tables(fn_, bas, G)
        iso = np.full(rows.size, -np.inf)
        m19 = qp >= 1.9
        rr = rows[m19]
        Yi = qp[m19] * B.sigma_finder(sign, mode, h["theta500"][rr])
        oi = B.run_virtual(qn, sign, tabs, h["y"][rr], h["x"][rr], Yi, h["theta500"][rr], B.window_pix(h["theta500"][rr], B.TH_K, B3.RW))
        iso[m19] = oi["q"]
        res[f"{mode}_qiso"] = iso.astype(np.float32); res[f"{mode}_sig_noise"] = sign
        del itsn, qn, tabs, Xcn
        tim.append((t2 - t1, time.time() - t2))
    res["t"] = time.time() - t0
    np.savez(fn, **res)
    return pid, res["t"], B.peak_rss_gb(), tim


if __name__ == "__main__":
    P0, P1, NPROC = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(os.path.join(B.OUT, "newtiles"), exist_ok=True)
    pids = [p for p in range(P0, P1) if os.path.exists(os.path.join(B.OUT, "truth_new", f"truth_{p:04d}.npz"))]
    t00 = time.time()
    print(f"b15: new tiles {pids[0]}..{pids[-1]} ({len(pids)}), production finder, {NPROC} processes", flush=True)
    rss_max = 0.0
    with Pool(NPROC, maxtasksperchild=30) as pool:
        for i, (pid, tt, rss, tim) in enumerate(pool.imap_unordered(work, pids)):
            rss_max = max(rss_max, rss)
            if tim is not None and (i % 25 == 0 or i == len(pids) - 1):
                print(f"  tile {pid}: {tt:.1f} s (" + "; ".join(f"{m}: realistic {a:.1f}, halo-free {b:.1f}" for m, (a, b) in zip(B.MODES, tim))
                      + f"); worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]", flush=True)
    print(f"b15 done: {len(pids)} tiles in {time.time() - t00:.0f} s wall; largest worker peak RSS {rss_max:.2f} GB", flush=True)
