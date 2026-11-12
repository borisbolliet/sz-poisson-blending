"""Step 3b / b02: per-halo tables of the realistic tiles (324 per noise level) for each finder variant.

Credit logic as step 3a's t05: a final detection (iteration 1) is 'real' when the exact lattice contribution of its
largest contributor (the primary) is >= 2 sigma at the detection's scale; a halo is credited when it is the primary of
at least one real detection with q > 5 in the catalogue after the q > 5 cut and the merge rule; its q_obs is the
highest q among those detections.  Merge rules (szsim.finder.apply_merge, cut first, then merge): none, 'closest' 1',
'fof' 10'.  The 'no real cluster' detections (q > 5, c_p < 2 sigma, merge none) are kept with their primary.

Layout (every file: numpy npz, arrays aligned within a file):
  data_real/step3b/halos_MODE.npz          one row per halo with q_true = Y500 / sigma_Y(theta500) >= 1.5 (no halo below
                                           can be credited: the credit limit is q' = 2 and r > 1); columns
     pid (int16)       tile id (step 3a / Q2 patch pid)
     hidx (int32)      row of the halo in data_real/step3/truth/truth_PID.npz
     z, theta500 [arcmin], Y500 [arcmin^2, lognormal scatter included], M500 [Msun], x, y [arcmin], q_true (float32)
  data_real/step3b/halos_VARIANT_MODE.npz  the same rows, for one finder variant
     qp (float32)                q' = Y500 / sigma_finder(theta500) of the halo's tile (s3blib.sigma_finder)
     qobs_none, qobs_closest1, qobs_fof10 (float32)   q_obs of the halo's highest credited q > 5 detection under that
                                 merge rule; -inf if not credited
     r1 (324, 15)                sigma_finder / sigma_Y after the iteration, per tile and filter scale (THETAS15)
     sig1 (324, 15)              sigma_finder [arcmin^2] after the iteration
     nr_pid, nr_q, nr_hidx, nr_z, nr_cp_sig, nr_j   the 'no real cluster' detections (q > 5, merge none): tile, q_obs,
                                 primary's truth row and z, c_p / sigma of the primary, scale index
     n5 (324,)                   q > 5 catalogue entries per tile (merge none)
Usage: python b02_halos.py VARIANT [VARIANT ...]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(2)
import numpy as np                                 # noqa: E402
F = B.F
QFLOOR = 1.5


def truth_rows(pid):
    with np.load(os.path.join(B.D3, "truth", f"truth_{pid:03d}.npz")) as t:
        d = {k: t[k] for k in ("x", "y", "z", "M500", "theta500", "Y500", "q_baseline", "q_goal")}
    return d


def common(mode):
    fn = os.path.join(B.OUT, f"halos_{mode}.npz")
    if os.path.exists(fn):
        return {k: v for k, v in np.load(fn).items()}
    cols = {k: [] for k in ("pid", "hidx", "z", "theta500", "Y500", "M500", "x", "y", "q_true")}
    for pid in range(324):
        d = truth_rows(pid)
        k = np.flatnonzero(d["q_" + mode] >= QFLOOR)
        cols["pid"].append(np.full(k.size, pid, np.int16)); cols["hidx"].append(k.astype(np.int32))
        for a, b in (("z", "z"), ("theta500", "theta500"), ("Y500", "Y500"), ("M500", "M500"), ("x", "x"), ("y", "y"),
                     ("q_true", "q_" + mode)):
            cols[a].append(d[b][k].astype(np.float32))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(fn, **out)
    return out


def tile_variant(pid, mode, variant, nh):
    fn = os.path.join(B.tile_dir(variant)[pid], f"tile_{pid:03d}.npz")
    with np.load(fn) as r:
        sig = r[f"{mode}_it1_sig"]; q = r[f"{mode}_it1_q"]; j = r[f"{mode}_it1_j"]
        iy = r[f"{mode}_it1_iy"]; ix = r[f"{mode}_it1_ix"]; ti = r[f"{mode}_d_top_idx"]; ex = r[f"{mode}_d_top_ex"]
    sd = sig[j]
    real = np.zeros(q.size, bool)
    ok = np.isfinite(ex[:, 0])
    real[ok] = ex[ok, 0] / sd[ok] >= B.CRED_MIN
    det = dict(q=q, y=iy * F.PIX, x=ix * F.PIX)
    qo = {}
    for m in B.MERGES:
        k = F.apply_merge(det, m, 5.0)
        kr = k[real[k]]
        a = np.full(nh, -np.inf)
        np.maximum.at(a, ti[kr, 0], q[kr])
        qo[m] = a
    k = F.apply_merge(det, "none", 5.0)
    knr = k[~real[k]]
    nr = dict(q=q[knr], prim=ti[knr, 0], cp_sig=np.where(ok[knr], ex[knr, 0], np.nan) / sd[knr], j=j[knr])
    return sig, qo, nr, int((q > 5).sum()), int(k.size), int(real[k].sum())


if __name__ == "__main__":
    t00 = time.time()
    variants = sys.argv[1:] or list(B.VARIANTS)
    for mode in B.MODES:
        C = common(mode)
        print(f"==================== {mode}: {C['pid'].size} halos with q_true >= {QFLOOR} in 324 tiles "
              f"({C['pid'].size / 324:.0f} per tile)", flush=True)
        start = np.searchsorted(C["pid"], np.arange(325))
        for v in variants:
            t0 = time.time()
            qp = np.zeros(C["pid"].size, np.float32)
            qobs = {m: np.full(C["pid"].size, -np.inf, np.float32) for m in B.MERGES}
            r1 = np.zeros((324, 15)); sig1 = np.zeros((324, 15)); n5 = np.zeros(324, np.int64)
            NR = {k: [] for k in ("pid", "q", "hidx", "z", "cp_sig", "j")}
            lost = {m: 0 for m in B.MERGES}
            ncred = {m: 0 for m in B.MERGES}
            nreal_entries = 0
            below = 0
            for pid in range(324):
                d = truth_rows(pid)
                nh = d["z"].size
                sig, qo, nr, n5_, nent, nre = tile_variant(pid, mode, v, nh)
                nreal_entries += nre
                sig1[pid] = sig; r1[pid] = B.r_of(sig, mode); n5[pid] = n5_
                rows = C["hidx"][start[pid]:start[pid + 1]]
                th = d["theta500"].astype(float); Y = d["Y500"].astype(float)
                qp_all = Y / B.sigma_finder(sig, mode, th)
                qp[start[pid]:start[pid + 1]] = qp_all[rows]
                for m in B.MERGES:
                    cr = np.flatnonzero(np.isfinite(qo[m]))
                    ncred[m] += cr.size
                    lost[m] += np.setdiff1d(cr, rows).size                 # credited halos outside the table (must be 0)
                    qobs[m][start[pid]:start[pid + 1]] = qo[m][rows]
                    if m == "none":
                        below += int((qp_all[cr] < B.CRED_MIN).sum())
                NR["pid"].append(np.full(nr["q"].size, pid, np.int16)); NR["q"].append(nr["q"].astype(np.float32))
                NR["hidx"].append(nr["prim"].astype(np.int32)); NR["z"].append(d["z"][nr["prim"]].astype(np.float32))
                NR["cp_sig"].append(nr["cp_sig"].astype(np.float32)); NR["j"].append(nr["j"].astype(np.int8))
            NR = {k: np.concatenate(a) for k, a in NR.items()}
            out = dict(qp=qp, r1=r1, sig1=sig1, n5=n5, **{f"qobs_{m}": a for m, a in qobs.items()},
                       **{f"nr_{k}": a for k, a in NR.items()})
            np.savez(os.path.join(B.OUT, f"halos_{v}_{mode}.npz"), **out)
            print(f"  [{v}] q > 5 catalogue entries {n5.sum()}; real {nreal_entries}; 'no real cluster' {NR['q'].size} "
                  f"({100 * NR['q'].size / n5.sum():.2f}% of entries); credited halos: "
                  + ", ".join(f"{m} {ncred[m]}" for m in B.MERGES)
                  + f"; credited halos outside the table (q_true < {QFLOOR}): " + ", ".join(f"{lost[m]}" for m in B.MERGES)
                  + f"; credited with q' < 2 (credit limit): {below}", flush=True)
            print(f"     r after the iteration (mean over tiles) at 0.5' / 1.04' / 4.45' / 15': "
                  + " / ".join(f"{v_:.4f}" for v_ in r1.mean(0)[[0, 3, 9, 14]])
                  + f"; tile-to-tile std " + " / ".join(f"{v_:.4f}" for v_ in r1.std(0)[[0, 3, 9, 14]])
                  + f"   [{time.time() - t0:.0f} s]", flush=True)
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {B.peak_rss_gb():.2f} GB")
