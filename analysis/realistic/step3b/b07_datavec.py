"""Step 3b / b07: the finder's data vector in step 2's bins, per variant, merge rule and noise level.

Bins: cosmolib.default_bins(q_cut), q_cut = 5 and 6 (16 z bins 0.01-3 x 9 q_obs bins from q_cut, the last one > 100).
Halos: the 324 realistic tiles per noise level (b02 tables, q_true >= 1.5; no halo below can be credited).
  N_det(z, q)  credited halos (each once, at the q_obs of its highest credited detection, at its true z);
  N_exp(z, q)  sum over halos of K(theta500, q'; e_lo) - K(theta500, q'; e_hi), K = the variant's no-blending kernel
               (b05), at each halo's own Y500 and its tile's sigma_finder (q' from b02);
  N_iso(z, q)  the same halos inserted alone, at their positions and q', into the noise-only version of their tile
               (b03 'iso'): one draw of the isolated selection on the same CMB + noise.
Three estimators of the blending ratio, all with the same tile bootstrap (NBOOT resamples of the 324 tiles):
  R   = N_det / N_exp per bin (the task's definition);
  R_p = N_det / N_iso per bin (paired: the binomial scatter of the selection is common to both);
  R_s = pooled paired correction.  On a Poisson sky a halo's neighbours do not depend on its redshift, so the blended
        selection is a function of (theta500, q') only.  The correction Delta K(theta, q'; e) = P_blended - K is
        estimated from all halos at once: d_h(e) = 1[q_obs,h > e] - 1[q_iso,h > e] (+ the halo's 'no real cluster'
        detections above e for prod_keepnr), averaged in bilinear cells (25 kernel theta nodes x the q' nodes
        s3bcosmo.QDK); R_s(z, q) = mu_{K + Delta K}(truth) / mu_K(truth) with the analyst's count model (s3bcosmo,
        exact by the cell integrals: mu is linear in the kernel).
  R = 1 (R_p: N_iso < 1) in bins with N_exp < 1.  Checks: N_iso / N_exp; chi^2 of R and R_p against R_s.
Setups: prod_none (headline), prod_closest1, prod_fof10 (the analyst's isolated kernel and N_iso of prod),
prod_keepnr (production, merge none, plus the 'no real cluster' detections at their primary's z), szifi3000_none,
szifi12000_none.  Printed: totals and step 3a's six coarse z bins at q > 5, per q bin; step 3a's preview recomputed on
the same halos with step 3a's selection (t05: pooled t03 + a02 fit, credit limit q_inj = 2 r_inj, q_inj = q' r_inj).
The production setups use the 324 Q2 tiles plus the new Poisson tiles of b14 / b15 (precision target); prod_none_q2 is
the headline on the 324 Q2 tiles alone.
Outputs: data_real/step3b/datavec_SETUP_MODE.npz (per q_cut: Ndet, Nexp, Niso, mu, R, Rp, Rs, their bootstrap
resamples and errors; dK).  Usage: python b07_datavec.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(2)
import numpy as np                                 # noqa: E402
from scipy.special import ndtr                     # noqa: E402
import s3bcosmo as SC                              # noqa: E402
F, CL = B.F, SC.CL
NBOOT = 2000
ZC = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
SETUPS = {"prod_none": ("prod", "none", False), "prod_closest1": ("prod", "closest1", False), "prod_fof10": ("prod", "fof10", False),
          "prod_keepnr": ("prod", "none", True), "szifi3000_none": ("szifi3000", "none", False),
          "szifi12000_none": ("szifi12000", "none", False), "prod_none_q2": ("prod", "none", False)}
# production setups use the 324 Q2 tiles plus the new tiles of b14 / b15 (when present); 'prod_none_q2' = the headline on
# the 324 Q2 tiles alone (comparison)
rng = np.random.default_rng(2026)
BOOT = rng.integers(0, 324, (NBOOT, 324))
BOOT100 = rng.integers(0, 100, (NBOOT, 100))
_BT = {324: BOOT}


def boot_idx(nt):
    """Tile-bootstrap resamples for nt tiles (the 324-tile set is the same for every setup)."""
    if nt not in _BT:
        _BT[nt] = np.random.default_rng(2026 + nt).integers(0, nt, (NBOOT, nt))
    return _BT[nt]


def boot_w(nt):
    b = boot_idx(nt)
    return np.stack([np.bincount(x, minlength=nt) for x in b]).astype(np.float32)
NE = B.EDGES.size
NCELL = B.TH_K.size * SC.QDK.size


def iso_qobs(variant, mode, nrow):
    q = np.full(nrow, -np.inf)
    seen = np.zeros(nrow, bool)
    for p in range(324):
        with np.load(os.path.join(B.OUT, "virtual", variant, f"v_{mode}_{p:03d}.npz")) as d:
            q[d["iso_row"]] = d["iso_qobs"]; seen[d["iso_row"]] = True
    return q, seen


def per_tile_hist(pid, z, q, zed, qed, w=None, nt=324):
    nz, nq = zed.size - 1, qed.size - 1
    iz = np.searchsorted(zed, z, side="right") - 1
    iq = np.searchsorted(qed, q, side="right") - 1
    ok = (iz >= 0) & (iz < nz) & (iq >= 0) & (iq < nq)
    idx = (pid[ok].astype(np.int64) * nz + iz[ok]) * nq + iq[ok]
    return np.bincount(idx, weights=None if w is None else w[ok], minlength=nt * nz * nq).reshape(nt, nz, nq)


def per_tile_exp(pid, z, Kh, sel, zed, nt=324):
    nz = zed.size - 1
    Ke = Kh[:, sel]
    D = Ke - np.concatenate([Ke[:, 1:], np.zeros((Ke.shape[0], 1))], 1)
    iz = np.searchsorted(zed, z, side="right") - 1
    ok = (iz >= 0) & (iz < nz)
    out = np.zeros((nt, nz, len(sel)))
    np.add.at(out, (pid[ok].astype(np.int64), iz[ok]), D[ok])
    return out


def ratio(num, den, floor=1.0):
    return np.where(den >= floor, num / np.where(den > 0, den, 1.0), 1.0)


def boot_tot(num, den, boot=None):
    boot = boot_idx(num.shape[0]) if boot is None else boot
    a = num.reshape(num.shape[0], -1).sum(1); b = den.reshape(den.shape[0], -1).sum(1)
    return a.sum() / b.sum() - 1, (a[boot].sum(1) / b[boot].sum(1) - 1).std()


def cell_sums(pid, theta, qp, dvals, extra=None, qnodes=None, nt=324):
    """Per tile: S[t, cell, e] = sum of tent weights x d_h(e); W[t, cell] = sum of tent weights.  dvals (n, NE);
    extra = list of (pid, theta, qp, d (n, NE), with_weight) groups added to S (and to W when with_weight)."""
    NCELL = B.TH_K.size * (SC.QDK.size if qnodes is None else len(qnodes))
    S = np.zeros((nt * NCELL, NE)); W = np.zeros(nt * NCELL)
    groups = [(pid, theta, qp, dvals, True)] + (extra or [])
    for p_, th_, q_, d_, ww in groups:
        if p_.size == 0:
            continue
        idx, w = SC.cell_weights(th_, q_, qnodes)
        flat = (p_.astype(np.int64)[:, None] * NCELL + idx).ravel()
        wf = w.ravel()
        if ww:
            W += np.bincount(flat, weights=wf, minlength=nt * NCELL)
        if d_ is not None:
            nz = np.flatnonzero(np.any(d_ != 0, 1))
            if nz.size:
                fl = (p_[nz].astype(np.int64)[:, None] * NCELL + idx[nz]).ravel()
                for e in range(NE):
                    S[:, e] += np.bincount(fl, weights=(w[nz] * d_[nz, e][:, None]).ravel(), minlength=nt * NCELL)
    return S.reshape(nt, NCELL, NE).astype(np.float32), W.reshape(nt, NCELL).astype(np.float32)


def low_halos(mode, variant):
    """Tent weights per tile and cell of the halos with q_true < 1.5 (outside the b02 table; only their weights matter):
    (324, NCELL), accumulated tile by tile."""
    V = np.load(os.path.join(B.OUT, f"halos_{variant}_{mode}.npz"))
    Wl = np.zeros((324, NCELL), np.float32)
    for p in range(324):
        with np.load(os.path.join(B.D3, "truth", f"truth_{p:03d}.npz")) as t:
            qt = t["q_" + mode]; m = qt < 1.5
            th = t["theta500"][m].astype(float); Y = t["Y500"][m].astype(float)
        idx, w = SC.cell_weights(th, Y / B.sigma_finder(V["sig1"][p], mode, th))
        Wl[p] = np.bincount(idx.ravel(), weights=w.ravel(), minlength=NCELL)
    return Wl


def nr_cells(mode, variant, V):
    """The 'no real cluster' detections: their primary's theta500 and q' (from the truth tables), d(e) = 1[q > e]."""
    pid, hidx, q = V["nr_pid"], V["nr_hidx"], V["nr_q"].astype(float)
    th = np.zeros(q.size); qp = np.zeros(q.size)
    for p in np.unique(pid):
        m = pid == p
        with np.load(os.path.join(B.D3, "truth", f"truth_{p:03d}.npz")) as t:
            th[m] = t["theta500"][hidx[m]].astype(float); Y = t["Y500"][hidx[m]].astype(float)
        qp[m] = Y / B.sigma_finder(V["sig1"][p], mode, th[m])
    return pid, th, qp, (q[:, None] > B.EDGES[None, :]).astype(float)


def step3a_selection(mode, theta, qp, rinj15):
    """t05's p_sel with the pooled (t03 + a02) fits: q_inj = q' r_inj(theta), 0 below q_inj = 2 r_inj."""
    SA = np.load(os.path.join(B.D3, "selection_arbiter.npz"))
    fits, TH = SA[mode + "_fit"], SA["theta_inj"]
    ri = np.interp(np.log(theta), np.log(F.THETAS15), rinj15)
    q = qp * ri
    P = np.array([ndtr((np.sqrt(np.maximum(q * q + b, 0.0)) - 5.0 - mu) / np.exp(ls)) for b, mu, ls in fits])
    x = np.clip(np.log(theta), np.log(TH[0]), np.log(TH[-1]))
    i = np.clip(np.searchsorted(np.log(TH), x) - 1, 0, len(TH) - 2)
    w = (x - np.log(TH[i])) / np.log(TH[i + 1] / TH[i])
    n = np.arange(q.size)
    return np.where(q < 2.0 * ri, 0.0, (1 - w) * P[i, n] + w * P[i + 1, n])


def fmt_z(vals):
    return ", ".join(f"{100 * a:+.1f}+-{100 * b:.1f}" for a, b in vals)


def new_tiles(mode):
    """The new tiles (b14 truth, b15 finder outputs): per-halo arrays with tile index 324, 325, ... (sorted pids)."""
    fs = sorted(f for f in os.listdir(os.path.join(B.OUT, "newtiles")) if f.endswith(".npz")) if os.path.isdir(os.path.join(B.OUT, "newtiles")) else []
    A = {k: [] for k in ("t", "z", "theta500", "qp", "qobs_none", "qobs_closest1", "qobs_fof10", "qiso", "Wlow",
                         "nt", "nz", "nth", "nqp", "nq")}
    pids = []
    for k, f in enumerate(fs):
        pid = int(f[2:6]); pids.append(pid); t = 324 + k
        with np.load(os.path.join(B.OUT, "newtiles", f)) as d, np.load(os.path.join(B.OUT, "truth_new", f"truth_{pid:04d}.npz")) as tr:
            z, th, Y, qt = tr["z"].astype(float), tr["theta500"].astype(float), tr["Y500"].astype(float), tr["q_" + mode].astype(float)
            rows = d[mode + "_rows"]; sig = d[mode + "_sig1"]
            A["t"].append(np.full(rows.size, t)); A["z"].append(z[rows]); A["theta500"].append(th[rows]); A["qp"].append(d[mode + "_qp"].astype(float))
            for m in B.MERGES:
                A["qobs_" + m].append(d[f"{mode}_qobs_{m}"].astype(float))
            A["qiso"].append(d[mode + "_qiso"].astype(float))
            lo = np.flatnonzero(qt < 1.5)
            idx, w = SC.cell_weights(th[lo], Y[lo] / B.sigma_finder(sig, mode, th[lo]))
            A["Wlow"].append(np.bincount(idx.ravel(), weights=w.ravel(), minlength=NCELL).astype(np.float32)[None, :])
            nh = d[mode + "_nr_hidx"]
            A["nt"].append(np.full(nh.size, t)); A["nz"].append(z[nh]); A["nth"].append(th[nh])
            A["nqp"].append(Y[nh] / B.sigma_finder(sig, mode, th[nh])); A["nq"].append(d[mode + "_nr_q"].astype(float))
    if not pids:
        return 0, None
    return len(pids), {k: np.concatenate(v) for k, v in A.items()}


if __name__ == "__main__":
    B.threads(4)
    t00 = time.time()
    for mode in B.MODES:
        C = {k: v for k, v in np.load(os.path.join(B.OUT, f"halos_{mode}.npz")).items()}
        nrow = C["pid"].size
        nnew, NW = new_tiles(mode)
        cache, mcache = {}, {}
        print(f"==================== {mode}: {nrow} halos (q_true >= 1.5) in the 324 Q2 tiles; new tiles (b14 / b15) for the "
              f"production setups: {nnew} ({0 if NW is None else NW['t'].size} halos); {NBOOT} tile-bootstrap resamples; "
              f"pooled-correction cells {B.TH_K.size} theta x {SC.QDK.size} q' nodes", flush=True)
        for setup, (variant, merge, keepnr) in SETUPS.items():
            t0 = time.time()
            if variant not in cache:
                V = {k: v for k, v in np.load(os.path.join(B.OUT, f"halos_{variant}_{mode}.npz")).items()}
                Kern = B.Kernel.load(os.path.join(B.OUT, f"kernel_{variant}_{mode}.npz"))
                qp = V["qp"].astype(float)
                qi, seen = iso_qobs(variant, mode, nrow)
                miss = int(((~seen) & (qp >= 1.9)).sum())
                cache[variant] = (V, Kern, qi, miss)
            V, Kern, qi0, miss = cache[variant]
            ext = variant == "prod" and nnew > 0 and setup != "prod_none_q2"
            nt = 324 + (nnew if ext else 0)
            pid = C["pid"].astype(np.int64); z = C["z"].astype(float); th = C["theta500"].astype(float)
            qp = V["qp"].astype(float); qo = V[f"qobs_{merge}"].astype(float); qi = qi0
            if ext:
                pid = np.concatenate([pid, NW["t"]]); z = np.concatenate([z, NW["z"]]); th = np.concatenate([th, NW["theta500"]])
                qp = np.concatenate([qp, NW["qp"]]); qo = np.concatenate([qo, NW["qobs_" + merge]]); qi = np.concatenate([qi, NW["qiso"]])
            m = qp >= B.CRED_MIN
            Kh = np.zeros((qp.size, NE)); Kh[m] = Kern(th[m], qp[m])
            out = dict(setup=setup, variant=variant, merge=merge, keepnr=keepnr, ntiles=nt)
            # ---- pooled correction: cell sums (independent of q_cut)
            d = (qo[:, None] > B.EDGES[None, :]).astype(np.float32) - (qi[:, None] > B.EDGES[None, :]).astype(np.float32)
            extra = []
            nr_t = nr_z = nr_q = None
            if keepnr:
                Wlow = low_halos(mode, variant)
                npid, nth, nqp, nd = nr_cells(mode, variant, V)
                nr_t, nr_z, nr_q = V["nr_pid"].astype(np.int64), V["nr_z"].astype(float), V["nr_q"].astype(float)
                if ext:
                    Wlow = np.concatenate([Wlow, NW["Wlow"]])
                    npid = np.concatenate([npid, NW["nt"]]); nth = np.concatenate([nth, NW["nth"]]); nqp = np.concatenate([nqp, NW["nqp"]])
                    nd = np.concatenate([nd, (NW["nq"][:, None] > B.EDGES[None, :]).astype(float)])
                    nr_t = np.concatenate([nr_t, NW["nt"]]); nr_z = np.concatenate([nr_z, NW["nz"]]); nr_q = np.concatenate([nr_q, NW["nq"]])
                extra.append((npid, nth, nqp, nd, False))
            St, Wt = cell_sums(pid, th, qp, d, extra, nt=nt)
            if keepnr:
                Wt += Wlow
            Wtot = Wt.sum(0, dtype=np.float64)
            dK = np.where(Wtot[:, None] > 0, St.sum(0, dtype=np.float64) / np.where(Wtot > 0, Wtot, 1.0)[:, None], 0.0)
            out["dK"] = dK.reshape(B.TH_K.size, SC.QDK.size, NE); out["Wcell"] = Wtot.reshape(B.TH_K.size, SC.QDK.size)
            if setup == "prod_none":                          # cell-grid check: every other q' node
                Q2 = SC.QDK[::2]
                St2, Wt2 = cell_sums(pid, th, qp, d, None, Q2, nt=nt)
                W2 = Wt2.sum(0, dtype=np.float64)
                dK2 = np.where(W2[:, None] > 0, St2.sum(0, dtype=np.float64) / np.where(W2 > 0, W2, 1.0)[:, None], 0.0).reshape(B.TH_K.size, Q2.size, NE)
                del St2, Wt2
            print(f"  [{setup}] tiles {nt}; halos with q' >= 1.9 without an isolated copy (Q2 tiles): {miss}; halos whose blended and "
                  f"isolated outcome differ at e = 5: {int(np.abs(d[:, np.argmin(np.abs(B.EDGES - 5))]).sum())} of "
                  f"{int(((qo > 5) | (qi > 5)).sum())} credited in either", flush=True)
            per = {}
            for qc in SC.QCUTS:
                zed, qed = CL.default_bins(qc)
                sel = [int(np.argmin(np.abs(B.EDGES - e))) for e in qed[:-1]]
                assert np.allclose(B.EDGES[sel], qed[:-1])
                Nd = per_tile_hist(pid, z, qo, zed, qed, nt=nt)
                if keepnr:
                    Nd = Nd + per_tile_hist(nr_t, nr_z, nr_q, zed, qed, nt=nt)
                Ne = per_tile_exp(pid, z, Kh, sel, zed, nt=nt)
                Ni = per_tile_hist(pid, z, qi, zed, qed, nt=nt)
                if (variant, qc) not in mcache:
                    M = SC.model(variant, mode, qc)
                    mcache[(variant, qc)] = (M.counts(SC.FID), SC.cell_integrals(M, SC.FID))
                mu, Icell = mcache[(variant, qc)]
                per[qc] = (zed, qed, sel, Nd, Ne, Ni, mu, Icell)
            # bootstrap of R_s: one pass over the resamples for both q_cuts
            WBt = boot_w(nt)
            Sflat = St.reshape(nt, -1)
            RsB = {qc: np.empty((NBOOT, 16, 9)) for qc in SC.QCUTS}
            for a in range(0, NBOOT, 200):
                wb = WBt[a:a + 200]
                Sb = (wb @ Sflat).reshape(-1, NCELL, NE); Wb = wb @ Wt
                dKb = np.where(Wb[..., None] > 0, Sb / np.where(Wb > 0, Wb, 1.0)[..., None], 0.0).reshape(-1, B.TH_K.size, SC.QDK.size, NE)
                for qc in SC.QCUTS:
                    zed, qed, sel, Nd, Ne, Ni, mu, Icell = per[qc]
                    RsB[qc][a:a + 200] = SC.rs_from_dk(Icell, mu, dKb, sel)
            del Sflat, St, Wt
            BOOTt = boot_idx(nt)
            for qc in SC.QCUTS:
                s = f"q{qc:.0f}"
                zed, qed, sel, Nd, Ne, Ni, mu, Icell = per[qc]
                Rd = ratio(Nd.sum(0), Ne.sum(0)); Rp = ratio(Nd.sum(0), Ni.sum(0))
                RdB = np.array([ratio(Nd[b].sum(0), Ne[b].sum(0)) for b in BOOTt])
                RpB = np.array([ratio(Nd[b].sum(0), Ni[b].sum(0)) for b in BOOTt])
                Rs = SC.rs_from_dk(Icell, mu, out["dK"], sel)
                if setup == "prod_none":
                    Rs2 = SC.rs_from_dk(SC.cell_integrals(SC.model(variant, mode, qc), SC.FID, SC.QDK[::2]), mu, dK2, sel)
                    out[f"Rs_coarse_{s}"] = Rs2
                    print(f"  [cell-grid check] q > {qc:.0f}: R_s with every other q' node: model-weighted total {100 * (np.sum(mu * Rs2) / mu.sum() - 1):+.3f}% "
                          f"vs {100 * (np.sum(mu * Rs) / mu.sum() - 1):+.3f}%; max |dR| over bins with mu > 10: {np.abs(Rs2 - Rs)[mu > 10].max():.4f}", flush=True)
                RsBq = RsB[qc]
                out.update({f"Ndet_{s}": Nd.sum(0), f"Nexp_{s}": Ne.sum(0), f"Niso_{s}": Ni.sum(0), f"mu_{s}": mu,
                            f"R_{s}": Rd, f"Rp_{s}": Rp, f"Rs_{s}": Rs,
                            f"Rboot_{s}": RdB.astype(np.float32), f"Rpboot_{s}": RpB.astype(np.float32), f"Rsboot_{s}": RsBq.astype(np.float32),
                            f"R_err_{s}": RdB.std(0), f"Rp_err_{s}": RpB.std(0), f"Rs_err_{s}": RsBq.std(0),
                            f"Ndet_tile_{s}": Nd.sum((1, 2)), f"Nexp_tile_{s}": Ne.sum((1, 2)), f"Niso_tile_{s}": Ni.sum((1, 2))})
                if qc == 5.0:
                    T, eT = boot_tot(Nd, Ne); Tp, eTp = boot_tot(Nd, Ni); Ti, eTi = boot_tot(Ni, Ne)
                    w_ = Ne.sum(0)
                    Ts = np.sum(w_ * (Rs - 1)) / w_.sum(); eTs = (np.sum(w_ * (RsBq - 1), (1, 2)) / w_.sum()).std()
                    Tm = np.sum(mu * (Rs - 1)) / mu.sum(); eTm = (np.sum(mu * (RsBq - 1), (1, 2)) / mu.sum()).std()
                    print(f"  [{setup}] q > 5: N_det {Nd.sum():.0f}, N_exp {Ne.sum():.1f}, N_iso {Ni.sum():.0f}; count change "
                          f"N_det/N_exp - 1 = {100 * T:+.2f} +- {100 * eT:.2f}%; N_det/N_iso - 1 = {100 * Tp:+.2f} +- {100 * eTp:.2f}%; "
                          f"R_s (tile-halo weights) {100 * Ts:+.2f} +- {100 * eTs:.2f}%, R_s (full-sky model weights) {100 * Tm:+.2f} +- "
                          f"{100 * eTm:.2f}%; check N_iso/N_exp - 1 = {100 * Ti:+.2f} +- {100 * eTi:.2f}%", flush=True)
                    zr, zp, zs = [], [], []
                    for a, b in zip(ZC[:-1], ZC[1:]):
                        kz = (zed[:-1] >= a - 1e-9) & (zed[1:] <= b + 1e-9)
                        zr.append(boot_tot(Nd[:, kz], Ne[:, kz])); zp.append(boot_tot(Nd[:, kz], Ni[:, kz]))
                        wz = w_[kz]
                        zs.append((np.sum(wz * (Rs[kz] - 1)) / wz.sum(), (np.sum(wz * (RsBq[:, kz] - 1), (1, 2)) / wz.sum()).std()))
                    print("     coarse z bins " + ", ".join(f"{a}-{b}" for a, b in zip(ZC[:-1], ZC[1:])) + " [%]:", flush=True)
                    print(f"        R  : {fmt_z(zr)}\n        R_p: {fmt_z(zp)}\n        R_s: {fmt_z(zs)}", flush=True)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        qr = [boot_tot(Nd[:, :, k], Ne[:, :, k]) for k in range(qed.size - 1)]
                    qs = [(np.sum(w_[:, k] * (Rs[:, k] - 1)) / max(w_[:, k].sum(), 1e-300), (np.sum(w_[:, k] * (RsBq[:, :, k] - 1), 1) / max(w_[:, k].sum(), 1e-300)).std())
                          for k in range(qed.size - 1)]
                    print("     q bins " + ", ".join(f"{a:.3g}-{b:.3g}" for a, b in zip(qed[:-1], qed[1:])) + f" [%]:\n        R  : {fmt_z(qr)}\n"
                          f"        R_s: {fmt_z(qs)}", flush=True)
                    ok = w_ >= 10
                    for lab, A_, AB in (("R", Rd, RdB), ("R_p", Rp, RpB)):
                        dd = A_ - Rs; ddB = AB - RsBq
                        chi = np.sum((dd[ok] / ddB.std(0)[ok]) ** 2)
                        print(f"     consistency {lab} vs R_s over {int(ok.sum())} bins with N_exp >= 10: chi^2 (diagonal, bootstrap errors of the "
                              f"difference) = {chi:.1f}", flush=True)
                    print(f"     median bootstrap error per bin (N_exp >= 10): R {np.median(RdB.std(0)[ok]):.4f}, R_p {np.median(RpB.std(0)[ok]):.4f}, "
                          f"R_s {np.median(RsBq.std(0)[ok]):.4f}; bins with N_exp < 1: {int((w_ < 1).sum())} holding {Nd.sum(0)[w_ < 1].sum():.0f} "
                          f"credited halos", flush=True)
            np.savez(os.path.join(B.OUT, f"datavec_{setup}_{mode}.npz"), **out)
            print(f"     [{time.time() - t0:.0f} s]", flush=True)
        # ---- step 3a's preview recomputed on the same halos (production finder and SZiFi defaults), Q2 tiles
        SA = np.load(os.path.join(B.D3, "selection_arbiter.npz"))
        rinj = SA[mode + "_rinj"]
        e5 = int(np.argmin(np.abs(B.EDGES - 5.0)))
        pid, th = C["pid"].astype(np.int64), C["theta500"].astype(float)
        for variant, tiles in (("prod", np.arange(324)), ("szifi3000", np.arange(100)), ("szifi3000", np.arange(324)),
                               ("szifi12000", np.arange(324))):
            V, Kern, qi, miss = cache[variant]
            qp = V["qp"].astype(float)
            m = np.isin(pid, tiles) & (qp >= 1.0)
            p3a = np.zeros(nrow); p3a[m] = step3a_selection(mode, th[m], qp[m], rinj)
            Kh5 = np.zeros(nrow); mm = m & (qp >= B.CRED_MIN); Kh5[mm] = Kern(th[mm], qp[mm], e5)[:, 0]
            a = np.bincount(pid[np.isin(pid, tiles) & (V["qobs_none"] > 5)], minlength=324).astype(float)
            N3 = np.bincount(pid[m], weights=p3a[m], minlength=324)
            Nk = np.bincount(pid[m], weights=Kh5[m], minlength=324)
            sl = np.zeros(324, bool); sl[tiles] = True
            bt = BOOT if tiles.size == 324 else tiles[BOOT100]
            T3 = a[sl].sum() / N3[sl].sum() - 1; Tk = a[sl].sum() / Nk[sl].sum() - 1
            e3 = (a[bt].sum(1) / N3[bt].sum(1) - 1).std(); ek = (a[bt].sum(1) / Nk[bt].sum(1) - 1).std()
            print(f"  step 3a's preview recomputed, {variant}, tiles {tiles[0]}-{tiles[-1]}: N_det(q > 5) {a[sl].sum():.0f}; with step 3a's "
                  f"selection (production finder's injection fits) N_exp {N3[sl].sum():.1f} -> {100 * T3:+.2f} +- {100 * e3:.2f}%; with this "
                  f"variant's virtual kernel N_exp {Nk[sl].sum():.1f} -> {100 * Tk:+.2f} +- {100 * ek:.2f}% (kernel change "
                  f"{100 * (N3[sl].sum() / Nk[sl].sum() - 1):+.2f}% in N_exp)", flush=True)
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {B.peak_rss_gb():.2f} GB")
