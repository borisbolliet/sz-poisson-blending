"""Step 3b / b03: the analyst's no-blending selection, measured with virtual isolated clusters on noise-only tiles, and
its end-to-end validation with sparse injections.

Per noise-only tile pid = 0..323 (the realistic tile pid without halos: CMB seed 300000 + pid, noise seed 400000 + pid),
noise level and finder variant (s3blib.VARIANTS), the variant's finder runs on the tile (both iterations: covariance,
q > 4 masked, covariance re-estimated, filter rerun); its final filter gives the S/N cube qn (15 scales) and the response
tables.  Then (s3blib.virtual_credit: own response added in a window, the finder's peak rules, credited if own S/N >= 2,
q_obs = the highest credited peak):
  (K) kernel clusters: NK per tile at uniform random positions, theta500 on the 25 nodes TH_K (0.238'-61'), q' on the
      nodes Q_NODES (1-10 step 0.125 below 10: 70%; 10.5-30 step 0.5: 15%) or log-uniform in 25-400 (15%);
      Y500 = q' sigma_finder(theta500) of this tile (the variant's units).  The same clusters for every variant.
  (iso) every halo of the realistic tile pid with q' >= 1.9 (this variant's realistic q', b02) inserted alone at its true
      position with Y' = q' sigma_finder,noise-only(theta500): the same S/N in finder units on the same CMB + noise
      realisation, without neighbours (paired estimator of the blending ratio).
  (wc) window check (pids < 8): the kernel clusters again with 2x the window.
  (acc) accuracy check (pid 0): for 2 clusters per theta node, the virtual own S/N in a +-2W window against the exact
      filtered response (one cluster painted alone with szsim.finder.paint_y, filtered with the tile's filter by FFT),
      with and without the lattice harmonics.
  (sp) production variant only: sparse injections, end to end.  NSP sets of 10 clusters per tile (>= 100' apart: 4 x 3
      cells of 150' x 200', 10 random cells, +-25' jitter; theta500 a random TH_K node; q'_target uniform in 1.8-9 (85%) or
      9-30 (15%); Y500 = q'_target sigma_finder,noise-only), painted into this tile's CMB + noise, the production finder
      rerun from scratch; a final detection is credited to its nearest injected cluster when that cluster set's filtered
      tSZ there is >= 2 sigma (t03 / a02 rule); q_obs per cluster for merge none, closest 1', fof 10'.  The same clusters
      are also inserted virtually into the noise-only cube (paired comparison).  Layouts do not depend on the noise level.
Window half width: max(6', 1.25 r_0.1(theta) + 2') with r_0.1 the radius beyond which every scale's own response is below
10% of the cluster's peak S/N (measured on tile 0, production, baseline; 2.0' at 0.24' ... 7.9' at 61').
Outputs: data_real/step3b/virtual/VARIANT/v_MODE_PID.npz.  Usage: python b03_virtual.py VARIANT PID0 PID1 NPROC
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
from multiprocessing import Pool                   # noqa: E402
F, S3 = B.F, B.S3

NK = 4000
NSP, NSP_CL = 7, 10
WCHECK = range(8)
ACC_PID = 0
R01 = np.array([2.00, 2.00, 2.00, 2.05, 2.10, 2.15, 2.20, 2.25, 2.30, 2.40, 2.50, 2.60, 2.75, 2.95, 3.15, 3.40, 3.75, 4.10,
                4.60, 5.15, 5.75, 6.30, 6.85, 7.40, 7.85])          # r_0.1 per TH_K node (x_explore, tile 0 prod baseline)
RW = 1.25 * R01 + 2.0
_H = {}


def halos(mode, variant):
    key = (mode, variant)
    if key not in _H:
        C = dict(np.load(os.path.join(B.OUT, f"halos_{mode}.npz")))
        V = np.load(os.path.join(B.OUT, f"halos_{variant}_{mode}.npz"))
        C["qp"] = V["qp"]
        C["start"] = np.searchsorted(C["pid"], np.arange(325))
        _H[key] = C
    return _H[key]


def kernel_clusters(pid, mode_idx):
    rng = np.random.default_rng(9_000_000 + 1000 * mode_idx + pid)
    thi = rng.integers(0, B.TH_K.size, NK)
    u = rng.random(NK)
    lo = np.concatenate([B.QN_LO, B.QN_MID])
    q = np.where(u < 0.70, lo[rng.integers(0, lo.size, NK)],
                 np.where(u < 0.85, B.QN_HI[rng.integers(0, B.QN_HI.size, NK)],
                          np.exp(rng.uniform(np.log(B.Q_LOGU[0]), np.log(B.Q_LOGU[1]), NK))))
    y = rng.uniform(0, F.LBOX, NK); x = rng.uniform(0, F.LBOX, NK)
    return thi, q, y, x


def sparse_layout(pid, s):
    rng = np.random.default_rng(8_000_000 + 10 * pid + s)
    cells = rng.choice(12, NSP_CL, replace=False)
    cy, cx = cells // 4, cells % 4
    y = ((cy + 0.5) * 200.0 + rng.uniform(-25, 25, NSP_CL)) % F.LBOX
    x = ((cx + 0.5) * 150.0 + rng.uniform(-25, 25, NSP_CL)) % F.LBOX
    thi = rng.integers(0, B.TH_K.size, NSP_CL)
    q = np.where(rng.random(NSP_CL) < 0.85, rng.uniform(1.8, 9.0, NSP_CL), rng.uniform(9.0, 30.0, NSP_CL))
    return y, x, thi, q


def credit_injected(filt, det, Xt, y, x):
    """Per injected cluster: highest credited q (nearest injected cluster, filtered tSZ >= 2 sigma) for each merge rule."""
    nd = det["q"].size
    out = {m: np.full(y.size, -np.inf) for m in B.MERGES}
    if nd == 0:
        return out, 0
    dy = (det["y"][:, None] - y[None, :] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
    dx = (det["x"][:, None] - x[None, :] + F.LBOX / 2) % F.LBOX - F.LBOX / 2
    dist = np.hypot(dy, dx)
    near = dist.argmin(1)
    close = dist[np.arange(nd), near] < 30.0
    s_tsz = np.zeros(nd)
    cx_t = filt.CX(Xt)
    for j in np.unique(det["j"][close]):
        m = close & (det["j"] == j)
        s_tsz[m] = (filt.yhat_map(cx_t, j) / F.AR ** 2)[det["iy"][m], det["ix"][m]]
    cred = s_tsz / filt.sigma_arcmin2[det["j"]] >= B.CRED_MIN
    for mname in B.MERGES:
        k = F.apply_merge(det, mname, 0.0 if mname == "none" else 5.0)
        if mname == "none":
            k = np.arange(nd)
        kk = k[cred[k]]
        np.maximum.at(out[mname], near[kk], det["q"][kk])
    return out, int(((~cred) & (det["q"] > 5)).sum())


def accuracy_check(filt, sig, tabs, mode, rng, G):
    """Max |virtual own S/N - exact own S/N| / peak own S/N over a +-2W window and all scales, 2 clusters per node, for
    the angle-averaged tables alone (column 0) and with the lattice harmonics (column 1).  Exact: the cluster painted
    alone (szsim.finder.paint_y), filtered with the tile's final filter by FFT."""
    tables = B.basis()[1]
    Rn, Rf, A, targs = tabs
    va = B.vargs(Rn, Rf, A, targs)
    va0 = B.vargs(Rn, Rf, np.zeros_like(A), targs)
    errs = np.zeros((B.TH_K.size, 2, 2))
    for k, th in enumerate(B.TH_K):
        for rep in range(2):
            y0 = 300.0 + rng.uniform(-0.5, 0.5); x0 = 300.0 + rng.uniform(-0.5, 0.5)
            Y = 10.0 * B.sigma_finder(sig, mode, th)
            Xt = F.tsz_channels(F.paint_y([y0], [x0], [Y], [th], tables, G), G)
            cx = filt.CX(Xt)
            w = int(np.ceil(2 * max(6.0, RW[k]) / F.PIX))
            cy0, cx0 = int(np.rint(y0 / F.PIX)), int(np.rint(x0 / F.PIX))
            ex = np.stack([(filt.yhat_map(cx, j) / F.AR ** 2 / sig[j])[cy0 - w:cy0 + w + 1, cx0 - w:cx0 + w + 1] for j in range(15)])
            pk = np.abs(ex).max()
            for col, v in enumerate((va0, va)):
                tb = B.own_grid(Y, y0, x0, np.log(th), cy0, cx0, w, sig, *v, F.PIX)
                errs[k, rep, col] = np.abs(tb - ex).max() / pk
    return errs


def work(job):
    variant, pid, mode = job
    fn = os.path.join(B.OUT, "virtual", variant, f"v_{mode}_{pid:03d}.npz")
    if os.path.exists(fn):
        return job, 0.0, B.peak_rss_gb(), None
    t0 = time.time()
    G = F.grid()
    bas, tables = B.basis()
    mi = B.MODES.index(mode)
    its, X = B.noise_only_filter(pid, mode, variant, G)
    filt = its[-1]["filt"]; qn = its[-1]["q"]; sig = filt.sigma_arcmin2.copy()
    det0 = its[-1]["det"]
    tabs = B.filter_tables(filt, bas, G)
    t_find = time.time() - t0
    res = dict(pid=pid, mode=mode, variant=variant, sig=sig, r=B.r_of(sig, mode), n_it=len(its), fsky1=float(its[-1]["fsky"]))
    detn = dict(q=det0["q"], y=det0["y"], x=det0["x"])
    res["noise_n"] = np.array([[F.apply_merge(detn, m, qc).size for qc in (4.0, 4.5, 5.0)] for m in B.MERGES])
    # ---- (K) kernel clusters
    t1 = time.time()
    thi, q, y, x = kernel_clusters(pid, mi)
    th = B.TH_K[thi]
    Y = q * B.sigma_finder(sig, mode, th)
    W = B.window_pix(th, B.TH_K, RW)
    o = B.run_virtual(qn, sig, tabs, y, x, Y, th, W)
    res.update(k_thi=thi.astype(np.int8), k_q=q.astype(np.float32), k_qobs=o["q"].astype(np.float32), k_j=o["j"].astype(np.int8),
               k_dy=o["dy"].astype(np.int8), k_dx=o["dx"].astype(np.int8), k_own=o["own"].astype(np.float32),
               k_nc=o["nc"].astype(np.int16), k_q0=o["q0"].astype(np.float32))
    t_k = time.time() - t1
    if pid in WCHECK:
        o2 = B.run_virtual(qn, sig, tabs, y, x, Y, th, 2 * W)
        res["wc_qobs"] = o2["q"].astype(np.float32)
    # ---- (iso) the realistic tile's halos, alone, at their positions, same q'
    t1 = time.time()
    H = halos(mode, variant)
    a, b = H["start"][pid], H["start"][pid + 1]
    rows = np.arange(a, b)[H["qp"][a:b] >= 1.9]
    thh = H["theta500"][rows].astype(float)
    Yh = H["qp"][rows].astype(float) * B.sigma_finder(sig, mode, thh)
    oi = B.run_virtual(qn, sig, tabs, H["y"][rows].astype(float), H["x"][rows].astype(float), Yh, thh,
                       B.window_pix(thh, B.TH_K, RW))
    res.update(iso_row=rows.astype(np.int32), iso_qobs=oi["q"].astype(np.float32), iso_q0=oi["q0"].astype(np.float32),
               iso_j=oi["j"].astype(np.int8))
    t_iso = time.time() - t1
    # ---- (acc) accuracy of the virtual response
    if pid == ACC_PID:
        res["acc_err"] = accuracy_check(filt, sig, tabs, mode, np.random.default_rng(77 + mi), G)
    # ---- (sp) sparse injections, end to end (production only).  The virtual partners first, then the noise-only cube is
    #      released before the seven finder reruns (lower peak memory; the results do not depend on the order).
    t1 = time.time()
    if variant == "prod":
        lay = [sparse_layout(pid, s) for s in range(NSP)]
        vpart = []
        for ys, xs, ti_, qt in lay:
            ths = B.TH_K[ti_]
            Ys = qt * B.sigma_finder(sig, mode, ths)
            vpart.append((ths, Ys, B.run_virtual(qn, sig, tabs, ys, xs, Ys, ths, B.window_pix(ths, B.TH_K, RW))))
        del qn, its, tabs, filt
        its = None
        rows_sp = {k: [] for k in ("set", "thi", "y", "x", "Y", "qt", "qp", "qv", "q_none", "q_closest1", "q_fof10", "qv0")}
        sig_sp, nspur = [], 0
        for s, ((ys, xs, ti_, qt), (ths, Ys, ov)) in enumerate(zip(lay, vpart)):
            Xt = F.tsz_channels(F.paint_y(ys, xs, Ys, ths, tables, G), G)
            its_s = F.run_finder(X + Xt, G)
            fs, ds = its_s[-1]["filt"], its_s[-1]["det"]
            qo, nsp_ = credit_injected(fs, ds, Xt, ys, xs)
            nspur += nsp_
            ss = fs.sigma_arcmin2.copy()
            sig_sp.append(ss)
            rows_sp["set"].append(np.full(NSP_CL, s)); rows_sp["thi"].append(ti_); rows_sp["y"].append(ys); rows_sp["x"].append(xs)
            rows_sp["Y"].append(Ys); rows_sp["qt"].append(qt); rows_sp["qp"].append(Ys / B.sigma_finder(ss, mode, ths))
            rows_sp["qv"].append(ov["q"]); rows_sp["qv0"].append(ov["q0"])
            for m in B.MERGES:
                rows_sp["q_" + m].append(qo[m])
            del its_s, fs, Xt
        res.update({"sp_" + k: np.concatenate(v) for k, v in rows_sp.items()})
        res["sp_sig"] = np.array(sig_sp); res["sp_nspur"] = nspur
    t_sp = time.time() - t1
    res["timing"] = np.array([t_find, t_k, t_iso, t_sp])
    res["t"] = time.time() - t0
    np.savez(fn, **res)
    return job, res["t"], B.peak_rss_gb(), (t_find, t_k, t_iso, t_sp, rows.size)


if __name__ == "__main__":
    variant, P0, P1, NPROC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(os.path.join(B.OUT, "virtual", variant), exist_ok=True)
    jobs = [(variant, pid, mode) for pid in range(P0, P1) for mode in B.MODES]
    t00 = time.time()
    print(f"b03: variant {variant}, noise-only tiles {P0}..{P1 - 1} x {B.MODES}, {NPROC} processes; {NK} kernel clusters per "
          f"tile{'; ' + str(NSP) + ' sparse injection sets of ' + str(NSP_CL) if variant == 'prod' else ''}", flush=True)
    rss_max = 0.0
    with Pool(NPROC, maxtasksperchild=40) as pool:
        for i, (job, tt, rss, tim) in enumerate(pool.imap_unordered(work, jobs)):
            rss_max = max(rss_max, rss)
            if tim is None:
                continue
            if i % 20 == 0 or i == len(jobs) - 1:
                print(f"  {job[1]:3d} {job[2]:8s}: {tt:5.1f} s (finder {tim[0]:.1f}, kernel {tim[1]:.1f}, iso {tim[2]:.1f} "
                      f"[{tim[4]} halos], sparse {tim[3]:.1f}); worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]", flush=True)
    print(f"b03 done: {len(jobs)} jobs in {time.time() - t00:.0f} s wall; largest worker peak RSS {rss_max:.2f} GB", flush=True)
