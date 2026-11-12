"""f03 (step-3b verification, lens 'fits', task b): the no-blending kernel K(theta500, q'; e) re-measured at selected
(theta500, q') nodes with MY OWN virtual-cluster code on the build's noise-only tiles.

Noise-only tile pid (the realistic tile without halos): CMB seed 300000 + pid, noise seed 400000 + pid (szsim.finder
sim_cmb / sim_noise); the variant's finder (szsim.finder.run_finder, both iterations; production 'whitened' or SZiFi
defaults 'szifi' with l_sep 3000, blend, 100 inpainting sweeps) gives the final filter and the noise-only S/N cube qn.
Checked: my sigma_finder of the tile against the build's stored 'sig' of that tile (virtual/VARIANT/v_MODE_PID.npz).
Clusters (mine): Y500 = q' sigma_finder(theta500), sigma_finder(theta) = r(theta) sigma_Y(theta), r = the tile's
sigma_finder / sigma_Y at the 15 scales, linear in ln theta (constant beyond).
OWN RESPONSE (not the build's response tables): ONE cluster at a random sub-pixel offset from the tile centre is painted
with the finder's own painter (szsim.finder.paint_y, the template the realistic halos were painted with), turned into
the six channels (tsz_channels) and filtered with the tile's FINAL filter by FFT (filt.CX, filt.yhat_map / sigma): the
exact own S/N cube of an isolated cluster at all 15 scales (a batch of several clusters per paint was rejected: the
filtered response has a tile-wide floor of 1e-3 - 9e-3 of the peak, so batch members leak into each other).  The
window of that response is placed at NPOS random pixel positions of the tile (the filter is translation invariant on
the periodic tile: checked against a cluster painted at the shifted position).
OWN PEAK FINDING (my numba code, the rules of the finder's documentation): in a window of half width
max(10', 2 theta500 + 6') around the cluster, on tot = qn + own: v > 4, v strictly above its 4 neighbours along the
map axes, scale rules q_j > q_{j-1} and q_j >= q_{j+1} (q_{-1} = 0; only q_14 > q_13 at the last scale).
OWN CREDIT RULE: a peak is credited when the cluster's own S/N there is >= 2; q_obs = the highest credited peak.
K_mine(e) = fraction of clusters with q_obs > e, compared with the build's table (kernel_VARIANT_MODE.npz: K (PAVA),
Kraw and n at discrete nodes; the interpolated table at q' = 33, which comes from the build's bright-end model).
Usage: f03_kernel.py VARIANT NPROC     Output: out/f03/k_VARIANT_MODE_PID.npz, log.
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import numba as nb

ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, ROOT)
OUTD = os.path.join(HERE, "out", "f03")
D3B = os.path.join(ROOT, "data_real", "step3b")
VARIANTS = {"prod": ("whitened", {}), "szifi3000": ("szifi", dict(lsep=3000.0, blend=True, n_inpaint=100))}
TH_K = 0.3 * 2.0 ** ((np.arange(25) - 1) / 3.0)                 # the build's theta nodes (checked against the file)
# (theta node index, q') per variant
NODES = {"prod": [(0, 5.0), (6, 5.0), (9, 4.5), (12, 6.0), (15, 5.5), (18, 6.5), (21, 8.0), (12, 33.0), (9, 1.75)],
         "szifi3000": [(6, 5.0), (12, 6.0), (18, 6.5)]}
TILES = {"prod": np.arange(5, 324, 20)[:16], "szifi3000": np.arange(7, 324, 27)[:12]}
SEED_CMB, SEED_NOISE = 300_000, 400_000
QTH, CMIN = 4.0, 2.0


@nb.njit(cache=True)
def credit_window(qn, ow, cy, cx, w, qth, cmin):
    """Peaks of tot = qn + own in the window (cy +- w, cx +- w) interior; qn periodic (nsc, N, N), ow = the cluster's own
    S/N on the (nsc, 2w+3, 2w+3) window centred on its pixel.  Returns (q_obs or -inf, own at that peak, scale, dy, dx,
    number of credited peaks, number of peaks)."""
    nsc, n0, n1 = qn.shape
    L = 2 * w + 3
    tot = np.empty((nsc, L, L))
    for a in range(L):
        iy = (cy + a - w - 1) % n0
        for b in range(L):
            ix = (cx + b - w - 1) % n1
            for j in range(nsc):
                tot[j, a, b] = qn[j, iy, ix] + ow[j, a, b]
    best = -np.inf; bo = 0.0; bj = -1; by = 0; bx = 0; nc = 0; npk = 0
    for j in range(nsc):
        for a in range(1, L - 1):
            for b in range(1, L - 1):
                v = tot[j, a, b]
                if not (v > qth):
                    continue
                if not (v > tot[j, a - 1, b] and v > tot[j, a + 1, b] and v > tot[j, a, b - 1] and v > tot[j, a, b + 1]):
                    continue
                if j == 0:
                    ok = (v > 0.0) and (v >= tot[1, a, b])
                elif j == nsc - 1:
                    ok = v > tot[j - 1, a, b]
                else:
                    ok = (v > tot[j - 1, a, b]) and (v >= tot[j + 1, a, b])
                if not ok:
                    continue
                npk += 1
                if ow[j, a, b] >= cmin:
                    nc += 1
                    if v > best:
                        best = v; bo = ow[j, a, b]; bj = j; by = a - w - 1; bx = b - w - 1
    return best, bo, bj, by, bx, nc, npk


def sigma_Y(mode, theta, _c={}):
    if mode not in _c:
        from szsim import instrument as ins
        _c[mode] = ins.MMFModel(ins.Instrument.so_lat(mode, lmin=100.0, lmax=20000.0))
    return _c[mode].sigma_Y(np.asarray(theta, float))


THETAS15 = np.geomspace(0.5, 15.0, 15)


def own_cube(F, filt, G, tables, ys, xs, Ys, ths):
    Xt = F.tsz_channels(F.paint_y(ys, xs, Ys, ths, tables, G), G)
    cx = filt.CX(Xt)
    del Xt
    return np.stack([filt.yhat_map(cx, j) / filt.sigma[j] for j in range(len(filt.thetas))])


NOFF, NPOS = 8, 12                                               # sub-pixel offsets x positions per node per tile


def work(job):
    variant, mode, pid = job
    from szsim import finder as F
    fn = os.path.join(OUTD, f"k_{variant}_{mode}_{pid:03d}.npz")
    if os.path.exists(fn):
        return job, None
    t0 = time.time()
    G = F.grid()
    tables = F.build_paint_tables()
    X = F.sim_cmb(SEED_CMB + pid, G) + F.sim_noise(SEED_NOISE + pid, mode, G)
    meth, kw = VARIANTS[variant]
    its = F.run_finder(X, G, keep_cube=True, mask_method=meth, mask_kw=kw)
    filt, qn = its[-1]["filt"], its[-1]["q"]
    del X
    sig = filt.sigma_arcmin2.copy()
    t_find = time.time() - t0
    ref = np.load(os.path.join(D3B, "virtual", variant, f"v_{mode}_{pid:03d}.npz"))["sig"]
    r15 = sig / sigma_Y(mode, THETAS15)
    rng = np.random.default_rng(123_000 + 1000 * (mode == "goal") + pid + (500_000 if variant != "prod" else 0))
    rows = {k: [] for k in ("node", "thi", "q", "y", "x", "qobs", "own", "j", "dy", "dx", "nc", "npk", "own0")}
    shiftchk = []
    c0 = F.N // 2
    for kn, (thi, qp) in enumerate(NODES[variant]):
        th = TH_K[thi]
        sfin = np.interp(np.log(th), np.log(THETAS15), r15) * sigma_Y(mode, th)
        Y = qp * sfin
        w = int(np.ceil(max(10.0, 2.0 * th + 6.0) / F.PIX))
        for o in range(NOFF):
            dy, dx = rng.uniform(-0.25, 0.25, 2)                       # sub-pixel offset [arcmin] from pixel (c0, c0)
            y0, x0 = c0 * F.PIX + dy, c0 * F.PIX + dx
            own = own_cube(F, filt, G, tables, np.array([y0]), np.array([x0]), np.array([Y]), np.array([th]))
            win = np.ascontiguousarray(own[:, c0 - w - 1:c0 + w + 2, c0 - w - 1:c0 + w + 2])
            o0 = float(own[:, c0, c0].max())
            if o == 0 and pid == TILES[variant][0]:
                # translation check: the same cluster painted at a shifted position
                sy, sx = 137, 811
                o2 = own_cube(F, filt, G, tables, np.array([y0 + sy * F.PIX]), np.array([x0 + sx * F.PIX]), np.array([Y]), np.array([th]))
                iy = (c0 + sy + np.arange(-w - 1, w + 2)) % F.N; ix = (c0 + sx + np.arange(-w - 1, w + 2)) % F.N
                shiftchk.append(float(np.abs(o2[:, iy][:, :, ix] - win).max() / np.abs(win).max()))
                del o2
            del own
            pos = rng.integers(0, F.N, (NPOS, 2))
            for cy, cx in pos:
                res = credit_window(qn, win, int(cy), int(cx), w, QTH, CMIN)
                rows["node"].append(kn); rows["thi"].append(thi); rows["q"].append(qp)
                rows["y"].append(cy * F.PIX + dy); rows["x"].append(cx * F.PIX + dx)
                rows["qobs"].append(res[0]); rows["own"].append(res[1]); rows["j"].append(res[2]); rows["dy"].append(res[3])
                rows["dx"].append(res[4]); rows["nc"].append(res[5]); rows["npk"].append(res[6]); rows["own0"].append(o0)
    out = {k: np.array(v) for k, v in rows.items()}
    out.update(pid=pid, sig=sig, sig_build=ref, r15=r15, t=time.time() - t0, t_find=t_find, shiftchk=np.array(shiftchk))
    np.savez(fn, **out)
    import resource
    return job, (time.time() - t0, t_find, float(np.abs(sig / ref - 1).max()), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)


def compare(variant):
    """K_mine at the nodes vs the build's table."""
    from scipy.special import ndtr
    print(f"\n==== comparison, {variant}", flush=True)
    allpulls = []
    for mode in ("baseline", "goal"):
        fs = sorted(f for f in os.listdir(OUTD) if f.startswith(f"k_{variant}_{mode}_"))
        A = [np.load(os.path.join(OUTD, f)) for f in fs]
        kb = np.load(os.path.join(D3B, f"kernel_{variant}_{mode}.npz"))
        assert np.allclose(kb["th"], TH_K)
        E = kb["EDGES"]
        r05 = np.array([a["r15"][0] for a in A])
        dsig = max(float(np.abs(a["sig"] / a["sig_build"] - 1).max()) for a in A)
        print(f"  {mode}: {len(A)} noise-only tiles; my sigma_finder vs the build's stored sigma of the same tiles: max |rel| {dsig:.1e}; "
              f"r(0.5') = {r05.mean():.4f} +- {r05.std(ddof=1) / np.sqrt(r05.size):.4f} (tile mean)", flush=True)
        sc = np.concatenate([a["shiftchk"] for a in A if a["shiftchk"].size])
        if sc.size:
            print(f"     translation check (cluster painted at a shifted position vs the shifted window): max |diff| / peak = {sc.max():.1e} "
                  f"({sc.size} nodes)", flush=True)
        node = np.concatenate([a["node"] for a in A]); qobs = np.concatenate([a["qobs"] for a in A])
        own = np.concatenate([a["own"] for a in A]); own0 = np.concatenate([a["own0"] for a in A])
        for kn, (thi, qp) in enumerate(NODES[variant]):
            m = node == kn
            n = int(m.sum())
            Km = np.array([(qobs[m] > e).mean() for e in E])
            em = np.sqrt((Km * (1 - Km) + 0.25 / n) / n)
            qa = kb["QA"]
            ia = np.flatnonzero(np.isclose(qa, qp))
            if ia.size:
                Kt = kb["K"][thi, ia[0]]
                nod = np.flatnonzero(np.isclose(kb["nodes"], qp))
                if nod.size:
                    Kr = kb["Kraw"][thi, nod[0]]; nr = kb["n"][thi, ia[0]]
                else:
                    Kr = Kt; nr = kb["n"][thi, ia[0]]
                src = "node"
            else:                                    # bright end: interpolate the table in q' (dense model grid)
                k = np.searchsorted(qa, qp) - 1; u = (qp - qa[k]) / (qa[k + 1] - qa[k])
                Kt = (1 - u) * kb["K"][thi, k] + u * kb["K"][thi, k + 1]; Kr = Kt; nr = kb["n"][thi, k]
                src = "model"
            eb = np.sqrt((Kr * (1 - Kr) + 0.25 / max(nr, 1)) / max(nr, 1))
            sens = (Km > 0.02) & (Km < 0.98) | (Kt > 0.02) & (Kt < 0.98)
            pr = (Km - Kr) / np.sqrt(em ** 2 + eb ** 2)
            pt = (Km - Kt) / np.sqrt(em ** 2 + eb ** 2)
            allpulls += list(pr[sens])
            lam = np.median(own[m][np.isfinite(qobs[m])] / qp) if np.isfinite(qobs[m]).any() else np.nan
            es = [int(i) for i in np.flatnonzero(sens)]
            print(f"  {mode} node theta {TH_K[thi]:6.3f}' q' {qp:5.2f} ({src}): {n} clusters (build n {nr:.0f}); credited {np.isfinite(qobs[m]).mean():.4f}; "
                  f"own S/N at the credited peak / q' (median) {lam:.4f}; centre own/q' {np.median(own0[m]) / qp:.4f}", flush=True)
            if es:
                print("      e      : " + " ".join(f"{E[i]:7.2f}" for i in es), flush=True)
                print("      K mine : " + " ".join(f"{Km[i]:7.4f}" for i in es) + f"   (+- {np.median(em[es]):.4f})", flush=True)
                print("      K raw  : " + " ".join(f"{Kr[i]:7.4f}" for i in es) + f"   (+- {np.median(eb[es]):.4f})", flush=True)
                print("      K table: " + " ".join(f"{Kt[i]:7.4f}" for i in es), flush=True)
                print("      pull raw/table: " + " ".join(f"{pr[i]:+.1f}/{pt[i]:+.1f}" for i in es), flush=True)
            else:
                print(f"      K mine at e = 4: {Km[0]:.4f} (n {n}); build table {Kt[0]:.4f}", flush=True)
            # N_exp-like weight: mean difference over the sensitive edges
    ap = np.array(allpulls)
    if ap.size:
        print(f"  all sensitive (node, edge) pairs, mine vs the build's measured fraction: {ap.size} pulls, mean {ap.mean():+.2f}, rms {np.sqrt(np.mean(ap ** 2)):.2f}, "
              f"max |pull| {np.abs(ap).max():.2f} (pulls within a node are correlated across edges)", flush=True)


if __name__ == "__main__":
    variant, NPROC = sys.argv[1], int(sys.argv[2])
    os.makedirs(OUTD, exist_ok=True)
    jobs = [(variant, mode, int(p)) for p in TILES[variant] for mode in ("baseline", "goal")]
    t00 = time.time()
    print(f"f03: {variant}, {len(jobs)} noise-only tile runs (tiles {list(TILES[variant])}), nodes {NODES[variant]}, "
          f"{NOFF} x {NPOS} clusters per node per tile, {NPROC} processes", flush=True)
    if NPROC > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(NPROC, maxtasksperchild=8) as p:
            it = p.imap_unordered(work, jobs)
            for k, (job, info) in enumerate(it):
                if info:
                    print(f"  {k + 1}/{len(jobs)} {job}: {info[0]:.0f} s (finder {info[1]:.0f} s); sigma vs build max |rel| {info[2]:.1e}; "
                          f"worker peak RSS {info[3]:.2f} GB [{time.time() - t00:.0f} s]", flush=True)
    else:
        for k, job in enumerate(jobs):
            job, info = work(job)
            if info:
                print(f"  {k + 1}/{len(jobs)} {job}: {info[0]:.0f} s (finder {info[1]:.0f} s); sigma vs build max |rel| {info[2]:.1e}; "
                      f"peak RSS {info[3]:.2f} GB [{time.time() - t00:.0f} s]", flush=True)
    compare(variant)
    print(f"total {time.time() - t00:.0f} s")
