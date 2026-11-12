"""y03 (sims lens): own sparse and dense injections on the halo-free skies of the own fresh tiles (production finder).

Per tile t and noise level, the halo-free sky (CMB seed 9_320_000 + t, noise seed 9_330_000 + t, as y02) is run
through the finder (final S/N cube, r_free, response tables).  Then:
  * SPARSE (NSET sets per tile): 12 clusters per set on a 6 x 6 grid of 100' cells (12 random cells, positions
    cell centre +- 10': >= 80' apart), theta from the 25 kernel nodes, target q' = Y / sigma_free(theta) uniform in
    2.5-8.5 (85%) or 8.5-30 (15%).  Painted (own direct Fourier painter = the NUFFT painter's template mixture) into
    the halo-free sky, the full finder run end to end (both iterations).  Each cluster: q' in the injection tile's
    own units (Y / sigma_finder,inj), q_obs of its highest credited detection (credited: primary among the injected
    clusters by table response, c_p >= 2 sigma) for merge none / closest 1' / fof 10'; and its virtual copy (same Y)
    on the halo-free S/N cube (y02's kernel route) -> painted vs virtual, cluster by cluster.
  * DENSE (tiles t < NDENSE): a02/t03-like design, one size per tile (0.3, 0.6, 1.2, 2.4, 4.8, 9.6' cycling),
    225 / 100 / 64 clusters on 40' / 60' / 75' grids (+- 2' jitter), q' uniform 1.8-8; painted and run end to end.
    sigma_dense / sigma_free per scale (same sky); each cluster's painted q_obs (q' in dense units); its virtual copy
    through the HALO-FREE filter at the same q' (halo-free units: the kernel's reference); and through the DENSE
    tile's own filter (CMB + noise through the dense filter, same q' in dense units).
Output: out/inject/i_TTTT.npz.   Usage: python y03_inject.py T0 T1 NPROC NSET NDENSE
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib as L
import numpy as np
from multiprocessing import Pool
V = L.V

SEED_C, SEED_N, SEED_I, SEED_D = 9_320_000, 9_330_000, 9_350_000, 9_360_000
OUTD = os.path.join(L.OUT, "inject")
DSIZES = (0.3, 0.6, 1.2, 2.4, 4.8, 9.6)
_S = {}


def setup():
    if not _S:
        G = V.grid()
        _S["G"] = G
        _S["P"] = V.Painter(G, eps=1e-9, scheme="lag4")
        _S["HN"] = V.HankelNodes(G)
        _S["kk"] = 2 * np.pi * G.k / L.LBOX
    return _S


def paint_direct(S, ys, xs, Ys, ths):
    """T(k) on the in-band modes of the painter's template mixture (direct Fourier sum, exact phases)."""
    G, P, kk = S["G"], S["P"], S["kk"]
    out = np.zeros(G.nm, complex)
    first, w = P.stencil(np.log(np.asarray(ths, float)))
    for c in range(len(ys)):
        g = sum(w[c, a] * P.g_node(int(first[c]) + a) for a in range(w.shape[1]))[G.mshell]
        ph = np.exp(-1j * kk * ys[c])[G.mrow] * np.exp(-1j * kk * xs[c])[G.mcol]
        out += Ys[c] * g * ph
    return out / V.OMP_AM


def credit_injected(det, sig, RT, ys, xs, Ys, ths, qcut=4.0, rcut=30.0):
    """q_obs per injected cluster (highest credited detection with q > qcut; primary = largest table contribution
    among the injected clusters within rcut; credited if c_p >= 2 sigma), for merge none / closest1 / fof10."""
    nc = len(ys)
    out = {}
    lth = np.log(ths)
    for m in ("none", "closest1", "fof10"):
        keep = V.apply_merge(det["iy"] * L.PIX, det["ix"] * L.PIX, det["q"], m, qcut=qcut)
        qo = np.full(nc, -np.inf)
        for d in keep:
            py, px, j = det["iy"][d] * L.PIX, det["ix"][d] * L.PIX, int(det["j"][d])
            dy = (ys - py + 0.5 * L.LBOX) % L.LBOX - 0.5 * L.LBOX
            dx = (xs - px + 0.5 * L.LBOX) % L.LBOX - 0.5 * L.LBOX
            r = np.sqrt(dy * dy + dx * dx)
            near = np.flatnonzero(r < rcut)
            if near.size == 0:
                continue
            c = np.array([Ys[i] * V.resp_eval(RT.R, j, RT.n0, lth[i], r[i], RT.dr, L.LNTH0, L.DNODE) for i in near]) / sig[j]
            k = int(np.argmax(c))
            if c[k] >= 2.0 and det["q"][d] > qo[near[k]]:
                qo[near[k]] = det["q"][d]
        out[m] = qo
    return out


def work(task):
    t, nset, ndense = task
    fn = os.path.join(OUTD, f"i_{t:04d}.npz")
    if os.path.exists(fn):
        return t, None
    S = setup()
    G, P, HN = S["G"], S["P"], S["HN"]
    t0 = time.time()
    res = dict(tile=t)
    nodes = L.nodes_for(np.r_[L.TH_K, DSIZES])
    for li, mode in enumerate(L.MODES):
        pre = mode + "_"
        Xcn = V.cmb_mine(np.random.default_rng(SEED_C + t), G)
        Xcn += V.noise_mine(np.random.default_rng(SEED_N + t), mode, G)
        sy = V.sigma_Y(mode, L.THETAS)
        Rf = L.run_finder(Xcn, G, method="whitened", keep_cube=True)
        rf = Rf["sig1"] / sy; cube = Rf["cube"]; sigf = Rf["sig1"]
        RTf = L.response_tables(Rf["F1"], HN, nodes, 30.0)
        del Rf
        res[pre + "rf"] = rf
        # ---------------- sparse sets
        rng = np.random.default_rng(SEED_I + 10 * t + li)
        cols = {k: [] for k in ("set", "th", "Y", "y", "x", "qt", "qinj", "qv", "q_none", "q_closest1", "q_fof10")}
        sig_inj = []
        for s in range(nset):
            cells = rng.choice(36, 12, replace=False)
            ys = (cells // 6) * 100.0 + 50.0 + rng.uniform(-10, 10, 12)
            xs = (cells % 6) * 100.0 + 50.0 + rng.uniform(-10, 10, 12)
            ths = L.TH_K[rng.integers(0, L.TH_K.size, 12)]
            bright = rng.random(12) < 0.15
            qt = np.where(bright, rng.uniform(8.5, 30.0, 12), rng.uniform(2.5, 8.5, 12))
            Ys = qt * L.sigma_finder(mode, ths, rf)
            if t == 0 and s == 0 and li == 0:                       # direct painter vs the NUFFT painter
                T1 = paint_direct(S, ys, xs, Ys, ths); T2 = P.T(ys, xs, Ys, ths)
                res["paint_check"] = float(np.abs(T1 - T2).max() / np.abs(T2).max())
            X = Xcn + V.tsz_channels(paint_direct(S, ys, xs, Ys, ths), G)
            Ri = L.run_finder(X, G, method="whitened")
            del X
            ri = Ri["sig1"] / sy
            sig_inj.append(ri / rf)
            RTi = L.response_tables(Ri["F1"], HN, nodes, 30.0)
            qo = credit_injected(Ri["det1"], Ri["sig1"], RTi, ys, xs, Ys, ths)
            del Ri, RTi
            qv, _ = L.virt_many(cube, ys, xs, ths, Ys, RTf, sigf, np.array([1.0]))
            cols["set"].append(np.full(12, s)); cols["th"].append(ths); cols["Y"].append(Ys); cols["y"].append(ys)
            cols["x"].append(xs); cols["qt"].append(qt); cols["qinj"].append(Ys / L.sigma_finder(mode, ths, ri))
            cols["qv"].append(qv[:, 0])
            for m in ("none", "closest1", "fof10"):
                cols["q_" + m].append(qo[m])
        for k, v in cols.items():
            res[pre + "sp_" + k] = np.concatenate(v)
        res[pre + "sp_sigratio"] = np.array(sig_inj)
        # ---------------- dense
        if t < ndense:
            rngd = np.random.default_rng(SEED_D + 10 * t + li)
            th = DSIZES[t % 6]
            sp, ng = (40.0, 15) if th <= 1.2 else ((60.0, 10) if th <= 4.8 else (75.0, 8))
            gy, gx = np.meshgrid(np.arange(ng), np.arange(ng), indexing="ij")
            ys = (gy.ravel() + 0.5) * sp + rngd.uniform(-2, 2, ng * ng)
            xs = (gx.ravel() + 0.5) * sp + rngd.uniform(-2, 2, ng * ng)
            ys %= L.LBOX; xs %= L.LBOX
            nc = ys.size
            ths = np.full(nc, th)
            qt = rngd.uniform(1.8, 8.0, nc)
            Ys = qt * L.sigma_finder(mode, ths, rf)
            X = Xcn + V.tsz_channels(paint_direct(S, ys, xs, Ys, ths), G)
            Rd = L.run_finder(X, G, method="whitened")
            del X
            rd = Rd["sig1"] / sy
            RTd = L.response_tables(Rd["F1"], HN, nodes, 30.0)
            qo = credit_injected(Rd["det1"], Rd["sig1"], RTd, ys, xs, Ys, ths)
            qd = Ys / L.sigma_finder(mode, ths, rd)                        # q' in the dense tile's units
            # virtual through the halo-free filter at the same q' (halo-free units)
            qvf, _ = L.virt_many(cube, ys, xs, ths, qd * L.sigma_finder(mode, ths, rf), RTf, sigf, np.array([1.0]))
            # virtual through the dense filter: CMB + noise through the dense tile's final filter
            Fd = Rd["F1"]
            D = Fd.combine(Xcn)
            cubed = np.empty_like(cube)
            for j in range(15):
                cubed[j] = Fd.qmap(D, j)
            del D
            qvd, _ = L.virt_many(cubed, ys, xs, ths, Ys, RTd, Rd["sig1"], np.array([1.0]))
            del cubed, Rd, RTd, Fd
            res.update({pre + "dn_th": th, pre + "dn_y": ys, pre + "dn_x": xs, pre + "dn_qt": qt, pre + "dn_Y": Ys,
                        pre + "dn_qd": qd, pre + "dn_rd": rd, pre + "dn_qpaint": qo["none"], pre + "dn_qvf": qvf[:, 0],
                        pre + "dn_qvd": qvd[:, 0], pre + "dn_qpaint_c1": qo["closest1"]})
        del cube, RTf, Xcn
    res["t_total"] = time.time() - t0
    res["peak_rss_gb"] = L.peak_rss_gb()
    os.makedirs(OUTD, exist_ok=True)
    np.savez_compressed(fn, **res)
    return t, (res["t_total"], res["peak_rss_gb"])


if __name__ == "__main__":
    T0, T1, NPROC, NSET, NDENSE = (int(a) for a in sys.argv[1:6])
    log = V.Tee(os.path.join(L.HERE, f"y03_inject_{T0}_{T1}.log"))
    t00 = time.time()
    log(f"y03: tiles {T0}..{T1 - 1}, {NSET} sparse sets of 12 clusters per tile and level, dense tiles < {NDENSE}; "
        f"{NPROC} processes; threads {L.threads_info()}")
    tasks = [(t, NSET, NDENSE) for t in range(T0, T1)]
    it = map(work, tasks) if NPROC == 1 else Pool(NPROC, maxtasksperchild=6).imap_unordered(work, tasks)
    for t, info in it:
        if info is None:
            log(f"  tile {t:4d}: exists"); continue
        log(f"  tile {t:4d}: {info[0]:5.1f} s; worker peak RSS {info[1]:.2f} GB [elapsed {time.time() - t00:.0f} s]")
    log(f"y03 done in {time.time() - t00:.0f} s wall")
