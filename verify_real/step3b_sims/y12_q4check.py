"""y12 (sims lens): why do painted sparse clusters lose at the q = 4 detection threshold against their virtual copies
on the halo-free sky (y08 (A): -0.5% above q = 4, 0.0% above q = 5)?  For the sparse sets of tiles T0..T1-1 (same
seeds and design as y03), each painted cluster is compared with three virtual copies (same Y):
  vf: on the halo-free tile's final S/N maps (y03's reference, the kernel's route);
  vi: on the CMB + noise maps filtered with the INJECTION tile's own final filter (same filter as the painted run);
and whether the painted cluster's pixel was inside the iteration-0 mask.  Counts above e = 4, 4.1, 4.5, 5.
Usage: python y12_q4check.py T0 T1"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import slib as L
import y03_inject as Y
V = L.V

T0, T1 = int(sys.argv[1]), int(sys.argv[2])
log = V.Tee(os.path.join(L.HERE, f"y12_q4check_{T0}_{T1}.log"))
t00 = time.time()
S = Y.setup(); G, P, HN = S["G"], S["P"], S["HN"]
nodes = L.nodes_for(np.r_[L.TH_K, Y.DSIZES])
res = {m: {k: [] for k in ("qp", "vf", "vi", "masked")} for m in L.MODES}
for t in range(T0, T1):
    for li, mode in enumerate(L.MODES):
        Xcn = V.cmb_mine(np.random.default_rng(Y.SEED_C + t), G) + V.noise_mine(np.random.default_rng(Y.SEED_N + t), mode, G)
        sy = V.sigma_Y(mode, L.THETAS)
        Rf = L.run_finder(Xcn, G, method="whitened", keep_cube=True)
        rf = Rf["sig1"] / sy; cube = Rf["cube"]; sigf = Rf["sig1"]
        RTf = L.response_tables(Rf["F1"], HN, nodes, 30.0)
        del Rf
        rng = np.random.default_rng(Y.SEED_I + 10 * t + li)
        for s in range(5):
            cells = rng.choice(36, 12, replace=False)
            ys = (cells // 6) * 100.0 + 50.0 + rng.uniform(-10, 10, 12)
            xs = (cells % 6) * 100.0 + 50.0 + rng.uniform(-10, 10, 12)
            ths = L.TH_K[rng.integers(0, L.TH_K.size, 12)]
            bright = rng.random(12) < 0.15
            qt = np.where(bright, rng.uniform(8.5, 30.0, 12), rng.uniform(2.5, 8.5, 12))
            Ys = qt * L.sigma_finder(mode, ths, rf)
            X = Xcn + V.tsz_channels(Y.paint_direct(S, ys, xs, Ys, ths), G)
            Ri = L.run_finder(X, G, method="whitened")
            del X
            RTi = L.response_tables(Ri["F1"], HN, nodes, 30.0)
            qo = Y.credit_injected(Ri["det1"], Ri["sig1"], RTi, ys, xs, Ys, ths)["none"]
            # virtual through the injection tile's own final filter
            Fi = Ri["F1"]; D = Fi.combine(Xcn)
            cubi = np.empty_like(cube)
            for j in range(15):
                cubi[j] = Fi.qmap(D, j)
            del D
            qvi, _ = L.virt_many(cubi, ys, xs, ths, Ys, RTi, Ri["sig1"], np.array([1.0]))
            qvf, _ = L.virt_many(cube, ys, xs, ths, Ys, RTf, sigf, np.array([1.0]))
            # was the painted cluster masked at iteration 0 (mask value at its pixel)?
            cy = (np.floor(ys / 0.5 + 0.5).astype(int)) % V.N; cx = (np.floor(xs / 0.5 + 0.5).astype(int)) % V.N
            masked = Ri["mask"][cy, cx] < 0.5
            res[mode]["qp"].append(qo); res[mode]["vf"].append(qvf[:, 0]); res[mode]["vi"].append(qvi[:, 0])
            res[mode]["masked"].append(masked)
            del cubi, Ri, RTi
        del cube, RTf
    log(f"  tile {t}: done [{time.time() - t00:.0f} s]")
for mode in L.MODES:
    r = {k: np.concatenate(v) for k, v in res[mode].items() if v}
    n = r["qp"].size
    log(f"\n[{mode}] {n} sparse clusters (tiles {T0}-{T1 - 1})")
    for e in (4.0, 4.1, 4.5, 5.0):
        p, vf, vi = r["qp"] > e, r["vf"] > e, r["vi"] > e
        log(f"  e = {e}: painted {p.sum()}, virtual halo-free filter {vf.sum()} (painted-only {np.sum(p & ~vf)}, "
            f"virtual-only {np.sum(vf & ~p)}), virtual injection filter {vi.sum()} (painted-only {np.sum(p & ~vi)}, "
            f"virtual-only {np.sum(vi & ~p)})")
    for lab, s in (("masked at iteration 0", r["masked"]), ("not masked", ~r["masked"])):
        p, vf = r["qp"][s] > 4.0, r["vf"][s] > 4.0
        log(f"  {lab}: {s.sum()} clusters; q > 4 painted-only {np.sum(p & ~vf)}, virtual(halo-free)-only {np.sum(vf & ~p)}")
log(f"\ny12 done in {time.time() - t00:.0f} s")
