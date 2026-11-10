"""V3: independent catalogue-blending MC on a polar cap (C4, C5, C6, C9, C10).
Own sampler (indep_model.draw), own response table (indep_model.Instrument.rho_table), neighbour search with
scipy cKDTree (analysis uses a numba dec/RA grid), own ybar for the mean removal.
Primaries: dec > 55 deg, q > 0.25; neighbours: all M>1e13 halos with dec > 50 deg (5 deg = 300' pad).
Usage: python v3_blend_patch.py BEAM SEED [SEED ...]
"""
import os, sys, time, itertools, resource, json
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import numba as nb
from scipy.spatial import cKDTree
import indep_model as im
from paircount import count_within

BEAM = float(sys.argv[1]); SEEDS = [int(s) for s in sys.argv[2:]]
ARC = im.ARC
inst = im.Instrument(BEAM, 4e-6)
thn = np.geomspace(0.04, 150, 130)
sepn = np.concatenate([np.linspace(0, 20, 401), np.geomspace(20, 700, 181)[1:]])
rho = inst.rho_table(thn, sepn)
lth0, dlth = np.log(thn[0]), np.log(thn[1] / thn[0])
dlogs = np.log(sepn[-1] / 20.0) / 180
Wd = np.load(os.path.join(im.CACHE, "W_own.npz"))


@nb.njit(parallel=True, cache=True)
def rho_eval(rho, lth0, dlth, dlogs, la, lb, s, out):
    nth = rho.shape[0]
    for i in nb.prange(la.shape[0]):
        fa = (la[i] - lth0) / dlth; fb = (lb[i] - lth0) / dlth
        fa = min(max(fa, 0.0), nth - 1.000001); fb = min(max(fb, 0.0), nth - 1.000001)
        ia = int(fa); ib = int(fb); fa -= ia; fb -= ib
        si = s[i]
        if si >= 700.0:
            out[i] = 0.0; continue
        if si < 20.0:
            fs = si / 0.05
        else:
            fs = 400.0 + np.log(si / 20.0) / dlogs
        js = int(fs)
        if js > 579: js = 579
        fs -= js
        v = 0.0
        for da in range(2):
            wa = fa if da else 1 - fa
            for db in range(2):
                wb = fb if db else 1 - fb
                v += wa * wb * (rho[ia + da, ib + db, js] * (1 - fs) + rho[ia + da, ib + db, js + 1] * fs)
        out[i] = v


def unit(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


EDGES = [0, 0.5, 1, 2, 4, 8, 16, 32, np.inf]
yb13 = im.ybar(); yb14 = im.ybar(Mmin=1e14)
res_all = []
t00 = time.time()
for seed in SEEDS:
    t0 = time.time()
    c = im.draw(seed, inst, Mmin=1e13, sinmin=np.sin(np.radians(50.0)))
    N = len(c["ra"])
    xyz = unit(c["ra"], c["dec"])
    chi = im.chi(c["z"]); Rc = im.R500(c["M"], c["z"]) * (1 + c["z"])
    prim = np.nonzero((c["dec"] > 55.0) & (c["q"] > 0.25))[0]
    npr = len(prim)
    print(f"[beam {BEAM} seed {seed}] N = {N} (expected {c['lam']:.0f}), primaries {npr}; {time.time()-t0:.0f}s", flush=True)
    thA = c["th"][prim]; lthA = np.log(thA)
    S = np.zeros((npr, 2)); X = np.zeros((npr, 2)); OV = np.zeros(npr); HASOV = np.zeros(npr, bool)
    npairs = 0
    for b0, b1 in zip(EDGES[:-1], EDGES[1:]):
        nb_idx = np.nonzero((c["th"] >= b0) & (c["th"] < b1))[0]
        if len(nb_idx) == 0:
            continue
        tree = cKDTree(xyz[nb_idx])
        thBmax = c["th"][nb_idx].max()
        rmax = np.minimum(5 * (thA + thBmax) + 4 * BEAM, 700.0)
        rch = 2 * np.sin(rmax * ARC / 2)
        for a in range(0, npr, 20000):
            bsl = slice(a, min(npr, a + 20000))
            lst = tree.query_ball_point(xyz[prim[bsl]], rch[bsl], return_sorted=False)
            lens = np.fromiter((len(x) for x in lst), np.int64, len(lst))
            flat = nb_idx[np.fromiter(itertools.chain.from_iterable(lst), np.int64, int(lens.sum()))]
            rep = np.repeat(np.arange(bsl.start, bsl.stop), lens)
            notself = flat != prim[rep]
            flat, rep = flat[notself], rep[notself]
            npairs += len(flat)
            ch = np.linalg.norm(xyz[flat] - xyz[prim[rep]], axis=1)
            sep = 2 * np.arcsin(np.clip(ch / 2, 0, 1)) / ARC
            r = np.empty(len(flat)); rho_eval(rho, lth0, dlth, dlogs, lthA[rep], np.log(c["th"][flat]), sep, r)
            t = r * c["Y"][flat]
            cls = (c["M"][flat] >= 1e14).astype(np.int64)
            for k in (0, 1):
                m = cls == k
                S[:, k] += np.bincount(rep[m], weights=t[m], minlength=npr)
                np.maximum.at(X[:, k], rep[m], t[m])
            rp = 0.5 * (chi[flat] + chi[prim[rep]]) * sep * ARC
            ov = rp ** 2 + (chi[flat] - chi[prim[rep]]) ** 2 < (Rc[flat] + Rc[prim[rep]]) ** 2
            OV += np.bincount(rep[ov], weights=t[ov], minlength=npr)
            HASOV[np.unique(rep[ov])] = True
            del lst, flat, rep, ch, sep, r, t
        del tree
    print(f"   neighbour sums done: {npairs/npr:.0f} pairs/primary; {time.time()-t0:.0f}s; rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB", flush=True)
    sA = inst.sigY(thA); q0 = c["q"][prim]; n = c["n"][prim]
    WA06 = np.interp(np.log(thA), np.log(Wd["thn"]), Wd["W06"]); WAy = np.interp(np.log(thA), np.log(Wd["thn"]), Wd["Wy"])
    R = dict(seed=seed)
    for cname, f, floor in (("base", 1.0, 1e13), ("noise_x0.5", 0.5, 1e13), ("noise_x2", 2.0, 1e13), ("floor_1e14", 1.0, 1e14)):
        q = q0 / f; keep = q > 0.5
        D = (S[:, 1] + (S[:, 0] if floor < 1e14 else 0)) / sA / f
        Xm = np.maximum(X[:, 1], X[:, 0] if floor < 1e14 else 0) / sA / f
        mu = sA * im.RATIO * (yb13 if floor < 1e14 else yb14) / 4e-6 ** 2 / f
        dom = Xm < q
        single = keep & (q + n > 5)
        r = dict(n_single=int(single.sum()), sat=int((single & ~dom).sum()),
                 sumD_single=float(D[single].sum()), summu_single=float(mu[single].sum()), sumD2_single=float((D[single] ** 2).sum()))
        for cc in (0, 1):
            dp = D - cc * mu
            det = keep & (q + n + dp > 5) & dom
            r[f"det{cc}"] = int(det.sum()); r[f"b2_{cc}"] = int((det & (dp / q > 0.1)).sum()); r[f"b3_{cc}"] = int((det & (q + n < 5)).sum())
            if cc == 1:
                r["b2ovl"] = int((det & (OV / sA / f / q > 0.1)).sum())
                for tag, WA in (("cl06", WA06), ("cly", WAy)):
                    dpc = dp + WA * mu
                    detc = keep & (q + n + dpc > 5) & dom
                    r[f"det_{tag}"] = int(detc.sum()); r[f"b2_{tag}"] = int((detc & (dpc / q > 0.1)).sum()); r[f"b3_{tag}"] = int((detc & (q + n < 5)).sum())
                if cname == "base":
                    r["hasov_single"] = int((single & HASOV).sum())
                    # pair statistics of the blended catalogue and of single detections (centres dec > 56)
                    for tag, sel in (("b", det), ("s", single)):
                        ii = np.nonzero(sel)[0]
                        cnt = count_within(c["ra"][prim[ii]], c["dec"][prim[ii]], [0.5, 1.0, 2.0, 5.0])
                        cen = c["dec"][prim[ii]] > 56.0
                        r[f"pairs_{tag}_ncen"] = int(cen.sum())
                        r[f"pairs_{tag}_ge1"] = [int(np.sum(cnt[cen, k] >= 1)) for k in range(4)]
                        r[f"pairs_{tag}_N55"] = int(len(ii))
        R[cname] = r
        print(f"   {cname:11s}: N_single {r['n_single']}, B2(0.1) c1 {r['b2_1']/r['det1']:.4f}, B3 c1 {r['b3_1']/r['det1']:.4f}, "
              f"B2 c0 {r['b2_0']/r['det0']:.4f}, B3 c0 {r['b3_0']/r['det0']:.4f}, sat {r['sat']/r['n_single']:.4f}, "
              f"<D>_single {r['sumD_single']/r['n_single']:.4f} vs <mu> {r['summu_single']/r['n_single']:.4f}", flush=True)
    res_all.append(R)
    json.dump(R, open(os.path.join(im.CACHE, f"v3_beam{BEAM:.1f}_seed{seed}.json"), "w"))
    del c, xyz, S, X
print(f"total {time.time()-t00:.0f}s; peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
