"""z10: test (e), part 2 -- SZiFi's own merging (z09, merge_szifi.npz) against szsim.finder's merging on the same
inputs (the interior q >= 5 entries of SZiFi's S1 final catalogues and of the production finder's final catalogues):
  'closest' R = 1', 10': szsim.finder.merge_closest (apply_merge's 'closest1' / 'closest10');
  'fof' R = 10': szsim.finder.merge_fof (apply_merge's 'fof10'); R = 1': merge_fof(..., 1.0).
Identical kept entries?  Differences are traced to pairs whose flat distance lies within 0.4% of R (SZiFi's flat-tile ->
sphere map shrinks east-west separations by up to cos(5 deg) at the tile's top and bottom edges) or exactly at R.
For the production catalogue: the fraction of real clusters removed by each merge, with step 3a's definitions (t05):
real = primary contribution c_p >= 2 sigma at the detection; credited halos = distinct primaries of real q > 5 entries;
removed = 1 - (distinct primaries of the kept real entries) / (credited halos before merging); SZiFi's and szsim's
merges on the same interior input.
Usage: python -I -B z10_merge_compare.py
"""
import os, sys, time
sys.dont_write_bytecode = True
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(ROOT, "data_real", "step3_szifi_real")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "out", "numba_cache_proj")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, ROOT)
import numpy as np
from szsim import finder as F                       # noqa: E402

t0 = time.time()
M = dict(np.load(os.path.join(OUT, "merge_szifi.npz")))
RULES = [str(r) for r in M["rules"]]
REF = dict(np.load(os.path.join(OUT, "reference.npz")))
SHIFT = 0.5


def szsim_merge(y, x, q, rule):
    R = float(rule.replace("closest", "").replace("fof", ""))
    if rule.startswith("closest"):
        return np.sort(F.merge_closest(y, x, q, R))
    return np.sort(F.merge_fof(y, x, q, R))


def near_radius_pairs(y, x, R, diff):
    """Pairs whose flat distance lies within 0.4% of R (and exactly at R) in the tile; and how many of the differing
    entries have such a partner themselves."""
    d = np.hypot(y[:, None] - y[None, :], x[:, None] - x[None, :])
    iu = np.triu_indices(y.size, 1)
    dd = d[iu]
    near = np.abs(d - R) <= 0.004 * R
    np.fill_diagonal(near, False)
    own = sum(bool(near[e].any()) for e in diff)
    return int(np.sum(np.abs(dd - R) <= 0.004 * R)), int(np.sum(dd == R)), own


print(f"z10: SZiFi's merge_detections vs szsim.finder on the same inputs (interior >= {float(M['interior_arcmin'])}' "
      f"from the edges, q >= 5)")
res = {}
loss = {}
for src in ("szifi", "prod"):
    for mode in ("baseline", "goal"):
        pids = sorted(int(k.split("_")[2]) for k in M if k.startswith(f"{src}_{mode}_") and k.endswith("_sel"))
        for pid in pids:
            key = f"{src}_{mode}_{pid:03d}"
            sel = M[key + "_sel"]
            if src == "szifi":
                r = np.load(os.path.join(OUT, "runs", f"S1_{mode}_real_{pid:03d}.npz"))
                y, x, q = (r["it1_i"] - SHIFT)[sel] * 0.5, (r["it1_j"] - SHIFT)[sel] * 0.5, r["it1_q"][sel]
            else:
                t = np.load(os.path.join(ROOT, "data_real", "step3", "tiles", f"tile_{pid:03d}.npz"))
                y, x, q = t[f"{mode}_it1_iy"][sel] * 0.5, t[f"{mode}_it1_ix"][sel] * 0.5, t[f"{mode}_it1_q"][sel]
                sig = t[f"{mode}_it1_sig"][t[f"{mode}_it1_j"][sel]]
                cp = t[f"{mode}_d_top_ex"][sel, 0]
                prim = t[f"{mode}_d_top_idx"][sel, 0]
                real = cp / sig >= 2.0
            for rule in RULES:
                ks = M[f"{key}_{rule}"]
                kz = szsim_merge(y, x, q, rule)
                same = np.array_equal(ks, kz)
                R = float(rule.replace("closest", "").replace("fof", ""))
                a = res.setdefault((src, mode, rule), dict(tiles=0, same=0, n_in=0, n_sz=0, n_ss=0, nd=0, near=0, eq=0,
                                                           own=0, dt_near=0))
                a["tiles"] += 1; a["same"] += same; a["n_in"] += q.size; a["n_sz"] += ks.size; a["n_ss"] += kz.size
                if not same:
                    diff = sorted(set(ks.tolist()) ^ set(kz.tolist()))
                    a["nd"] += len(diff)
                    nr, eq, own = near_radius_pairs(y, x, R, diff)
                    a["near"] += nr; a["eq"] += eq; a["own"] += own; a["dt_near"] += nr > 0
                if src == "prod":
                    cred = np.unique(prim[real])
                    for who, kk in (("szifi", ks), ("szsim", kz)):
                        kept = np.unique(prim[kk[real[kk]]])
                        b = loss.setdefault((mode, rule, who), [0, 0])
                        b[0] += cred.size; b[1] += kept.size
print("\n(1) kept entries, SZiFi's merge_detections vs szsim.finder, same input")
print(f"  {'input':6s} {'level':8s} {'rule':10s} {'tiles':>5s} {'identical':>9s} {'entries in':>10s} {'kept SZiFi':>10s} "
      f"{'kept szsim':>10s} {'differing':>9s}  in the tiles that differ: pairs within 0.4% of R (exactly R); tiles with such "
      f"a pair; differing entries that are themselves in such a pair")
tot = dict(tiles=0, same=0, nd=0, dt=0, dt_near=0, own=0)
for (src, mode, rule), a in sorted(res.items()):
    print(f"  {src:6s} {mode:8s} {rule:10s} {a['tiles']:5d} {a['same']:9d} {a['n_in']:10d} {a['n_sz']:10d} {a['n_ss']:10d} "
          f"{a['nd']:9d}  {a['near']:4d} ({a['eq']}); {a['dt_near']} of {a['tiles'] - a['same']}; {a['own']} of {a['nd']}")
    tot["tiles"] += a["tiles"]; tot["same"] += a["same"]; tot["nd"] += a["nd"]; tot["dt"] += a["tiles"] - a["same"]
    tot["dt_near"] += a["dt_near"]; tot["own"] += a["own"]
print(f"  all: identical in {tot['same']} of {tot['tiles']} tile runs; the {tot['dt']} that differ all contain a pair within "
      f"0.4% of R: {tot['dt_near'] == tot['dt']} ({tot['dt_near']} of {tot['dt']}); differing entries {tot['nd']}, of which "
      f"{tot['own']} are themselves in such a pair (the others follow from it through the merge order)")
print("\n(2) production finder: real clusters removed by the merge (interior, q > 5), SZiFi's merge vs szsim's")
for (mode, rule, who), (n0, n1) in sorted(loss.items()):
    print(f"  {mode:8s} {rule:10s} {who:6s}: credited halos {n0}, after merging {n1}: removed {100 * (1 - n1 / n0):.2f}%")
print(f"\ndone [{time.time() - t0:.0f} s]")
