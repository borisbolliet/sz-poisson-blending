"""j06 (project .venv): merging, claim 7 / the build's C7a, with the verifier's own calls.

(1) szsim.finder.apply_merge on every input the build merged with SZiFi (the build's merge_szifi.npz gives, per tile,
    the interior q >= 5 entries it fed to SZiFi and the entries SZiFi kept; read as data): are the kept entries
    identical, and does every difference involve a pair whose separation lies within 0.4% of the radius R?
(2) after j05: SZiFi's own merge_detections, called by the verifier on 2 catalogues, against the build's result for
    the same input and against szsim.finder.apply_merge.
Distances for the check in (1): flat, in arcmin, from the pixel positions (0.5' pixels).
Usage: .venv/bin/python -I -B j06_merge_compare.py
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.dont_write_bytecode = True
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np
from szsim import finder as F          # noqa: E402  (project: the production finder's merge rules)
import jlib as L                        # noqa: E402

t0 = time.time()
MB = L.load(os.path.join(ROOT, "data_real", "step3_szifi_real", "merge_szifi.npz"))
RULES = {"closest1": ("closest1", "closest", 1.0), "closest10": ("closest10", "closest", 10.0),
         "fof1": (None, "fof", 1.0), "fof10": ("fof10", "fof", 10.0)}
PIDS = sorted({int(k.split("_")[2]) for k in MB if k.endswith("_sel")})


def source_cat(src, mode, pid):
    """(row, col, q) of the full final catalogue: SZiFi S1 (the build's stored run) or the production finder."""
    if src == "szifi":
        r = L.load(L.run_path("build", "S1", mode, "real", pid), ["it1_i", "it1_j", "it1_q"])
        return r["it1_i"] - 0.5, r["it1_j"] - 0.5, np.asarray(r["it1_q"], float)
    st = L.prod_tile(pid)
    return st[f"{mode}_it1_iy"] * 1.0, st[f"{mode}_it1_ix"] * 1.0, np.asarray(st[f"{mode}_it1_q"], float)


def szsim_keep(y, x, q, rule):
    """szsim.finder's merge on the given entries (positions in arcmin); returns kept indices into the input."""
    det = {"q": q, "y": y, "x": x}
    if rule == "fof1":
        idx = np.flatnonzero(q > 0.0)
        k = F.merge_fof(y[idx], x[idx], q[idx], 1.0)
        return np.sort(idx[k])
    return np.sort(F.apply_merge(det, RULES[rule][0], 0.0))


def near_R_pairs(y, x, R, tol=0.004):
    d = np.hypot(y[:, None] - y[None, :], x[:, None] - x[None, :])
    iu = np.triu_indices(len(y), 1)
    m = np.abs(d[iu] / R - 1) <= tol
    return set(map(tuple, np.c_[iu[0][m], iu[1][m]])), int(np.sum(np.abs(d[iu] - R) < 1e-9))


print("j06 (1): szsim.finder's merge vs the build's stored SZiFi merge results, same inputs (the build's interior, q >= 5)")
tot = same = 0
ndiff_runs_with_pair = 0
ndiff_entries = ndiff_in_pair = 0
rows = {}
for src in ("prod", "szifi"):
    for mode in L.MODES:
        for pid in PIDS:
            key = f"{src}_{mode}_{pid:03d}"
            y, x, q = source_cat(src, mode, pid)
            sel = MB[key + "_sel"]
            ys, xs, qs = 0.5 * y[sel], 0.5 * x[sel], q[sel]
            for rule, (_, _, R) in RULES.items():
                kb = np.sort(MB[f"{key}_{rule}"])
                km = szsim_keep(ys, xs, qs, rule)
                tot += 1
                ident = np.array_equal(kb, km)
                same += ident
                rw = rows.setdefault((src, mode, rule), [0, 0, 0, 0, 0])
                rw[0] += 1
                rw[1] += ident
                rw[2] += kb.size
                rw[3] += km.size
                if not ident:
                    diff = np.setxor1d(kb, km)
                    rw[4] += diff.size
                    pairs, nexact = near_R_pairs(ys, xs, R)
                    inpair = {i for p in pairs for i in p}
                    ndiff_runs_with_pair += bool(pairs)
                    ndiff_entries += diff.size
                    ndiff_in_pair += sum(int(i) in inpair for i in diff)
                    rows.setdefault("difftiles", []).append((src, mode, pid, rule, diff.size, len(pairs), nexact))
print("  input  level    rule        tiles  identical  kept (build's SZiFi)  kept (szsim)  differing entries")
for (src, mode, rule), rw in [(k, v) for k, v in rows.items() if k != "difftiles"]:
    print(f"  {src:6s} {mode:8s} {rule:10s} {rw[0]:5d} {rw[1]:10d} {rw[2]:21d} {rw[3]:13d} {rw[4]:18d}")
ndr = tot - same
print(f"  all: identical in {same} of {tot} tile runs; of the {ndr} that differ, {ndiff_runs_with_pair} contain a pair within "
      f"0.4% of R; differing entries {ndiff_entries}, of which {ndiff_in_pair} are in such a pair")
print("  differing runs (input, level, tile, rule, entries, pairs within 0.4% of R, pairs exactly at R):")
for t in rows.get("difftiles", []):
    print(f"    {t}")

# ---- (2) the verifier's own SZiFi calls (j05)
fn = os.path.join(OUT, "j05_merge.npz")
if os.path.exists(fn):
    M = L.load(fn)
    print("\nj06 (2): SZiFi's merge_detections called by the verifier (j05) vs the build's result and szsim, same input")
    for cat in [str(c) for c in M["cats"]]:
        src, mode, pid = cat.split("_")[0], cat.split("_")[1], int(cat.split("_")[2])
        y, x, q = source_cat(src, mode, pid)
        sel = M[cat + "_sel"]
        ys, xs, qs = 0.5 * y[sel], 0.5 * x[sel], q[sel]
        print(f"  {cat}: {sel.size} interior entries with q >= 5 (input identical to the build's: {np.array_equal(sel, MB[cat + '_sel'])})")
        for rule, (_, _, R) in RULES.items():
            km = szsim_keep(ys, xs, qs, rule)
            kb = np.sort(MB[f"{cat}_{rule}"])
            for pixtag in [str(t) for t in M["pixtags"]]:
                kv = np.sort(M[f"{cat}_{rule}_{pixtag}"])
                pairs, nexact = near_R_pairs(ys, xs, R)
                dsz = np.setxor1d(kv, km)
                inpair = {i for p in pairs for i in p}
                print(f"    {rule:9s} HEALPix {pixtag:9s}: verifier's SZiFi call keeps {kv.size}; = build's SZiFi result: "
                      f"{np.array_equal(kv, kb)}; = szsim: {np.array_equal(kv, km)}"
                      + (f" (differs in {dsz.size} entries, {sum(int(i) in inpair for i in dsz)} of them in a pair within 0.4% of R; "
                         f"pairs within 0.4% of R: {len(pairs)}, exactly at R: {nexact})" if dsz.size else ""))

# ---- (3) real clusters removed by the merge (production catalogue, the build's interior q >= 5 input, step 3a's rules:
#      a detection is real if its primary halo's exact contribution c_p >= 2 sigma; credited halos = distinct primaries
#      of the real detections).  SZiFi's kept sets: the build's stored results (identical to the verifier's own SZiFi
#      calls on the 2 catalogues of (2)); szsim's: the verifier's apply_merge.
print("\nj06 (3): real clusters removed by merging, production finder's final catalogue, interior, q >= 5")
for mode in L.MODES:
    tot_b = {r: [0, 0] for r in RULES}
    tot_s = {r: [0, 0] for r in RULES}
    for pid in PIDS:
        with np.load(os.path.join(L.D3, "tiles", f"tile_{pid:03d}.npz")) as t:
            prim = t[f"{mode}_d_top_idx"][:, 0]
            cp = t[f"{mode}_d_top_ex"][:, 0]
            sig = t[f"{mode}_it1_sig"][t[f"{mode}_it1_j"]]
        key = f"prod_{mode}_{pid:03d}"
        sel = MB[key + "_sel"]
        real = (cp[sel] / sig[sel]) >= 2.0
        before = np.unique(prim[sel][real])
        y, x, q = source_cat("prod", mode, pid)
        for rule in RULES:
            for tot, kept in ((tot_b, np.sort(MB[f"{key}_{rule}"])),
                              (tot_s, szsim_keep(0.5 * y[sel], 0.5 * x[sel], q[sel], rule))):
                after = np.unique(prim[sel][kept][real[kept]])
                tot[rule][0] += before.size
                tot[rule][1] += after.size
    for rule in RULES:
        (b0, b1), (s0, s1) = tot_b[rule], tot_s[rule]
        print(f"  {mode:8s} {rule:9s}: credited halos {b0}; after SZiFi's merge {b1} (removed {100 * (1 - b1 / b0):.2f}%); "
              f"after szsim's merge {s1} (removed {100 * (1 - s1 / s0):.2f}%)")
print(f"\ndone [{time.time() - t0:.0f} s]")
