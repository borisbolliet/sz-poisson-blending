"""j02: bit-level reproducibility of the build's SZiFi runs, and SZiFi's sigma bug (claim E1), from the verifier's runs.

For each of the build's runs that the verifier repeated with its own survey file and driver, every stored output is
compared with the verifier's: catalogues of both iterations (q, y0, theta500, pixel, sky coordinates), SZiFi's
sigma_vec, the spectra of each iteration, the unmasked fraction f, and SZiFi's parameter dictionary.
Claim E1: SZiFi's sigma_vec at the first scale is 0 with 'maxima_lomem'.  The verifier records the std that
mmf.get_mmf_q_map returns at every scale; the build recovered sigma(0.5') as y0/q from the catalogue.
Usage: .venv/bin/python -I -B j02_repro.py
"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.dont_write_bytecode = True
import numpy as np

BUILD = "/Users/boris/GitHub/sz-poisson-blending/data_real/step3_szifi_real/runs"
REPRO = [("S1", "baseline", "real", 11), ("S2", "goal", "noise", 22), ("S3it0", "goal", "real", 33)]


def load(fn):
    with np.load(fn) as d:
        return {k: d[k] for k in d.files}


def cmp(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if a.size == 0:
        return "identical (empty)"
    if np.array_equal(a, b):
        return "identical"
    if np.issubdtype(a.dtype, np.number):
        return f"differ: max |diff| {np.abs(a.astype(float) - b.astype(float)).max():.2e}"
    return "differ"


print("j02: the verifier's SZiFi runs vs the build's stored runs (same configuration, tile and noise level)")
nall = nsame = 0
for cfg, mode, kind, pid in REPRO:
    mine_fn = os.path.join(OUT, "runs", f"{cfg}_{mode}_{kind}_{pid:03d}.npz")
    if not os.path.exists(mine_fn):
        print(f"  {cfg} {mode} {kind} {pid}: not run")
        continue
    a, b = load(mine_fn), load(os.path.join(BUILD, f"{cfg}_{mode}_{kind}_{pid:03d}.npz"))
    print(f"\n  {cfg} {mode} {kind} {pid:3d}: iterations {int(a['n_it'])} (build {int(b['n_it'])})")
    for k in range(int(a["n_it"])):
        rows = [("q", f"it{k}_q", f"it{k}_q"), ("y0", f"it{k}_y0", f"it{k}_y0"), ("theta500", f"it{k}_theta", f"it{k}_theta"),
                ("pixel i", f"it{k}_i", f"it{k}_i"), ("pixel j", f"it{k}_j", f"it{k}_j"),
                ("theta_x", f"it{k}_theta_x", f"it{k}_theta_x"), ("theta_y", f"it{k}_theta_y", f"it{k}_theta_y"),
                ("SZiFi sigma_vec", f"it{k}_sigvec", f"it{k}_sigvec"), ("spectra", f"spec{k}", f"spec{k}"),
                ("spectra l", f"spec{k}_ell", f"spec{k}_ell")]
        for lab, ka, kb in rows:
            r = cmp(a[ka], b[kb])
            nall += 1
            nsame += r.startswith("identical")
            print(f"    it{k} {lab:16s}: {r}" + (f"  ({a[ka].size} values)" if r.startswith("identical") else ""))
        # sigma at the first scale: SZiFi's sigma_vec, the verifier's recorded std, the build's y0/q rule
        s_rec, s_vec, s_b = a[f"it{k}_sig_rec"], a[f"it{k}_sigvec"], b[f"it{k}_sig"]
        print(f"    it{k} sigma(0.5'): SZiFi sigma_vec {s_vec[0]:.3e}; recorded std {s_rec[0]:.6e}; build (y0/q) {s_b[0]:.6e} "
              f"(ratio - 1 = {s_b[0] / s_rec[0] - 1:+.1e}); scales 2-15: recorded std = SZiFi sigma_vec: "
              f"{np.array_equal(s_rec[1:], s_vec[1:])}")
    if int(a["n_it"]) > 1:
        r = cmp(a["fsky_rec"], b["rec_fsky"])
        nall += 1
        nsame += r.startswith("identical")
        print(f"    unmasked fraction f: {r} ({a['fsky_rec']} vs {b['rec_fsky']})")
    pa, pb = json.loads(str(a["params_json"])), json.loads(str(b["params_json"]))
    dif = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
    print(f"    SZiFi parameter dictionary: {len(pa)} keys; keys whose values differ: {dif if dif else 'none'}")
    for k in dif:
        print(f"      {k}: verifier {pa.get(k)!r} | build {pb.get(k)!r}")
print(f"\n  compared arrays: {nall}; identical: {nsame}")

# ---- claim E1 over all of the verifier's runs
print("\nE1: SZiFi's sigma_vec at the first scale ('maxima_lomem'), over all of the verifier's runs")
nz = tot = 0
worst = 0.0
for fn in sorted(glob.glob(os.path.join(OUT, "runs", "*.npz"))):
    a = load(fn)
    for k in range(int(a["n_it"])):
        tot += 1
        nz += a[f"it{k}_sigvec"][0] == 0.0
        q, y0, jt = a[f"it{k}_q"], a[f"it{k}_y0"], a[f"it{k}_jt"]
        for j in range(1, 15):
            m = jt == j
            if m.any():
                worst = max(worst, float(np.abs(y0[m] / q[m] / a[f"it{k}_sig_rec"][j] - 1).max()))
print(f"  iterations with sigma_vec[0] == 0: {nz} of {tot}; the y0/q rule vs the recorded std at scales 2-15: max |ratio - 1| {worst:.1e}")
