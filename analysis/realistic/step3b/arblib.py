"""Step 3b arbitration helpers: per-halo loaders for the build's production data (324 Q2 tiles + 1,224 new tiles) and
for the sims verifier's own 400 tiles (verify_real/step3b_sims/out/tiles_whitened), plus small statistics helpers.

Every array is read once per file (no npz[key][i] in loops).  New code; nothing of the build is modified.
Per-halo dict keys: t (tile index), z, th (theta500 [arcmin]), qp (q' in the finder's units), qo_none / qo_closest1 /
qo_fof10 (q_obs of the highest credited detection, -inf if not credited), qiso (isolated copy, -inf if not credited or
no copy), plus 'ntiles'.
"""
import os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(ROOT, "data_real", "step3b")
ARB = os.path.join(OUT, "arbiter")
SIMS = os.path.join(ROOT, "verify_real", "step3b_sims", "out")
ZC = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0])
MERGES = ("none", "closest1", "fof10")


def threads(n=1):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(n)


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def load_build(mode, which="all", variant="prod"):
    """Production per-halo arrays.  which: 'q2' (324 Q2 tiles), 'new' (1,224 new tiles), 'all' (both; tile index 0-1547:
    Q2 pids 0-323, new tiles 324 + k in sorted pid order, as b07)."""
    parts = []
    if which in ("q2", "all"):
        C = np.load(os.path.join(OUT, f"halos_{mode}.npz"))
        V = np.load(os.path.join(OUT, f"halos_{variant}_{mode}.npz"))
        pid = C["pid"].astype(np.int64)
        d = dict(t=pid, z=C["z"].astype(float), th=C["theta500"].astype(float), qp=V["qp"].astype(float),
                 Y=C["Y500"].astype(float), M=C["M500"].astype(float))
        for m in MERGES:
            d["qo_" + m] = V["qobs_" + m].astype(float)
        qi = np.full(pid.size, -np.inf)
        for p in range(324):
            with np.load(os.path.join(OUT, "virtual", variant, f"v_{mode}_{p:03d}.npz")) as v:
                qi[v["iso_row"]] = v["iso_qobs"]
        d["qiso"] = qi
        d["sig1"] = V["sig1"]
        parts.append(d)
    if which in ("new", "all") and variant == "prod":
        fs = sorted(f for f in os.listdir(os.path.join(OUT, "newtiles")) if f.endswith(".npz"))
        A = {k: [] for k in ("t", "z", "th", "qp", "Y", "M", "qiso", "qo_none", "qo_closest1", "qo_fof10")}
        sig = []
        for k, f in enumerate(fs):
            pid = int(f[2:6])
            t = (324 if which == "all" else 0) + k
            with np.load(os.path.join(OUT, "newtiles", f)) as dd, np.load(os.path.join(OUT, "truth_new", f"truth_{pid:04d}.npz")) as tr:
                rows = dd[mode + "_rows"]
                A["t"].append(np.full(rows.size, t)); A["z"].append(tr["z"][rows].astype(float))
                A["th"].append(tr["theta500"][rows].astype(float)); A["qp"].append(dd[mode + "_qp"].astype(float))
                A["Y"].append(tr["Y500"][rows].astype(float)); A["M"].append(tr["M500"][rows].astype(float))
                A["qiso"].append(dd[mode + "_qiso"].astype(float))
                for m in MERGES:
                    A["qo_" + m].append(dd[f"{mode}_qobs_{m}"].astype(float))
                sig.append(dd[mode + "_sig1"])
        d = {k: np.concatenate(v) for k, v in A.items()}
        d["sig1"] = np.array(sig)
        parts.append(d)
    if len(parts) == 1:
        out = parts[0]
    else:
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["ntiles"] = int(out["t"].max()) + 1
    return out


def load_sims(mode, meth="whitened"):
    """The sims verifier's own tiles: per-halo arrays (rows with q' >= 1.5; copies for q' >= 1.9)."""
    fs = sorted(glob.glob(os.path.join(SIMS, f"tiles_{meth}", "t_*.npz")))
    A = {k: [] for k in ("t", "z", "th", "qp", "Y", "M", "qiso", "qo_none", "qo_closest1", "qo_fof10")}
    extra = {k: [] for k in ("ndet5", "ncred_none", "ncred_fof10", "nh")}
    for k, fn in enumerate(fs):
        with np.load(fn) as d:
            n = d[mode + "_rows"].size
            A["t"].append(np.full(n, k)); A["z"].append(d[mode + "_z"]); A["th"].append(d[mode + "_th"])
            A["qp"].append(d[mode + "_qp"]); A["Y"].append(d[mode + "_Y"]); A["M"].append(d[mode + "_M"])
            A["qiso"].append(d[mode + "_qiso"])
            for m in MERGES:
                A["qo_" + m].append(d[f"{mode}_qobs_{m}"])
            extra["ndet5"].append(int(d[mode + "_ndet5"])); extra["ncred_none"].append(int(d[mode + "_ncred_none"]))
            extra["ncred_fof10"].append(int(d[mode + "_ncred_fof10"])); extra["nh"].append(int(d["nh"]))
    out = {k: np.concatenate(v).astype(float) if k != "t" else np.concatenate(v) for k, v in A.items()}
    out.update({k: np.array(v) for k, v in extra.items()})
    out["ntiles"] = len(fs)
    return out


def load_xpipe(mode, t0, t1):
    """The arbiter's a02 runs (the build's pipeline on tiles made with the sims verifier's generators), tiles t0..t1-1:
    per-halo arrays as load_sims; z and theta500 re-drawn with the verifier's halo sampler (the rows index its halos)."""
    sys.path.insert(0, os.path.join(ROOT, "verify_real", "step3b_sims"))
    import vlib3b as V
    A = {k: [] for k in ("t", "z", "th", "qp", "Y", "M", "qiso", "qo_none", "qo_closest1", "qo_fof10")}
    extra = {k: [] for k in ("tile", "nr_q", "nr_z", "nr_t")}
    k = 0
    for t in range(t0, t1):
        fn = os.path.join(ARB, "xpipe", f"x_{mode}_{t:04d}.npz")
        if not os.path.exists(fn):
            continue
        h = V.draw_tile_halos(9_310_000 + t)
        with np.load(fn) as d:
            rows = d["rows"]
            A["t"].append(np.full(rows.size, k)); A["z"].append(h["z"][rows]); A["th"].append(h["theta500"][rows])
            A["Y"].append(h["Y500"][rows]); A["M"].append(h["M500"][rows])
            A["qp"].append(d["qp"].astype(float)); A["qiso"].append(d["qiso"].astype(float))
            for m in MERGES:
                A["qo_" + m].append(d[f"qobs_{m}"].astype(float))
            nr = d["det5_cp"] < 2.0
            extra["nr_q"].append(d["det5_q"][nr].astype(float)); extra["nr_t"].append(np.full(int(nr.sum()), k))
            pr = d["det5_prim"][nr]
            extra["nr_z"].append(np.where(pr >= 0, h["z"][np.clip(pr, 0, None)], -1.0))
        extra["tile"].append(t)
        k += 1
    out = {kk: np.concatenate(v) for kk, v in A.items()}
    out.update({kk: np.concatenate(v) if kk != "tile" else np.array(v) for kk, v in extra.items()})
    out["ntiles"] = k
    return out


def boot_w(nt, nboot=2000, seed=2026):
    """Tile-bootstrap multiplicities (nboot, nt)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nt, (nboot, nt))
    return np.stack([np.bincount(x, minlength=nt) for x in idx]).astype(np.float64)


def per_tile(t, w, nt):
    return np.bincount(t, weights=w, minlength=nt).astype(float)


def ratio_boot(a, b, W):
    """sum(a)/sum(b) - 1 and its bootstrap std for per-tile arrays a, b (nt,) and multiplicities W (nboot, nt)."""
    v = a.sum() / b.sum() - 1.0
    vb = (W @ a) / (W @ b) - 1.0
    return v, vb.std(), vb


def fmt(v, e, s=2):
    return f"{100 * v:+.{s}f} +- {100 * e:.{s}f}%"
