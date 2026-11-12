"""z09: test (e), part 1 -- SZiFi's own catalogue merging (cat.merge_detections), as its Planck script calls it
(get_catalogue_q_th(cat, 5.) then merge_detections(cat, radius_arcmin=R, return_merge_flag=True, mode=M)), on
  'szifi' : SZiFi's S1 final catalogues (iteration 1), and
  'prod'  : the production finder's final catalogues (data_real/step3/tiles/tile_PID.npz, iteration 1),
restricted to the analysis interior (z07), for M = 'closest' and 'fof', R = 1' and 10'.  The sky positions are made by
SZiFi's own catalogue.get_lonlat (its flat-tile -> sphere map for 'healpix' tiles: a Cartesian projection centred on a
HEALPix nside-8 pixel, here the one at lon 180, lat 0).  Kept entries are identified by an 'index' column.
Both modes first run as SZiFi calls them (numpy 2.5.4, healpy 1.20.1, scikit-learn 1.9.1); both fail, so the merges
use two thin adapters (z_szlib.sklearn_affinity_adapter, z_szlib.healpy_angdist_adapter).
Output: data_real/step3_szifi_real/merge_szifi.npz (the merged inputs and SZiFi's kept indices); compared with
szsim.finder in z10_merge_compare.py (project venv).
Usage: szifi-venv/bin/python -I -B z09_merge_szifi.py
"""
import os, sys, time, io, contextlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402
import healpy as hp                                   # noqa: E402
from szifi import cat, maps                           # noqa: E402

t0 = time.time()
N, DX = Z.N, Z.DX
EB = np.load(os.path.join(Z.OUT, "edge_buffer.npz"))
M_INT, SHIFT = float(EB["interior_arcmin"]), 0.5
REF = np.load(os.path.join(Z.OUT, "reference.npz"))
PIX = maps.pixel(N, DX)
IPIX = int(hp.ang2pix(8, 180.0, 0.0, lonlat=True))
RULES = (("closest", 1.0), ("closest", 10.0), ("fof", 1.0), ("fof", 10.0))
TH = Z.THETAS15


def dist_edge(y, x):
    return np.minimum(np.minimum(y, N - 1 - y), np.minimum(x, N - 1 - x)) * 0.5


def make_cat(i, j, q, th):
    c = cat.cluster_catalogue()
    c.catalogue["q_opt"] = np.asarray(q, float)
    c.catalogue["theta_500"] = np.asarray(th, float)
    c.catalogue["theta_x"] = (np.asarray(j, float) + 0.5) * DX
    c.catalogue["theta_y"] = (N - np.asarray(i, float) - 0.5) * DX
    c.catalogue["y0"] = np.zeros(len(q))
    c.catalogue["index"] = np.arange(len(q), dtype=float)
    c.get_lonlat(8, IPIX, PIX)
    return c


def szifi_merge(c, mode, R):
    cq = Z.szifi.get_catalogue_q_th(c, 5.0)
    with contextlib.redirect_stdout(io.StringIO()):
        out = Z.szifi.merge_detections(cq, radius_arcmin=R, return_merge_flag=True, mode=mode)
    if isinstance(out, tuple):
        out = out[0]
    return np.sort(np.asarray(out.catalogue["index"], float).astype(np.int64))


print(f"z09: SZiFi's merge_detections on interior catalogues (>= {M_INT}' from the edges); sky map via SZiFi's "
      f"get_lonlat, HEALPix nside 8 pixel {IPIX}")
# SZiFi's merging as it is, with the installed numpy, healpy and scikit-learn; then with the two adapters
r0 = np.load(os.path.join(Z.RUND, "S1_baseline_real_000.npz"))
c_try = make_cat(r0["it1_i"][:40], r0["it1_j"][:40], r0["it1_q"][:40] + 5.0, r0["it1_theta"][:40])
vers = (f"numpy {np.__version__}, healpy {hp.__version__}, scikit-learn {__import__('sklearn').__version__}")


def attempt(mode, lab):
    try:
        szifi_merge(c_try, mode, 10.0)
        print(f"  '{mode}' {lab}: runs")
    except Exception as e:
        print(f"  '{mode}' {lab}: fails ({vers}): {type(e).__name__}: {e}")


attempt("closest", "as SZiFi calls it")
attempt("fof", "as SZiFi calls it")
Z.sklearn_affinity_adapter()
attempt("fof", "with the scikit-learn adapter only")
Z.healpy_angdist_adapter()
attempt("closest", "with the healpy adapter")
attempt("fof", "with both adapters")
print("  -> all merges below use both adapters (z_szlib.sklearn_affinity_adapter, healpy_angdist_adapter)")
out = {"rules": np.array([f"{m}{int(R)}" for m, R in RULES]), "interior_arcmin": M_INT}
ntile = {}
for mode in ("baseline", "goal"):
    for pid in [int(p) for p in REF["pids_real"]]:
        fn = os.path.join(Z.RUND, f"S1_{mode}_real_{pid:03d}.npz")
        if not os.path.exists(fn):
            continue
        r = np.load(fn)
        t = np.load(os.path.join(Z.ROOT, "data_real", "step3", "tiles", f"tile_{pid:03d}.npz"))
        srcs = {"szifi": (r["it1_i"], r["it1_j"], r["it1_q"], r["it1_theta"], r["it1_i"] - SHIFT, r["it1_j"] - SHIFT),
                "prod": (t[f"{mode}_it1_iy"], t[f"{mode}_it1_ix"], t[f"{mode}_it1_q"], TH[t[f"{mode}_it1_j"]],
                         t[f"{mode}_it1_iy"] * 1.0, t[f"{mode}_it1_ix"] * 1.0)}
        for src, (i, j, q, th, yy, xx) in srcs.items():
            sel = np.flatnonzero((dist_edge(yy, xx) >= M_INT - 1e-9) & (q >= 5.0))
            key = f"{src}_{mode}_{pid:03d}"
            out[key + "_sel"] = sel                       # indices into the run's / stored final catalogue
            c = make_cat(i[sel], j[sel], q[sel], th[sel])
            for m, R in RULES:
                out[f"{key}_{m}{int(R)}"] = szifi_merge(c, m, R)
            ntile[(src, mode)] = ntile.get((src, mode), 0) + 1
np.savez(os.path.join(Z.OUT, "merge_szifi.npz"), **out)
for (src, mode), n in sorted(ntile.items()):
    print(f"  {src:5s} {mode:8s}: {n} tiles merged")
print(f"done [{time.time() - t0:.0f} s]")
