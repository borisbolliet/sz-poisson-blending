"""j05 (SZiFi's venv): SZiFi's own catalogue tools on 2 catalogues, called by the verifier (claim 7, the build's E3).

As in SZiFi's Planck scripts: get_catalogue_q_th(cat, 5) then merge_detections(cat, radius_arcmin=R,
return_merge_flag=True, mode=M), M = 'closest' / 'fof', R = 1' / 10'.  Sky positions from SZiFi's own
cluster_catalogue.get_lonlat (a Cartesian projection around a HEALPix nside-8 pixel), for two pixels: the one at
(lon, lat) = (180, 0) and one at (45, 60); angular distances should not depend on the pixel.
Inputs (chosen from j06 (1): runs where szsim and the build's SZiFi merge differ): the production finder's final
catalogue, goal, tile 88 (4 differing entries, 3 pairs exactly at 10'), and SZiFi S1's final catalogue, goal, tile 198
(a difference without a pair exactly at 10').  The entries are the build's interior q >= 5 selection (merge_szifi.npz
'_sel', read as data), so that the input is the same as the build's.  theta_x = (j + 1/2) dx, theta_y = (N - i - 1/2) dx
(SZiFi's convention for a detection at pixel (i, j)).
First SZiFi's merging is tried as shipped (it fails with numpy 2.5.4 / scikit-learn 1.9.1, the build's E3); then with
the verifier's own two adapters, which change no merging logic:
  * scikit-learn: in szifi.cat's namespace AgglomerativeClustering accepts the removed keyword 'affinity' and passes it
    on as 'metric' (the callable then returns SZiFi's full spherical distance matrix, as before);
  * healpy: in szifi.cat's and szifi.sphere's namespaces, healpy.rotator.angdist returns a plain float when it computes a
    single distance (numpy 2.5 no longer stores a length-1 array into one element).
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B j05_merge_szifi.py
"""
import os, sys, time, io, contextlib, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "numba_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(OUT, "mpl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/boris/GitHub/szifi")
import numpy as np
import healpy as hp
import sklearn
import szifi                                  # noqa: E402
from szifi import cat, sphere, maps           # noqa: E402

t0 = time.time()
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
N, DX = 1200, 0.5 / 60.0 / 180.0 * np.pi
TH = np.geomspace(0.5, 15.0, 15)
PIX = maps.pixel(N, DX)
MB = np.load(os.path.join(ROOT, "data_real", "step3_szifi_real", "merge_szifi.npz"))
CATS = ["prod_goal_088", "szifi_goal_198"]
PIXELS = {"lon180lat0": int(hp.ang2pix(8, 180.0, 0.0, lonlat=True)), "lon45lat60": int(hp.ang2pix(8, 45.0, 60.0, lonlat=True))}
RULES = (("closest", 1.0), ("closest", 10.0), ("fof", 1.0), ("fof", 10.0))
VERS = f"numpy {np.__version__}, healpy {hp.__version__}, scikit-learn {sklearn.__version__}"


def entries(name):
    src, mode, pid = name.split("_")[0], name.split("_")[1], int(name.split("_")[2])
    if src == "szifi":
        with np.load(os.path.join(ROOT, "data_real", "step3_szifi_real", "runs", f"S1_{mode}_real_{pid:03d}.npz")) as r:
            i, j, q, th = r["it1_i"], r["it1_j"], np.asarray(r["it1_q"], float), np.asarray(r["it1_theta"], float)
    else:
        with np.load(os.path.join(ROOT, "data_real", "step3", "tiles", f"tile_{pid:03d}.npz")) as t:
            i, j, q, th = t[f"{mode}_it1_iy"], t[f"{mode}_it1_ix"], np.asarray(t[f"{mode}_it1_q"], float), TH[t[f"{mode}_it1_j"]]
    sel = MB[name + "_sel"]
    return sel, i[sel], j[sel], q[sel], th[sel]


def szifi_catalogue(i, j, q, th, ipix):
    c = cat.cluster_catalogue()
    c.catalogue["q_opt"] = np.asarray(q, float)
    c.catalogue["y0"] = np.zeros(len(q))
    c.catalogue["theta_500"] = np.asarray(th, float)
    c.catalogue["theta_x"] = (np.asarray(j, float) + 0.5) * DX
    c.catalogue["theta_y"] = (N - np.asarray(i, float) - 0.5) * DX
    c.catalogue["row"] = np.arange(len(q), dtype=float)          # carried through the merge: which input entry is kept
    c.get_lonlat(8, ipix, PIX)
    return c


def merge(c, mode, R):
    cq = szifi.get_catalogue_q_th(c, 5.0)
    with contextlib.redirect_stdout(io.StringIO()):
        out = szifi.merge_detections(cq, radius_arcmin=R, return_merge_flag=True, mode=mode)
    if isinstance(out, tuple):
        out = out[0]
    return np.sort(np.asarray(out.catalogue["row"], float).astype(np.int64))


print(f"j05: SZiFi's merge_detections, called by the verifier ({VERS})")
sel, i, j, q, th = entries(CATS[0])
c_try = szifi_catalogue(i, j, q, th, PIXELS["lon180lat0"])
for mode in ("closest", "fof"):
    try:
        merge(c_try, mode, 10.0)
        print(f"  '{mode}' as shipped: runs")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        where = [f"{os.path.basename(t.filename)}:{t.lineno} ({t.name})" for t in tb if "szifi" in t.filename][-1:]
        print(f"  '{mode}' as shipped: fails: {type(e).__name__}: {e} [last SZiFi frame {where}]")

# ---- the verifier's adapters
_AC = cat.AgglomerativeClustering


def ac_with_affinity(*a, affinity=None, **k):
    if affinity is not None:
        k["metric"] = affinity
    return _AC(*a, **k)


class _Rot:
    def __getattr__(self, name):
        return getattr(hp.rotator, name)

    @staticmethod
    def angdist(*a, **k):
        v = np.asarray(hp.rotator.angdist(*a, **k))
        return v.reshape(-1)[0].item() if v.size == 1 else v


class _Hp:
    rotator = _Rot()

    def __getattr__(self, name):
        return getattr(hp, name)


cat.AgglomerativeClustering = ac_with_affinity
for mode in ("closest", "fof"):
    try:
        merge(c_try, mode, 10.0)
        print(f"  '{mode}' with the scikit-learn adapter only: runs")
    except Exception as e:
        print(f"  '{mode}' with the scikit-learn adapter only: fails: {type(e).__name__}: {e}")
cat.hp = sphere.hp = _Hp()
for mode in ("closest", "fof"):
    try:
        merge(c_try, mode, 10.0)
        print(f"  '{mode}' with both adapters: runs")
    except Exception as e:
        print(f"  '{mode}' with both adapters: fails: {type(e).__name__}: {e}")

out = {"cats": np.array(CATS), "pixtags": np.array(list(PIXELS))}
for name in CATS:
    sel, i, j, q, th = entries(name)
    out[name + "_sel"] = sel
    print(f"\n  {name}: {sel.size} interior entries with q >= 5")
    for tag, ipix in PIXELS.items():
        c = szifi_catalogue(i, j, q, th, ipix)
        for mode, R in RULES:
            k = merge(c, mode, R)
            out[f"{name}_{mode}{int(R)}_{tag}"] = k
            same_b = np.array_equal(k, np.sort(MB[f"{name}_{mode}{int(R)}"]))
            print(f"    HEALPix pixel {ipix:3d} ({tag}): {mode:7s} {R:4.0f}': keeps {k.size}; identical to the build's SZiFi result: {same_b}")
np.savez(os.path.join(OUT, "j05_merge.npz"), **out)
print(f"done [{time.time() - t0:.0f} s]")
