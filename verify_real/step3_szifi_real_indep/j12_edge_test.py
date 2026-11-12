"""j12 (SZiFi's venv): a direct test of the edge mechanism found in j11.

On two of the verifier's new tiles (baseline, realistic: tile 41, whose cluster mask covers 20% of the tile boundary and
whose SZiFi r ends far above the emulation; tile 83, 1% boundary coverage; tile 9 as a further control), SZiFi's own
noise-map step of iteration 1 is repeated twice with SZiFi's own functions and the SAME cluster mask (SZiFi's
mmf.get_cluster_mask from SZiFi's iteration-0 catalogue of the verifier's S1 run):
  A  as SZiFi runs it: maps.filter_tmap, maps.inpaint_freq (diffusive, 100 sweeps) and
     maps.circular_cosine_blend_adaptive on the bare tile (holes cut at the tile edge are filled from one side only);
  B  the same calls on the tile extended periodically by 100 pixels on every side (the extension is the tile's own
     wrapped map and is unmasked), then cut back to the tile: the holes at the edge are filled from both sides, as on
     the periodic sky the tile comes from.
The binned auto spectra (own periodogram, 6 channels averaged) of A and B are compared with the unmasked map's, and the
mean squared jump across the periodic tile boundary is measured.  No division by f (that step is the same for A and B).
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B j12_edge_test.py   (run alone: about 0.6 GB)
"""
import os, sys, time, io, contextlib, resource
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.environ["NUMBA_CACHE_DIR"] = os.path.join(OUT, "numba_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(OUT, "mpl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/boris/GitHub/szifi")
import numpy as np
from szifi import maps, mmf, cat      # noqa: E402

t0 = time.time()
N, DX = 1200, 0.5 / 60.0 / 180.0 * np.pi
PAD = 100
TH = np.geomspace(0.5, 15.0, 15)
pix = maps.pixel(N, DX)
pixP = maps.pixel(N + 2 * PAD, DX)
LR = ((100, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 20000))
li = np.fft.fftfreq(N) * N
ell = 36.0 * np.sqrt(li[:, None] ** 2 + np.arange(N // 2 + 1)[None, :] ** 2)
w = np.full(N // 2 + 1, 2.0)
w[0] = w[-1] = 1.0


def band_power(m):
    """mean over the six channels of the l-range averages of |rfft2|^2 (full-plane weights)."""
    out = np.zeros((6, len(LR)))
    for k in range(6):
        P = np.abs(np.fft.rfft2(m[:, :, k].astype(np.float64))) ** 2 * w[None, :]
        for b, (lo, hi) in enumerate(LR):
            s = (ell >= lo) & (ell < hi)
            out[k, b] = P[s].sum() / (w[None, :] * s).sum()
    return out


def jump(m):
    return float(np.mean((m[0] - m[-1]) ** 2) + np.mean((m[:, 0] - m[:, -1]) ** 2)) / 2


def inpaint_blend(t, mask, p):
    inp = maps.inpaint_freq(t, mask, n_inpaint=100, pix=p, noise=None, inpaint_type="diffusive")
    for k in range(t.shape[2]):
        inp[:, :, k] = maps.circular_cosine_blend_adaptive(t[:, :, k], inp[:, :, k], mask, inner_zero_frac=0.)
    return inp


print("j12: SZiFi's iteration-1 noise maps on the bare tile (A) and on the periodically extended tile (B), same mask")
for pid in (41, 83, 9):
    t1 = time.time()
    mode = "baseline"
    tmap = np.load(os.path.join(OUT, "maps", f"real_{mode}_{pid:03d}.npy"))
    r = np.load(os.path.join(OUT, "runs", f"S1_{mode}_real_{pid:03d}.npz"))
    c = cat.cluster_catalogue()
    c.catalogue["q_opt"] = np.asarray(r["it0_q"], float)
    c.catalogue["theta_500"] = np.asarray(r["it0_theta"], float)
    c.catalogue["theta_x"] = np.asarray(r["it0_theta_x"], float)
    c.catalogue["theta_y"] = np.asarray(r["it0_theta_y"], float)
    with contextlib.redirect_stdout(io.StringIO()):
        mask = mmf.get_cluster_mask(pix, c, 4.0, 3.0)
    edge = np.concatenate([mask[0] < 0.5, mask[-1] < 0.5, mask[:, 0] < 0.5, mask[:, -1] < 0.5]).mean()
    t_or = maps.filter_tmap(tmap, pix, [100, 20000])
    A = maps.filter_tmap(inpaint_blend(t_or, mask, pix), pix, [100, 20000])
    tP = np.pad(t_or, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="wrap")
    mP = np.pad(mask, PAD, mode="constant", constant_values=1.0)
    B = maps.filter_tmap(inpaint_blend(tP, mP, pixP)[PAD:PAD + N, PAD:PAD + N], pix, [100, 20000])
    p0, pa, pb = band_power(t_or), band_power(A), band_power(B)
    ra, rb = (pa / p0).mean(0), (pb / p0).mean(0)
    ch = (pa / p0 - pb / p0)[:, 3]
    print(f"\n  tile {pid} {mode}: unmasked fraction {mask.mean():.4f} (SZiFi recorded {float(r['fsky_rec'][0]):.4f}); "
          f"boundary pixels covered by the mask {edge:.3f}")
    print(f"    auto spectra / unmasked, l ranges {LR}:")
    print(f"      A (bare tile, as SZiFi):  " + " ".join(f"{v:.4f}" for v in ra))
    print(f"      B (periodic extension):   " + " ".join(f"{v:.4f}" for v in rb))
    print(f"      A - B:                    " + " ".join(f"{v:+.4f}" for v in ra - rb))
    print(f"      A - B at l 6000-12000 per channel (27 ... 280 GHz): " + " ".join(f"{v:+.4f}" for v in ch))
    print(f"    mean squared jump across the periodic boundary / unmasked map's: A {np.mean([jump(A[:, :, k]) / jump(t_or[:, :, k]) for k in range(6)]):.2f}, "
          f"B {np.mean([jump(B[:, :, k]) / jump(t_or[:, :, k]) for k in range(6)]):.2f}  [{time.time() - t1:.0f} s]")
print(f"\ndone [{time.time() - t0:.0f} s]; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
