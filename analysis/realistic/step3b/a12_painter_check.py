"""Step 3b arbitration / a12: does the painter matter?  The sims verifier's tiles painted with the BUILD's painter.

a11: the verifier's generator (800 tiles, the build's pipeline) gives a baseline paired loss 0.28 +- 0.11 pp larger than
the build's 1,548 tiles.  The two generators share the halo model, the CMB draw (same white-noise map and spectrum) and
the noise model; they differ in the halo sampler (per-tile Poisson cells vs full-sky catalogues cut into patches), the
noise construction and the painter (exact NUFFT synthesis with 4-point Lagrange nodes vs the build's real-space painting).
Paired test of the painter: for the verifier's tiles T0..T1-1 (baseline), the same halos (draw_tile_halos(9_310_000 + t))
are painted with szsim.finder.paint_y and put on the same CMB + noise (the verifier's generators); the build's finder and
credit rule run as in a02.  Compared halo by halo with a02's output for the NUFFT-painted map (same halos, same CMB +
noise): credited halos at q > 5 and their q_obs; sigma_finder.
Usage: python a12_painter_check.py T0 T1 NPROC
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
from multiprocessing import Pool                   # noqa: E402
import a02_cross_pipeline as A2                    # noqa: E402
import b15_newtiles as B15                         # noqa: E402
F = B.F


def work(t):
    mode = "baseline"
    t0 = time.time()
    V, VG, P = A2.sims_lib()
    G = F.grid()
    bas, tables = B.basis()
    h = V.draw_tile_halos(A2.SEED_H + t)
    hh = {k: np.asarray(h[k], np.float64) for k in ("y", "x", "z", "theta500", "Y500")}
    Xt = F.tsz_channels(F.paint_y(hh["y"], hh["x"], hh["Y500"], hh["theta500"], tables, G), G)
    Xcn = A2.to_rfft(V.cmb_mine(np.random.default_rng(A2.SEED_C + t), VG) + V.noise_mine(np.random.default_rng(A2.SEED_N + t), mode, VG), VG)
    its = F.run_finder(Xt + Xcn, G)
    filt, det = its[-1]["filt"], its[-1]["det"]
    sig = filt.sigma_arcmin2.copy()
    sel, prim, cps = B15.credit(filt, det, hh, bas)
    real = cps >= B.CRED_MIN
    qo = np.full(hh["z"].size, -np.inf)
    np.maximum.at(qo, prim[real], det["q"][sel[real]])
    ref = np.load(os.path.join(A2.OUTD, f"x_{mode}_{t:04d}.npz"))
    rows = ref["rows"]
    a = qo[rows]; b = ref["qobs_none"].astype(float)
    both = np.isfinite(a) & np.isfinite(b)
    return t, dict(n_both=int(np.sum((a > 5) & (b > 5))), only_build_paint=int(np.sum((a > 5) & ~(b > 5))),
                   only_nufft=int(np.sum(~(a > 5) & (b > 5))), dq=(a[both] - b[both]), dsig=float(np.abs(sig / ref["sig1"] - 1).max()),
                   sec=time.time() - t0)


if __name__ == "__main__":
    T0, T1, NPROC = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    t00 = time.time()
    print(f"a12: the verifier's tiles {T0}..{T1 - 1} (baseline) painted with the build's painter, against the NUFFT-painted a02 runs", flush=True)
    it = map(work, range(T0, T1)) if NPROC == 1 else Pool(NPROC).imap_unordered(work, range(T0, T1))
    tot = dict(n_both=0, only_build_paint=0, only_nufft=0); dq = []; dsig = 0.0
    for t, r in it:
        for k in tot:
            tot[k] += r[k]
        dq.append(r["dq"]); dsig = max(dsig, r["dsig"])
    dq = np.concatenate(dq)
    print(f"  credited halos at q > 5: both {tot['n_both']}, only with the build's painter {tot['only_build_paint']}, only with the NUFFT "
          f"painter {tot['only_nufft']}; q_obs (credited in both): mean difference {dq.mean():+.2e}, rms {dq.std():.2e}, max |.| "
          f"{np.abs(dq).max():.2e} ({dq.size} halos); max |sigma ratio - 1| {dsig:.1e}", flush=True)
    print(f"a12 done in {time.time() - t00:.0f} s", flush=True)
