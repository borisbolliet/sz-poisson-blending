"""Step 3 / t02: check (b), noise-only tiles (lensed CMB + white noise + atmosphere, no halos), SO baseline and goal.

Tiles pid = 0..NT-1 with the CMB and noise seeds of the realistic tiles (300000 + pid, 400000 + pid), so they are the
realistic tiles 0..NT-1 without their halos.  Per tile and noise level the finder is run twice:
  'whitened' : the production finder (iteration mask applied to the pre-whitened maps, pseudo-spectra / f);
  'binary'   : the literal f_sky estimate (band-limited maps x binary mask, pseudo-spectra / f).
Reported:
  (1) estimated binned spectra (iteration 0) against the input model, per l range: mean ratio over the tiles and the
      pull (estimate - model) / (Gaussian sample-variance error of one bin);
  (2) sigma(theta) of the finder against the analytic sigma (same grid, analytic covariance), before and after the
      iteration, both mask methods;
  (3) false detections (final catalogue) above q = 4, 4.5, 5 per 100 deg^2 for each merge rule, Poisson errors.
Usage: python t02_noise_only.py NT NPROC
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3lib as S                                  # noqa: E402
F = S.F
NT, NPROC = int(sys.argv[1]), int(sys.argv[2])
OUTD = os.path.join(S.OUT, "noise_only")


def work(pid):
    fn = os.path.join(OUTD, f"noise_{pid:03d}.npz")
    if os.path.exists(fn):
        return pid, 0.0, S.peak_rss_gb()
    t0 = time.time()
    G = F.grid()
    res = dict(pid=pid)
    for mode in S.MODES:
        X = F.sim_cmb(S.SEED_CMB + pid, G) + F.sim_noise(S.SEED_NOISE + pid, mode, G)
        for meth in ("whitened", "binary"):
            its = F.run_finder(X, G, mask_method=meth)
            o = S.finder_outputs(its, prefix=f"{mode}_{meth}_")
            if meth == "binary":
                o = {k: v for k, v in o.items() if "_C" not in k[-3:]}
            res.update(o)
    res["t"] = time.time() - t0
    np.savez(fn, **res)
    return pid, res["t"], S.peak_rss_gb()


def poisson_str(n, area):
    return f"{100 * n / area:.3f} +- {100 * np.sqrt(max(n, 1)) / area:.3f}"


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    t0 = time.time()
    with Pool(NPROC, maxtasksperchild=10) as pool:
        for pid, tt, rss in pool.imap_unordered(work, range(NT)):
            if pid % 10 == 0:
                print(f"  tile {pid:3d}: {tt:.1f} s, worker peak RSS {rss:.2f} GB [elapsed {time.time() - t0:.0f} s]", flush=True)
    print(f"runs done: {NT} tiles x 2 noise levels x 2 mask methods in {time.time() - t0:.0f} s wall", flush=True)

    G = F.grid()
    area = NT * 100.0
    R = [dict(np.load(os.path.join(OUTD, f"noise_{p:03d}.npz"))) for p in range(NT)]
    cnt = np.bincount(G.m_bin, weights=G.m_w, minlength=G.nbin)          # full-plane modes per bin
    nind = cnt / 2.0                                                       # independent complex modes
    lc = (np.arange(G.nbin) + 0.5) * F.DBIN
    for mode in S.MODES:
        print(f"\n=========== {mode}: {NT} noise-only tiles ({area:.0f} deg^2)")
        Ca = F.analytic_bin_cov(mode, G)
        C0 = np.array([r[f"{mode}_whitened_it0_C"] for r in R])            # (NT, nbin, 6, 6)
        print("(1) estimated spectra (iteration 0) / input model, mean over tiles [pull mean, pull rms] per l range")
        for lo, hi in ((100, 500), (500, 2000), (2000, 5000), (5000, 10000), (10000, 20000)):
            b = np.flatnonzero((lc >= lo) & (lc < hi))
            b = b[np.isin(b, G.bins_used)]
            line = []
            for i in range(6):
                rat = C0[:, b, i, i] / Ca[b, i, i]
                pull = (C0[:, b, i, i] - Ca[b, i, i]) / (Ca[b, i, i] / np.sqrt(nind[b]))
                line.append(f"{G.freqs[i]:.0f}: {rat.mean():.4f} [{pull.mean():+.2f}, {pull.std():.2f}]")
            # cross spectra pulls (all pairs)
            cp = []
            for i in range(6):
                for j in range(i + 1, 6):
                    v = (Ca[b, i, i] * Ca[b, j, j] + Ca[b, i, j] ** 2) / (2 * nind[b])
                    cp.append((C0[:, b, i, j] - Ca[b, i, j]) / np.sqrt(v))
            cp = np.array(cp)
            print(f"  l {lo}-{hi} ({b.size} bins): " + "; ".join(line) + f" | 15 cross spectra: pull mean {cp.mean():+.2f}, rms {cp.std():.2f}")
        fa = F.Filter(G, mode=mode, analytic=True)
        sa = fa.sigma_arcmin2
        print("(2) sigma(theta) / analytic sigma: mean [tile-to-tile std] at theta = " + ", ".join(f"{t:.2f}'" for t in F.THETAS15[::2]))
        for lab, key in (("iteration 0", "whitened_it0_sig"), ("iteration 1, whitened (finder)", "whitened_it1_sig"),
                         ("iteration 1, literal binary f_sky", "binary_it1_sig")):
            arr = np.array([r[f"{mode}_{key}"] / sa if f"{mode}_{key}" in r else r[f"{mode}_whitened_it0_sig"] / sa for r in R])
            print(f"  {lab:34s}: " + ", ".join(f"{m:.4f} [{s:.4f}]" for m, s in zip(arr.mean(0)[::2], arr.std(0)[::2])))
        nit = np.array([r[f"{mode}_whitened_n_it"] for r in R])
        print(f"  tiles with no q > 4 peak at iteration 0 (no iteration): {(nit == 1).sum()} of {NT}")
        print("(3) false detections per 100 deg^2 (final catalogue), q > 4 / 4.5 / 5:")
        for meth in ("whitened", "binary"):
            for mname in F.MERGES:
                ns = []
                for qc in (4.0, 4.5, 5.0):
                    n = 0
                    for r in R:
                        k = f"{mode}_{meth}_it{r[f'{mode}_{meth}_n_it'] - 1}"
                        det = dict(q=r[k + "_q"], y=r[k + "_iy"] * F.PIX, x=r[k + "_ix"] * F.PIX)
                        n += F.apply_merge(det, mname, qc).size
                    ns.append(n)
                print(f"  {meth:8s} {mname:9s}: " + " / ".join(poisson_str(n, area) for n in ns) + f"   (counts {ns})")
        n0 = [sum((r[f"{mode}_whitened_it0_q"] > qc).sum() for r in R) for qc in (4.0, 4.5, 5.0)]
        print(f"  iteration 0 (no merge): " + " / ".join(poisson_str(n, area) for n in n0) + f"   (counts {n0})")
    rss = [r["t"] for r in R]
    print(f"\ndone [{time.time() - t0:.0f} s]; mean time per tile (2 levels x 2 methods) {np.mean(rss):.1f} s")
