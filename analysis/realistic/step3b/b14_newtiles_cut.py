"""Step 3b / b14: new Poisson halo tiles for the precision target (more halos for the headline data vector).

Drawn exactly like the Q2 patches 108-323 (analysis/q2_maps/repair/extract_patches_set2.py): one full-sky Poisson
catalogue per seed with szsim.sampler.draw_catalog(seed, Mmin=1e13) (the same HMF table, Y-M relation with lognormal
scatter and z = 0.01-3, M500 = 1e13-1e16 as the Q2 skies); the equatorial band |Y| < 5 deg of the equal-area map
X = ra, Y = (180/pi) sin dec is cut into 36 patches of 10 deg x 10 deg (exactly 100 deg^2 of sky each, a homogeneous
Poisson process, treated as a periodic tile).  New seeds 10..10 + NSKY - 1 (the Q2 skies used 1..9); tile id
pid = 324 + 36 (seed - 10) + p.  Truth table per tile in step 3a's format (x, y [arcmin], z, M500, theta500, Y500,
q_baseline, q_goal = Y500 / sigma_Y(theta500) of step 1's MMF).  Printed: full-sky counts against the expectation and
the halos per tile against the 324 Q2 tiles.
Outputs: data_real/step3b/truth_new/truth_PID.npz.  Usage: python b14_newtiles_cut.py NSKY
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(2)
import numpy as np                                 # noqa: E402
from szsim import sampler                          # noqa: E402

L = 10.0
SEED0 = 10

if __name__ == "__main__":
    NSKY = int(sys.argv[1])
    outd = os.path.join(B.OUT, "truth_new")
    os.makedirs(outd, exist_ok=True)
    t00 = time.time()
    nper = []
    for seed in range(SEED0, SEED0 + NSKY):
        t0 = time.time()
        if all(os.path.exists(os.path.join(outd, f"truth_{324 + 36 * (seed - SEED0) + p:04d}.npz")) for p in range(36)):
            print(f"seed {seed}: tiles exist (pids {324 + 36 * (seed - SEED0)}-{324 + 36 * (seed - SEED0) + 35})", flush=True)
            continue
        cat = sampler.draw_catalog(seed, Mmin=1e13, fields=("ra", "dec", "z", "M500", "Y500", "theta500"))
        Y = (180 / np.pi) * np.sin(np.radians(cat["dec"].astype(np.float64)))
        sel = np.abs(Y) < L / 2
        X = cat["ra"][sel].astype(np.float64); Yb = Y[sel]
        d = {k: cat[k][sel] for k in ("z", "M500", "Y500", "theta500")}
        N = cat["meta"]["N"]; Nexp = cat["meta"]["N_expected"]
        del cat, Y
        ip = np.floor(X / L).astype(int)
        for p in range(36):
            m = ip == p
            pid = 324 + 36 * (seed - SEED0) + p
            th = d["theta500"][m].astype(np.float64); Yv = d["Y500"][m].astype(np.float64)
            x = ((X[m] - p * L) * 60.0) % 600.0; y = ((Yb[m] + L / 2) * 60.0) % 600.0
            np.savez(os.path.join(outd, f"truth_{pid:04d}.npz"), x=x, y=y, z=d["z"][m].astype(np.float32),
                     M500=d["M500"][m].astype(np.float32), theta500=th.astype(np.float32), Y500=Yv.astype(np.float32),
                     q_baseline=(Yv / B.sigma_Y("baseline", th)).astype(np.float32),
                     q_goal=(Yv / B.sigma_Y("goal", th)).astype(np.float32), seed=seed)
        print(f"seed {seed}: N_fullsky {N} (expected {Nexp:.0f}, {(N - Nexp) / np.sqrt(Nexp):+.2f} sigma); band halos {sel.sum()}; "
              f"pids {324 + 36 * (seed - SEED0)}-{324 + 36 * (seed - SEED0) + 35} [{time.time() - t0:.1f} s]", flush=True)
    nper = np.array([np.load(os.path.join(outd, f"truth_{p:04d}.npz"))["z"].size for p in range(324, 324 + 36 * NSKY)], float)
    ref = np.array([np.load(os.path.join(B.D3, "truth", f"truth_{p:03d}.npz"))["z"].size for p in range(324)], float)
    print(f"halos per tile: new {nper.mean():.1f} +- {nper.std(ddof=1) / np.sqrt(nper.size):.1f} ({nper.size} tiles), Q2 tiles "
          f"{ref.mean():.1f} +- {ref.std(ddof=1) / np.sqrt(ref.size):.1f}; Poisson expectation per tile {sampler.expected_count(Mmin=1e13) * 100 / (4 * np.pi * (180 / np.pi) ** 2):.1f}")
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {B.peak_rss_gb():.2f} GB")
