"""z05: production runs of SZiFi itself, one configuration per process, tiles one after the other.
Each run: z_szlib.run_tile (SZiFi's input_data with our survey file, cluster_finder.find_clusters with two noise
iterations) -> data_real/step3_szifi_real/runs/CFG_MODE_KIND_PID.npz (catalogues, sigma, SZiFi's recorded spectra,
unmasked fraction, timings, peak RSS).  Existing outputs are skipped.  The full parameter dictionary is printed once.
Usage: szifi-venv/bin/python -I -B z05_run.py CFG BUFFER_ARCMIN MODE:KIND:PID [MODE:KIND:PID ...]
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402

cfg, buf = sys.argv[1], float(sys.argv[2])
jobs = [(a.split(":")[0], a.split(":")[1], int(a.split(":")[2])) for a in sys.argv[3:]]
t0 = time.time()
print(f"z05: config {cfg}, edge buffer {buf}', {len(jobs)} tiles; SZiFi {Z.szifi.__file__}", flush=True)
printed = False
for k, (mode, kind, pid) in enumerate(jobs):
    fn = os.path.join(Z.RUND, f"{cfg}_{mode}_{kind}_{pid:03d}.npz")
    if os.path.exists(fn):
        print(f"  {mode:8s} {kind:5s} {pid:3d}: exists, skipped", flush=True)
        continue
    r = Z.run_tile(cfg, mode, kind, pid, buf, save=True, params_fh=None if printed else sys.stdout)
    printed = True
    msg = []
    for it in range(r["n_it"]):
        q = r[f"it{it}_q"]
        msg.append(f"it{it} q>4/4.5/5 {(q > 4).sum()}/{(q > 4.5).sum()}/{(q > 5).sum()}")
    if r["n_it"] == 2:
        rr = r["it1_sig"] / r["it0_sig"]
        msg.append(f"sigma it1/it0 at 0.5'/4.45'/15' {rr[0]:.4f}/{rr[9]:.4f}/{rr[14]:.4f}")
    msg.append(f"f {np.round(r['rec_fsky'], 4).tolist()}")
    print(f"  {mode:8s} {kind:5s} {pid:3d}: {r['t_run']:5.1f} s, peak RSS {r['peak_rss_gb']:.2f} GB; " + "; ".join(msg)
          + f"  [{k + 1}/{len(jobs)}, elapsed {time.time() - t0:.0f} s]", flush=True)
print(f"z05 {cfg} done [{time.time() - t0:.0f} s]; process peak RSS {Z.peak_rss_gb():.2f} GB", flush=True)
