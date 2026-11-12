"""z02: smoke test -- SZiFi itself, one configuration, on one tile (realistic tile pid 0, SO baseline): does it run,
how long does it take, how much memory; the full parameter dictionary it used; a first look at sigma and the
catalogues.  One configuration per process (so /usr/bin/time -l gives that configuration's peak memory).
Usage: szifi-venv/bin/python -I -B z02_smoke.py CFG BUFFER_ARCMIN [MODE KIND PID]
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402  (sets the environment, imports SZiFi)
import numpy as np                                    # noqa: E402

cfg, buf = sys.argv[1], float(sys.argv[2])
mode, kind, pid = (sys.argv[3], sys.argv[4], int(sys.argv[5])) if len(sys.argv) > 5 else ("baseline", "real", 0)
print(f"z02 smoke test: SZiFi {Z.szifi.__file__} (clone at commit 73626fb), config {cfg}, tile {mode} {kind} {pid}, "
      f"edge buffer {buf}' in the peak-finding and selection masks")
print(f"parameters that differ from SZiFi's defaults: {sorted(Z.differs_from_default(Z.make_params(cfg)).keys())}")
if cfg.startswith("S3"):
    print(f"coupling matrix file {Z.CM_FILE}: exists before the run: {os.path.exists(Z.CM_FILE)}")
t0 = time.time()
r = Z.run_tile(cfg, mode, kind, pid, buf, save=True, params_fh=sys.stdout, tag=f"smoke{cfg}")
print(f"\nwall time {r['t_run']:.1f} s (loading {r['t_load']:.1f} s); peak RSS {r['peak_rss_gb']:.2f} GB")
print(f"  SZiFi steps [s]: cross spectra {np.round(r['rec_t_cross'], 1).tolist()}, inverse covariance "
      f"{np.round(r['rec_t_invcov'], 1).tolist()}, filtering + peaks {np.round(r['rec_t_filter'], 1).tolist()}, "
      f"inpainting {np.round(r['rec_t_inpaint'], 1).tolist()}, blend {np.round(r['rec_t_blend'], 1).tolist()}, "
      f"cluster mask {np.round(r['rec_t_mask'], 1).tolist()}")
print(f"  noise iterations run: {r['n_it']}; cluster mask: unmasked fraction {r['rec_fsky'].tolist()}, "
      f"masked detections {r['rec_n_masked'].tolist()}")
for it in range(r["n_it"]):
    q = r[f"it{it}_q"]
    sv, sc = r[f"it{it}_sigvec"], r[f"it{it}_sig_cat"]
    ok = np.isfinite(sc[1:])
    print(f"  iteration {it}: SZiFi's sigma_vec[0] = {sv[0]:.3e} (maxima_lomem skips it); sigma from its catalogue "
          f"(median y0/q per scale) vs sigma_vec at scales 1-14: max |ratio - 1| = {np.abs(sc[1:][ok] / sv[1:][ok] - 1).max():.1e}; "
          f"sigma(0.5') from the catalogue = {sc[0]:.4e} ({(r[f'it{it}_j_theta'] == 0).sum()} detections)")
    print(f"  iteration {it}: sigma [y0] at 0.5'/4.45'/15' = {r[f'it{it}_sig'][0]:.4e} / {r[f'it{it}_sig'][9]:.4e} / "
          f"{r[f'it{it}_sig'][14]:.4e}; detections q > 4 / 4.5 / 5: {(q > 4).sum()} / {(q > 4.5).sum()} / {(q > 5).sum()}")
if r["n_it"] == 2:
    print(f"  sigma it1 / it0: " + " ".join(f"{x:.4f}" for x in r["it1_sig"] / r["it0_sig"]))
    s0, s1 = r["spec0"], r["spec1"]
    e = r["spec0_ell"]
    for lo, hi in ((100, 3000), (3000, 20000)):
        b = (e >= lo) & (e < hi)
        rat = np.mean([np.mean(s1[b, i, i] / s0[b, i, i]) for i in range(6)])
        print(f"  auto spectra it1 / it0, l {lo}-{hi}: {rat:.4f}")
print(f"  spectra recorded: {r['n_spec']} calls, {r['spec0'].shape[0]} bins, l_eff {r['spec0_ell'][0]:.1f} .. {r['spec0_ell'][-1]:.1f}")
if cfg.startswith("S3"):
    print(f"coupling matrix file exists after the run: {os.path.exists(Z.CM_FILE)}")
print(f"done [{time.time() - t0:.0f} s]")
