"""z06: why does S3 (SZiFi's SO-example settings) find tens of thousands of peaks on our tiles?  Iteration 0 only
(iterative False, so SZiFi's per-detection cluster masking is not run), one tile, S3 with one setting changed at a
time, plus S0 for reference:
  S3        : as configured (mixed_gaussian [10, 10], master, powspec_lmax1d 10000, lrange 1-300000, orphics, ...);
  S3_lr     : lrange 100-20000 (as S0-S2);
  S3_iso    : cov_type isotropic (= S3i at iteration 0);
  S3_full   : powspec_lmax1d None (full-resolution spectra and covariance);
  S3_none   : decouple_type 'none';
  S3_lr_iso_full : lrange 100-20000, isotropic, full resolution, master (should be S0 up to the decoupling).
Reported: detections at q > 4 / 5 inside the edge buffer, by scale, and sigma(theta) / S0's sigma.
Usage: szifi-venv/bin/python -I -B z06_s3diag.py BUFFER_ARCMIN [MODE KIND PID]
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z_szlib as Z                                   # noqa: E402
import numpy as np                                    # noqa: E402

buf = float(sys.argv[1])
mode, kind, pid = (sys.argv[2], sys.argv[3], int(sys.argv[4])) if len(sys.argv) > 4 else ("baseline", "real", 0)
S3 = dict(Z.CONFIGS["S3"], iterative=False)
VAR = {"S0it0": dict(Z.CONFIGS["S0"], iterative=False),
       "S3": S3,
       "S3_lr": dict(S3, lrange=[100, 20000]),
       "S3_iso": dict(S3, cov_type="isotropic"),
       "S3_full": dict(S3, powspec_lmax1d=None, coupling_matrix_needed=False, decouple_type="none"),
       "S3_none": dict(S3, decouple_type="none", coupling_matrix_needed=False),
       "S3_lr_iso_full": dict(S3, lrange=[100, 20000], cov_type="isotropic", powspec_lmax1d=None,
                              coupling_matrix_needed=False, decouple_type="none")}
print(f"z06: S3 diagnosis on {mode} {kind} {pid}, iteration 0 only, edge buffer {buf}'")
print("  note: S3_full and S3_lr_iso_full use decouple 'none' (a full-resolution coupling matrix is not computed here)")
t0 = time.time()
res = {}
for name, conf in VAR.items():
    Z.CONFIGS["diag_" + name] = conf
    r = Z.run_tile("diag_" + name, mode, kind, pid, buf, save=True, tag=f"z06_{name}")
    res[name] = r
    q, jt = r["it0_q"], r["it0_j_theta"]
    byj = np.bincount(jt[q > 4], minlength=15)
    print(f"  {name:15s}: {r['t_run']:5.1f} s, peak RSS {r['peak_rss_gb']:.2f} GB; q > 4 / 5: {(q > 4).sum():6d} / "
          f"{(q > 5).sum():6d}; median q of the detections {np.median(q) if q.size else np.nan:.1f}; q > 4 by scale "
          + " ".join(str(x) for x in byj), flush=True)
ref = res["S0it0"]["it0_sig"]
print("\n  sigma(theta) / S0's sigma, scales 0.5' ... 15':")
for name, r in res.items():
    print(f"  {name:15s} " + " ".join(f"{x:6.3f}" for x in r["it0_sig"] / ref))
print(f"done [{time.time() - t0:.0f} s]")
