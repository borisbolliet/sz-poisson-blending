"""Step 3b arbitration / a05: claim P3 (ML / linearised bootstrap scatter of the headline's S8 bias).

The build (b09b) gives std(ML) / std(lin) = 1.05 / 1.09 (config d) and 1.16 / 1.18 (config e) on its first 60 tile-
bootstrap resamples of R_s; the fits verifier (f09) gives 1.04 / 1.07 and 1.07 / 1.06 on the first 12 of the same
resamples with its own model and fitter.  Here: (1) the build's own ML fits restricted to the same 12 resamples;
(2) resample by resample, build ML vs verifier ML (same R_s resample; different model code and fitter);
(3) the uncertainty of the 60-resample ratio (bootstrap over the 60 resamples).
Inputs: data_real/step3b/mlboot.json, verify_real/step3b_fits/out/f09.json.  Usage: python a05_mlboot_check.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arblib as A
A.threads(1)
import numpy as np

if __name__ == "__main__":
    B = json.load(open(os.path.join(A.OUT, "mlboot.json")))
    V = json.load(open(os.path.join(A.ROOT, "verify_real", "step3b_fits", "out", "f09.json")))
    rng = np.random.default_rng(1)
    for mode in ("baseline", "goal"):
        for cn in ("d", "e"):
            rows = sorted([r for r in B if r["mode"] == mode and r["config"] == cn and r["b"] >= 0], key=lambda r: r["b"])
            bb = np.array([r["b"] for r in rows]); ml = np.array([r["b_S8"] for r in rows]); li = np.array([r["lin_S8"] for r in rows])
            ratio60 = ml.std(ddof=1) / li.std(ddof=1)
            rb = []
            for _ in range(4000):
                s = rng.integers(0, ml.size, ml.size)
                rb.append(ml[s].std(ddof=1) / li[s].std(ddof=1))
            rb = np.array(rb)
            s12 = bb < 12
            ratio12 = ml[s12].std(ddof=1) / li[s12].std(ddof=1)
            sub = []
            for k in range(5):
                s = (bb >= 12 * k) & (bb < 12 * (k + 1))
                sub.append(ml[s].std(ddof=1) / li[s].std(ddof=1))
            vv = {r[0][3]: (r[1], r[2]) for r in V if r[0][0] == mode and r[0][1] == cn and r[0][2] == "build"}
            common = [b for b in range(12) if b in vv]
            dml = np.array([ml[bb == b][0] - vv[b][0] for b in common]); dli = np.array([li[bb == b][0] - vv[b][1] for b in common])
            vml = np.array([vv[b][0] for b in common]); vli = np.array([vv[b][1] for b in common])
            slope = np.polyfit(li, ml, 1)[0]
            print(f"{mode:8s} {cn}: build, 60 resamples: std ML {ml.std(ddof=1):.3f} / lin {li.std(ddof=1):.3f} = {ratio60:.3f} "
                  f"(bootstrap over the 60: 16-84% {np.percentile(rb, 16):.2f}-{np.percentile(rb, 84):.2f}); slope ML on lin {slope:.3f}; "
                  f"first 12 resamples: {ratio12:.3f}; the five blocks of 12: {', '.join(f'{x:.2f}' for x in sub)}", flush=True)
            print(f"            same 12 resamples, verifier's model + fitter: ratio {vml.std(ddof=1) / vli.std(ddof=1):.3f}; build - verifier per "
                  f"resample: ML {dml.mean():+.3f} (std {dml.std(ddof=1):.3f}), lin {dli.mean():+.3f} (std {dli.std(ddof=1):.3f})", flush=True)
    print(f"peak RSS {A.peak_rss_gb():.2f} GB")
