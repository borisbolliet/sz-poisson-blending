"""Step 3b arbitration / a08: claim S5 (the kernel's Monte Carlo error on N_exp, 0.21-0.23% from a 60-resample tile
bootstrap) against the even/odd half split (-1.08% at baseline, 2.5x the implied 0.43% scatter of a half split).

Production kernel, both noise levels, with b05's own construction (b05_kernel.build: measured fractions at the q' nodes,
bright-end model, credit limit, PAVA) and N_exp(q > 5) over the realistic halos of the 324 Q2 tiles (b05_kernel.n_exp):
  (1) NS random half splits of the 324 halo-free tiles: std of N_exp(half A) / N_exp(half B) - 1, to compare with
      2 x the claimed full-kernel error (two halves, each with twice the variance);
  (2) NBK2 more bootstrap resamples of the tiles (the build used 60): the full kernel's relative error on N_exp.
Usage: python a08_kernel_mc.py [NS] [NBK2]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
import b05_kernel as B5                            # noqa: E402

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NBK2 = int(sys.argv[2]) if len(sys.argv) > 2 else 100

if __name__ == "__main__":
    t00 = time.time()
    for mode in B.MODES:
        V, _ = B5.load("prod", mode)
        QA, K, *_ = B5.build(V, np.ones(V["k_q"].size, bool))
        Nf = B5.n_exp(B.Kernel(B.TH_K, QA, K), mode, "prod")[0]
        rng = np.random.default_rng(808 + B.MODES.index(mode))
        hs = []
        for s in range(NS):
            half = np.zeros(324, bool); half[rng.permutation(324)[:162]] = True
            m = half[V["pid"]]
            QAa, Ka, *_ = B5.build(V, m); QAb, Kb, *_ = B5.build(V, ~m)
            Na = B5.n_exp(B.Kernel(B.TH_K, QAa, Ka), mode, "prod")[0]; Nb = B5.n_exp(B.Kernel(B.TH_K, QAb, Kb), mode, "prod")[0]
            hs.append(Na / Nb - 1)
        hs = np.array(hs)
        bs = []
        for b in range(NBK2):
            wt = np.bincount(rng.integers(0, 324, 324), minlength=324)
            ii = np.repeat(np.arange(V["pid"].size), wt[V["pid"]])
            Vb = {k: v[ii] for k, v in V.items()}
            QAb, Kb, *_ = B5.build(Vb, np.ones(ii.size, bool))
            bs.append(B5.n_exp(B.Kernel(B.TH_K, QAb, Kb), mode, "prod")[0] / Nf - 1)
        bs = np.array(bs)
        ev = V["pid"] % 2 == 0
        QAe, Ke, *_ = B5.build(V, ev); QAo, Ko, *_ = B5.build(V, ~ev)
        eo = B5.n_exp(B.Kernel(B.TH_K, QAe, Ke), mode, "prod")[0] / B5.n_exp(B.Kernel(B.TH_K, QAo, Ko), mode, "prod")[0] - 1
        print(f"{mode}: N_exp(q > 5) full kernel {Nf:.1f}; {NS} random half splits: std of A/B - 1 = {100 * hs.std(ddof=1):.3f}% "
              f"(+- {100 * hs.std(ddof=1) / np.sqrt(2 * (NS - 1)):.3f}), i.e. full-kernel error {100 * hs.std(ddof=1) / 2:.3f}%; largest |A/B - 1| "
              f"{100 * np.abs(hs).max():.2f}%; even/odd split {100 * eo:+.3f}% = {eo / hs.std(ddof=1):+.2f} x the split std; {NBK2} bootstrap "
              f"resamples: relative std {100 * bs.std(ddof=1):.3f}% (+- {100 * bs.std(ddof=1) / np.sqrt(2 * (NBK2 - 1)):.3f}) [{time.time() - t00:.0f} s]",
              flush=True)
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {B.peak_rss_gb():.2f} GB", flush=True)
