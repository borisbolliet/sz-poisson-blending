"""vc03: my naive count model (vcmodel.CountModel: own HMF grid and integration, MMF sigma_Y) vs
  (1) the build's naive model N0 (data_real/step2/truth_<setup>.npz, read as data) bin by bin and in total;
  (2) step 1's data_real/expected_detections.json totals (szsim 1201x1201 HMF table, Gauss-Hermite scatter);
  (3) smoothness: mu along Om, s8 and slnY on fine steps across grid nodes (second differences);
  (4) timing of one model evaluation.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import vclib as V
import vcmodel as VM
import numpy as np

t0 = time.time()
D2 = f"{V.ROOT}/data_real/step2"
ed = json.load(open(f"{V.ROOT}/data_real/expected_detections.json"))
rth, RT = V.r_tables()
SETUPS = {"white": ("white", None, V.FID), "baseline_f1": ("baseline", None, V.FID), "goal_f1": ("goal", None, V.FID),
          "baseline_effi": ("baseline", (rth, RT[1]), V.FID), "baseline_effii": ("baseline", (rth, RT[2]), V.FID),
          "goal_effi": ("goal", (rth, RT[3]), V.FID), "goal_effii": ("goal", (rth, RT[4]), V.FID),
          "lowP_f1": ("baseline", None, V.FID_LOW), "lowP_effi": ("baseline", (rth, RT[5]), V.FID_LOW)}
EDK = {"white": "white (old)", "baseline_f1": "SO baseline lmin100", "goal_f1": "SO goal lmin100"}
print("(1)+(2) naive counts, full sky: mine vs the build's N0 (p05) and step 1's expected_detections")
for name, (filt, r, truth) in SETUPS.items():
    for qc in (5.0, 6.0):
        M = VM.CountModel(filt, qc, r)
        mu = M.counts(truth)
        N0 = np.load(f"{D2}/truth_{name}.npz")[f"N0_q{qc:.0f}"]
        rel = mu / N0 - 1
        big = N0 > 10
        pull = (mu - N0) / np.sqrt(np.maximum(N0, 1e-9))
        s = (f"  [{name:14s}] q>{qc:.0f}: mine {mu.sum():12,.1f} | build N0 {N0.sum():12,.1f} (rel {mu.sum()/N0.sum()-1:+.2e}); "
             f"per bin (N0 > 10) max |rel| {np.abs(rel[big]).max():.1e}, max |diff|/sqrt(N0) {np.abs(pull).max():.3f} (full sky)")
        if name in EDK:
            ref = ed[EDK[name]][int(qc) - 4]
            s += f"; expected_detections {ref:,.1f} (rel {mu.sum()/ref-1:+.2e})"
        print(s, flush=True)
        if qc == 5.0 and name in ("baseline_f1", "white"):
            print("      per z bin mine/build - 1: " + " ".join(f"{v:+.1e}" for v in mu.sum(1) / N0.sum(1) - 1))
            print("      per q bin mine/build - 1: " + " ".join(f"{v:+.1e}" for v in mu.sum(0) / N0.sum(0) - 1))
print(f"  low-pressure total q>5 (mine) vs the build's quoted MMFModel(one_minus_b=0.6).expected_detections 29,844.3: see row lowP_f1")

print("\n(3) smoothness of mu(theta) (baseline, q_cut 5): relative second differences of the total and of the largest bin")
M = VM.CountModel("baseline", 5.0)
for k, h, n in ((0, 0.0005, 41), (1, 0.0005, 41), (4, 0.0005, 41), (3, 0.001, 21)):
    xs = V.FID[k] + h * (np.arange(n) - n // 2)
    tot = []; b0 = []
    for x in xs:
        th = V.FID.copy(); th[k] = x
        c = M.counts(th); tot.append(c.sum()); b0.append(c[5, 0])
    tot = np.array(tot); b0 = np.array(b0)
    d2t = np.abs(np.diff(tot, 2)) / tot[1:-1]; d2b = np.abs(np.diff(b0, 2)) / b0[1:-1]
    d1t = np.diff(tot) / tot[1:]
    print(f"  {V.PN[k]:5s} step {h}: max |2nd diff|/N: total {d2t.max():.1e}, bin (z 0.5-0.6, q 5-7.3) {d2b.max():.1e}; "
          f"1st diff/N range {d1t.min():+.3e} .. {d1t.max():+.3e}")
ts = time.time()
for i in range(20):
    th = V.FID.copy(); th[1] += 1e-4 * i
    M.counts(th)
print(f"\n(4) one model evaluation: {(time.time()-ts)/20*1000:.0f} ms; total {time.time()-t0:.0f}s")
