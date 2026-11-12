"""vc08: comparison tables of the biases (my fitter and model, my Delta and the build's Delta) against the build's
data_real/step2/fits.json and Q3's c12_fits_v2 logs.  Reads out/fits_mine.json (vc05) and out/delta_mine.npz (vc04).
"""
import os, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["VC_THREADS"] = "1"
import vclib as V
import numpy as np

M = json.load(open(f"{HERE}/out/fits_mine.json"))
B = json.load(open(f"{V.ROOT}/data_real/step2/fits.json"))
Bc = {(r["setup"], r["qcut"], r["survey"], r["config"]): r for r in B if r["route"] == "cat" and r["start"] == "truth"}
Bs = [r for r in B if r["route"] == "self"]
fits = {(r["setup"], r["qcut"], r["survey"], r["config"], r["data"]): r for r in M if r["kind"] == "fit"}
lins = {(r["setup"], r["qcut"], r["survey"], r["config"]): r for r in M if r["kind"] == "lin"}
nsky = len(np.load(f"{HERE}/out/delta_mine.npz")["seeds"])
P3 = ["Om", "s8", "S8"]

print(f"Direct-route Delta from {nsky} full skies.  Bias = (fit - truth) / Fisher sigma at the truth.")
print("\n=== (A) ML biases (S8 / Om / s8, in sigma): build | my fitter + my model on the BUILD's Delta | on MY Delta (+- MC) ===")
worst_b = 0.0; worst_m = []
for (s, qc, sv, cfg, dn), r in sorted(fits.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][3], kv[0][2])):
    if dn != "build":
        continue
    b = Bc.get((s, qc, sv, cfg))
    mm = fits.get((s, qc, sv, cfg, "mine"))
    if b is None or mm is None:
        continue
    db = max(abs(r["ml"][p] - b[f"b_{p}"]) for p in P3)
    worst_b = max(worst_b, db)
    dm = mm["ml"]["S8"] - b["b_S8"]
    worst_m.append((abs(dm) / max(mm["mc_err"]["S8"], 1e-9), s, qc, sv, cfg, dm, mm["mc_err"]["S8"]))
    print(f"q{qc:.0f} {sv:4s} {cfg} {s:14s}: build {b['b_S8']:+6.2f} / {b['b_Om']:+6.2f} / {b['b_s8']:+6.2f} | on build's "
          f"{r['ml']['S8']:+6.2f} / {r['ml']['Om']:+6.2f} / {r['ml']['s8']:+6.2f} (max diff {db:.3f}) | on mine "
          f"{mm['ml']['S8']:+6.2f} +- {mm['mc_err']['S8']:.2f} / {mm['ml']['Om']:+6.2f} +- {mm['mc_err']['Om']:.2f} / "
          f"{mm['ml']['s8']:+6.2f} +- {mm['mc_err']['s8']:.2f} | S8 mine - build {dm:+.2f} ({dm/mm['mc_err']['S8']:+.1f} MC sigma)"
          + (f" | nuisance build {', '.join(f'{p} {b[f'b_{p}']:+.2f}' for p in ('omb', 'alpha', 'slnY'))}; on build's "
             f"{', '.join(f'{p} {r['ml'][p]:+.2f}' for p in ('omb', 'alpha', 'slnY'))}; on mine "
             f"{', '.join(f'{p} {mm['ml'][p]:+.2f}' for p in ('omb', 'alpha', 'slnY'))}" if cfg == 'd' else ""))
print(f"max |my fitter/model on the build's Delta - build| over these fits (S8, Om, s8): {worst_b:.3f} sigma")
worst_m.sort(reverse=True)
print("largest |S8 (my Delta) - S8 (build)| in units of the MC error: " +
      "; ".join(f"q{w[2]:.0f} {w[3]} {w[4]} {w[1]}: {w[5]:+.2f} ({w[0]:.1f})" for w in worst_m[:5]))

print("\n=== (B) Linearised S8 bias, all setups / configs: build lin | mine (my J, F) on build's Delta | on my Delta +- MC ===")
rows = []
for (s, qc, sv, cfg), r in sorted(lins.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2], kv[0][3])):
    b = Bc.get((s, qc, sv, cfg))
    if b is None:
        continue
    lb, lm, e = r["lin_build"]["S8"], r["lin_mine"]["S8"], r["mc_err_S8"]
    rows.append((s, qc, sv, cfg, b["lin_S8"], lb, lm, e, b["b_S8"], b["s_S8"], r["s_S8"], r.get("power_build"), b.get("power"),
                 r.get("power_mine")))
cur = None
for x in rows:
    if (x[0], x[1]) != cur:
        cur = (x[0], x[1]); print(f"  [{x[0]} q{x[1]:.0f}]")
    print(f"     {x[2]:4s} {x[3]}: build lin {x[4]:+6.2f} (ML {x[8]:+6.2f}) | mine on build's {x[5]:+6.2f} | on mine {x[6]:+6.2f} +- {x[7]:.2f} "
          f"| (mine - build)/MC {(x[6]-x[8])/x[7]:+5.1f} | sigma(S8) build {x[9]:.4f} mine {x[10]:.4f} ({x[10]/x[9]-1:+.1%}) | "
          f"power build {x[12]:.3f}, mine(lin) on build's {x[11]:.3f}, on mine {x[13]:.3f}")
d1 = max(abs(x[5] - x[4]) for x in rows)
print(f"max |my linearised on the build's Delta - build's linearised| = {d1:.3f} sigma over {len(rows)} (setup, cut, survey, config)")
print(f"max |sigma(S8) mine / build - 1| = {max(abs(x[10]/x[9]-1) for x in rows):.3f}")
pulls = np.array([(x[6] - x[8]) / x[7] for x in rows])
print(f"(my Delta - build ML)/MC over all {len(rows)} linearised rows: mean {pulls.mean():+.2f}, rms {np.sqrt(np.mean(pulls**2)):.2f}, "
      f"max |.| {np.abs(pulls).max():.1f}  (rows are strongly correlated: same skies)")

print("\n=== (C) Fisher sigmas (S8, Om, s8) at the truth: build vs mine, configs a and d, q5 ===")
for (s, qc, sv, cfg, dn), r in sorted(fits.items()):
    if dn != "build" or qc != 5.0:
        continue
    b = Bc[(s, qc, sv, cfg)]
    print(f"  {s:14s} {sv:4s} {cfg}: S8 {b['s_S8']:.5f} vs {r['s_S8']:.5f} ({r['s_S8']/b['s_S8']-1:+.2%}); Om {b['s_Om']:.5f} vs "
          f"{r['s']['Om']:.5f} ({r['s']['Om']/b['s_Om']-1:+.2%}); s8 {b['s_s8']:.5f} vs {r['s']['s8']:.5f} ({r['s']['s8']/b['s_s8']-1:+.2%})")

print("\n=== (D) self route ===")
mx = max(abs(v) for (s, qc, sv, cfg, dn), r in fits.items() if dn == "self" for v in r["ml"].values())
mo = max(abs(v) for (s, qc, sv, cfg, dn), r in fits.items() if dn == "self" for v in r["ml_offset_start"].values())
print(f"  my model fitted to its own expectation: max |bias| {mx:.1e} sigma (from the truth), {mo:.1e} sigma (from 2 sigma off), "
      f"{sum(1 for k in fits if k[4] == 'self')} fits")
print(f"  build's self route (fits.json): max |bias| {max(abs(r[f'b_{p}']) for r in Bs for p in r['free']):.1e} sigma over {len(Bs)} fits")
for (s, qc, sv, cfg, dn), r in sorted(fits.items()):
    if dn == "selfN0":
        print(f"  my model fitted to the BUILD's naive N0 ({s}, {sv}, {cfg}): S8 {r['ml']['S8']:+.3f}, Om {r['ml']['Om']:+.3f}, "
              f"s8 {r['ml']['s8']:+.3f} sigma; chi2_min {r['dev']:.3f}")

print("\n=== (E) Q3's own data vector (truth_catalogue_v2) with my white model + fitter vs the c12_fits_v2 logs ===")
c12 = {}
for fn in ("c12_fits_v2_q5.log", "c12_fits_v2_q6.log"):
    for ln in open(f"{V.ROOT}/analysis/cosmo/bias/{fn}"):
        if " cat      :" in ln:
            t = ln.split()
            vals = {k: float(re.search(rf"d{k}=[+-][0-9.]+ \(([+-][0-9.]+)s", ln).group(1)) for k in ("Om", "s8", "S8")}
            c12[(float(t[0][1:]), t[1], t[2])] = vals
wq = 0.0
for (s, qc, sv, cfg, dn), r in sorted(fits.items()):
    if dn != "Q3":
        continue
    ref = c12.get((qc, sv, cfg))
    if ref is None:
        continue
    d = max(abs(r["ml"][p] - ref[p]) for p in P3)
    wq = max(wq, d)
    print(f"  q{qc:.0f} {sv:4s} {cfg}: c12 S8/Om/s8 {ref['S8']:+.2f} / {ref['Om']:+.2f} / {ref['s8']:+.2f} | mine {r['ml']['S8']:+.2f} / "
          f"{r['ml']['Om']:+.2f} / {r['ml']['s8']:+.2f} | max |diff| {d:.2f}" + (f"  [at bound: {r['at_bound']}]" if r["at_bound"] else ""))
print(f"  max |mine - c12| = {wq:.2f} sigma")

print("\n=== (F) goodness-of-fit power (ML deviance; configs a, d; build vs mine on the build's / my Delta) ===")
for (s, qc, sv, cfg, dn), r in sorted(fits.items()):
    if dn != "build" or qc != 5.0:
        continue
    b = Bc[(s, qc, sv, cfg)]; mm = fits[(s, qc, sv, cfg, "mine")]
    print(f"  {s:14s} {sv:4s} {cfg}: chi2_min build {b['dev']:.2f} (power {b['power']:.3f}) | mine on build's {r['dev']:.2f} ({r['power']:.3f}) "
          f"| on mine {mm['dev']:.2f} ({mm['power']:.3f})")

print("\n=== (G) linearised decomposition of the S8 bias, SO40, q5 (my J, F): amplitude + z tilt + q shape = sum ===")
for r in sorted((r for r in M if r["kind"] == "decomp"), key=lambda r: (r["setup"], r["config"])):
    for dn in ("build", "mine"):
        d = r[dn]
        print(f"  {r['setup']:14s} {r['config']} on {dn:5s} Delta: {d['amplitude']['S8']:+.2f} + {d['ztilt']['S8']:+.2f} + "
              f"{d['qshape']['S8']:+.2f} = {d['sum']['S8']:+.2f}")

print("\n=== (H) the build's own Q3 reproduction (fits.json setup Q3_c11, route cat) vs the c12_fits_v2 logs (data only) ===")
wq = 0.0
for r in B:
    if r["setup"] == "Q3_c11" and r["route"] == "cat":
        ref = c12.get((r["qcut"], r["survey"], r["config"]))
        if ref is None:
            continue
        wq = max(wq, abs(round(r["b_S8"], 2) - ref["S8"]))
print(f"  max |b_S8 (fits.json, rounded to 0.01) - c12 log| = {wq:.3f} sigma over {sum(1 for r in B if r['setup']=='Q3_c11' and r['route']=='cat')} fits")

print("\n=== (I) the build's compute / memory logs (read as data): totals recomputed from p10_compute.log rows; p03 / p03b "
      "time -l lines ===")
L = f"{V.ROOT}/analysis/realistic/step2"
w = c = 0.0; n = 0
for ln in open(f"{L}/p10_compute.log"):
    t = ln.split()
    if len(t) == 5 and t[0].startswith("p") and t[0][1:3].isdigit():
        w += float(t[1]); c += float(t[2]); n += 1
print(f"  {n} rows: wall {w:.1f} s = {w/60:.1f} min; cpu user {c:.1f} s = {c/3600:.2f} h")
for fn in ("p03_kernels.log", "p03b_memcheck.log"):
    rss = fp = None
    for ln in open(f"{L}/{fn}"):
        if "maximum resident set size" in ln:
            rss = int(ln.split()[0])
        if "peak memory footprint" in ln:
            fp = int(ln.split()[0])
    print(f"  {fn}: max RSS {rss/1e9:.2f} GB, peak footprint {fp/1e9:.2f} GB")
