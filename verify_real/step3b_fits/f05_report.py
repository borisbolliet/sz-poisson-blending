"""f05 (step-3b verification, lens 'fits'): tables comparing my fits (f04) with the build's (fits.json, lin.json,
fits_analytic_cl.json; read as data).  Usage: f05_report.py   (reads out/f04_rows.jsonl)."""
import os, sys, json
os.environ["VF_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import f00lib as L

rows = [json.loads(l) for l in open(os.path.join(L.OUT, "f04_rows.jsonl"))]
BF = json.load(open(os.path.join(L.D3B, "fits.json")))
BL = json.load(open(os.path.join(L.D3B, "lin.json")))
BA = json.load(open(os.path.join(L.D3B, "fits_analytic_cl.json")))


def mine(kind, **kw):
    out = []
    for r in rows:
        t = r["task"]
        if t[0] != kind:
            continue
        if kind in ("ml", "self", "mlb"):
            d = dict(setup=t[1], mode=t[2], src=t[3], est=t[4], qcut=t[5], survey=t[6], config=t[7])
        elif kind == "an":
            d = dict(variant=t[1], mode=t[2], k=t[3], route=t[4], src=t[5], survey=t[6], config=t[7])
        else:
            d = dict(setup=t[1], mode=t[2], qcut=t[3], survey=t[4], config=t[5])
        if all(d.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def one(lst):
    return lst[0] if lst else None


def bfit(**kw):
    for r in BF:
        if r["route"] == "cat" and all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def blin(**kw):
    for r in BL:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


SETN = {"prod_none": "production, merge none (headline)", "szifi3000_none": "SZiFi defaults", "prod_fof10": "production, fof 10'"}
print("(1) ML S8 bias [sigma]: MINE (my model + fitter + my R_s) | build R_s with my model and fitter | BUILD (fits.json, R_s); "
      "build's tile-bootstrap error (lin.json); my bootstrap error (my R_s, linearised)")
diffs = []
for setup in SETN:
    for qc in (5.0, 6.0):
        for mode in L.MODES:
            for sv in ("SO40", "full"):
                cells = []
                for cfg in ("a", "d", "e"):
                    m = one(mine("ml", setup=setup, mode=mode, src="mine", est="Rs", qcut=qc, survey=sv, config=cfg))
                    mb = one(mine("ml", setup=setup, mode=mode, src="build", est="Rs", qcut=qc, survey=sv, config=cfg))
                    b = bfit(setup=setup, mode=mode, estimator="Rs", qcut=qc, survey=sv, config=cfg)
                    bl = blin(setup=setup, mode=mode, estimator="Rs", qcut=qc, survey=sv, config=cfg)
                    ml_ = one(mine("lin", setup=setup, mode=mode, qcut=qc, survey=sv, config=cfg))
                    if m is None or b is None:
                        cells.append(f"{cfg}: n/a")
                        continue
                    eb = bl["boot_S8"] if bl else np.nan
                    em = ml_["mine_Rs"]["boot_S8"] if ml_ else np.nan
                    s_mb = f"{mb['ml']['S8']:+.2f}" if mb else "  -  "
                    cells.append(f"{cfg}: {m['ml']['S8']:+.2f} | {s_mb} | {b['b_S8']:+.2f} (+-{eb:.2f}; mine +-{em:.2f})")
                    diffs.append(dict(setup=setup, mode=mode, qcut=qc, survey=sv, config=cfg, mine=m["ml"]["S8"], build=b["b_S8"],
                                      build_mymodel=mb["ml"]["S8"] if mb else np.nan, eb=eb, em=em, bound_m=m["at_bound"], bound_b=b["at_bound"]))
                print(f"  {SETN[setup]:36s} q>{qc:.0f} {mode:8s} {sv:4s}  " + ";  ".join(cells), flush=True)
D = [d for d in diffs]
for setup in SETN:
    dd = np.array([d["mine"] - d["build"] for d in D if d["setup"] == setup])
    if dd.size == 0:
        continue
    pull = np.array([(d["mine"] - d["build"]) / d["eb"] for d in D if d["setup"] == setup])
    print(f"  {setup}: mine - build over {dd.size} fits: max |diff| {np.abs(dd).max():.3f} sigma; in units of the build's bootstrap error: "
          f"max |.| {np.abs(pull).max():.2f}, mean {pull.mean():+.2f}")
    if setup == "prod_none":
        dm = np.array([d["build_mymodel"] - d["build"] for d in D if d["setup"] == setup])
        dr = np.array([d["mine"] - d["build_mymodel"] for d in D if d["setup"] == setup])
        print(f"     model + fitter (build R_s, my model - build): max |diff| {np.nanmax(np.abs(dm)):.3f}, mean {np.nanmean(dm):+.3f}; "
              f"estimator (my R_s - build R_s, both my model): max |diff| {np.nanmax(np.abs(dr)):.3f}, mean {np.nanmean(dr):+.3f}")
print("  parameters at a bound (mine): " + str([(d['setup'], d['mode'], d['qcut'], d['survey'], d['config'], d['bound_m']) for d in D if d['bound_m']]))
print("  parameters at a bound (build): " + str([(d['setup'], d['mode'], d['qcut'], d['survey'], d['config'], d['bound_b']) for d in D if d['bound_b']]))

print("\n(1b) Om / s8 bias [sigma], headline, q_cut 5, ML (mine | build)")
for mode in L.MODES:
    for sv in ("SO40", "full"):
        c = []
        for cfg in ("a", "d", "e"):
            m = one(mine("ml", setup="prod_none", mode=mode, src="mine", est="Rs", qcut=5.0, survey=sv, config=cfg))
            b = bfit(setup="prod_none", mode=mode, estimator="Rs", qcut=5.0, survey=sv, config=cfg)
            if m and b:
                c.append(f"{cfg}: {m['ml']['Om']:+.2f}/{m['ml']['s8']:+.2f} | {b['b_Om']:+.2f}/{b['b_s8']:+.2f}")
        print(f"  {mode:8s} {sv:4s}: " + ";  ".join(c))

print("\n(2) headline ML fits with each estimator, q_cut 5, S8 [sigma]: mine (my R, R_p, R_s; my model) | build (fits.json)")
for mode in L.MODES:
    for est, best in (("R", "R"), ("Rp", "Rp"), ("Rs", "Rs")):
        c = []
        for cfg in ("a", "d", "e"):
            for sv in ("SO40", "full"):
                m = one(mine("ml", setup="prod_none", mode=mode, src="mine", est=est, qcut=5.0, survey=sv, config=cfg))
                b = bfit(setup="prod_none", mode=mode, estimator=best, qcut=5.0, survey=sv, config=cfg)
                if m and b:
                    c.append(f"{cfg} {sv}: {m['ml']['S8']:+.2f} | {b['b_S8']:+.2f}")
        print(f"  {mode:8s} {est:2s}: " + ";  ".join(c))

print("\n(3) linearised S8 bias +- tile bootstrap, q_cut 5 (my model): my R_s | build R_s | my R (bilin) | my R_p  -- vs the build's lin.json (R_s, R, R_p)")
for setup in ("prod_none", "prod_none_q2", "szifi3000_none", "prod_fof10"):
    for mode in L.MODES:
        for sv in ("SO40", "full"):
            c = []
            for cfg in ("a", "d", "e"):
                m = one(mine("lin", setup=setup, mode=mode, qcut=5.0, survey=sv, config=cfg))
                if m is None:
                    continue
                b = {e: blin(setup=setup, mode=mode, estimator=e, qcut=5.0, survey=sv, config=cfg) for e in ("Rs", "R", "Rp")}
                c.append(f"{cfg}: {m['mine_Rs']['S8']:+.2f}+-{m['mine_Rs']['boot_S8']:.3f} | {m['build_Rs']['S8']:+.2f}+-{m['build_Rs']['boot_S8']:.3f} | "
                         f"{m['mine_R']['S8']:+.2f}+-{m['mine_R']['boot_S8']:.3f} | {m['mine_Rp']['S8']:+.2f}+-{m['mine_Rp']['boot_S8']:.3f}  "
                         f"[build {b['Rs']['lin_S8']:+.2f}+-{b['Rs']['boot_S8']:.3f}, R {b['R']['lin_S8']:+.2f}+-{b['R']['boot_S8']:.3f}, "
                         f"R_p {b['Rp']['lin_S8']:+.2f}+-{b['Rp']['boot_S8']:.3f}]")
            print(f"  {setup:15s} {mode:8s} {sv:4s} " + ";  ".join(c))

print("\n(4) p06b split of the linearised S8 bias (R_s, q_cut 5): amplitude + z tilt + q shape = sum; mine (my R_s) | build (lin.json)")
for setup in ("prod_none", "prod_fof10", "szifi3000_none"):
    for mode in L.MODES:
        for sv in ("SO40", "full"):
            for cfg in ("a", "d"):
                m = one(mine("lin", setup=setup, mode=mode, qcut=5.0, survey=sv, config=cfg))
                b = blin(setup=setup, mode=mode, estimator="Rs", qcut=5.0, survey=sv, config=cfg)
                if m is None or b is None:
                    continue
                p, q = m["mine_Rs"]["parts"], b["parts"]
                pb = m["build_Rs"]["parts"]
                print(f"  {setup:15s} {mode:8s} {sv:4s} {cfg}: mine {p['amplitude']:+.2f} {p['ztilt']:+.2f} {p['qshape']:+.2f} = {m['mine_Rs']['S8']:+.2f} | "
                      f"build R_s, my model {pb['amplitude']:+.2f} {pb['ztilt']:+.2f} {pb['qshape']:+.2f} = {m['build_Rs']['S8']:+.2f} | "
                      f"build {q['amplitude']:+.2f} {q['z tilt']:+.2f} {q['q shape']:+.2f} = {q['sum']:+.2f}")

print("\n(4b) nuisance shifts of the ML fit, config d, q_cut 5 [sigma]: (1-b) / alpha / slnY -- mine (my R_s) | build R_s with my model | build")
for setup in ("prod_none", "szifi3000_none", "prod_fof10"):
    for mode in L.MODES:
        for sv in ("SO40", "full"):
            m = one(mine("ml", setup=setup, mode=mode, src="mine", est="Rs", qcut=5.0, survey=sv, config="d"))
            mb = one(mine("ml", setup=setup, mode=mode, src="build", est="Rs", qcut=5.0, survey=sv, config="d"))
            b = bfit(setup=setup, mode=mode, estimator="Rs", qcut=5.0, survey=sv, config="d")
            if m and b:
                sb = f"{mb['ml']['omb']:+.2f}/{mb['ml']['alpha']:+.2f}/{mb['ml']['slnY']:+.2f}" if mb else "-"
                print(f"  {setup:15s} {mode:8s} {sv:4s}: {m['ml']['omb']:+.2f}/{m['ml']['alpha']:+.2f}/{m['ml']['slnY']:+.2f} | {sb} | "
                      f"{b['b_omb']:+.2f}/{b['b_alpha']:+.2f}/{b['b_slnY']:+.2f}; chi2min mine {m['dev']:.2f} vs build {b['dev']:.2f}")

print("\n(5) goodness-of-fit power (ML, R_s, q_cut 5), max over configs a, d, e: mine | build")
for setup in SETN:
    for mode in L.MODES:
        c = []
        for sv in ("SO40", "full"):
            pm = [one(mine("ml", setup=setup, mode=mode, src="mine", est="Rs", qcut=5.0, survey=sv, config=cfg)) for cfg in ("a", "d", "e")]
            pb = [bfit(setup=setup, mode=mode, estimator="Rs", qcut=5.0, survey=sv, config=cfg) for cfg in ("a", "d", "e")]
            if all(pm) and all(pb):
                c.append(f"{sv}: a {pm[0]['power']:.2f}|{pb[0]['power']:.2f}, d {pm[1]['power']:.2f}|{pb[1]['power']:.2f}, e {pm[2]['power']:.2f}|{pb[2]['power']:.2f}")
        print(f"  {setup:15s} {mode:8s} " + ";  ".join(c))

print("\n(6) ML vs linearised (mine, R_s fits): ratio ML / lin of the S8 bias for |bias| >= 0.1")
rr = [(r["task"], r["ml"]["S8"], r["lin"]["S8"]) for r in mine("ml", src="mine", est="Rs") if abs(r["ml"]["S8"]) >= 0.1]
rel = np.array([abs(a - b) / abs(a) for _, a, b in rr])
print(f"  {len(rr)} fits: median |ML - lin| / |ML| {np.median(rel):.3f}; > 10%: {int((rel > 0.1).sum())}; largest: "
      + str(sorted([(round(a / b, 2), t[1], t[2], t[5], t[6], t[7]) for t, a, b in rr], key=lambda x: -abs(x[0] - 1))[:6]))
fr = [r for r in mine("ml", src="mine", est="Rs") if "ml_from_truth" in r]
if fr:
    print(f"  ML started at the truth vs at the linearised point (headline d, e, q_cut 5): max |diff| "
          f"{max(abs(r['ml']['S8'] - r['ml_from_truth']['S8']) for r in fr):.4f} sigma over {len(fr)} fits")

print("\n(7) self route (my model fitted to its own expectation): max |bias| over parameters")
for r in mine("self"):
    print(f"  {r['task'][2]:8s} {r['task'][7]}: max |ml bias| {max(abs(v) for v in r['ml'].values()):.1e}, lin {max(abs(v) for v in r['lin'].values()):.1e}, "
          f"from truth {max(abs(v) for v in r['ml_from_truth'].values()):.1e}")

print("\n(8) analytic analyst Phi(sqrt(q'^2 + 3) - e), zero below q' = 2, q_cut 5: S8 bias [sigma] (i) blended = (ii) isolated + blending (i) - (ii)")
print("    mine: my model, my measured-kernel expectation, my R_s [build R_s in brackets] | build (fits_analytic_cl.json); measured-kernel analyst (ML, R_s): mine | build")
for variant in ("prod", "szifi3000"):
    for mode in L.MODES:
        for sv in ("SO40", "full"):
            c = []
            for cfg in ("a", "d", "e"):
                bi = one(mine("an", variant=variant, mode=mode, k=3.0, route="blended", src="mine", survey=sv, config=cfg))
                bb = one(mine("an", variant=variant, mode=mode, k=3.0, route="blended", src="build", survey=sv, config=cfg))
                ii = one(mine("an", variant=variant, mode=mode, k=3.0, route="isolated", src="mine", survey=sv, config=cfg))
                B_ = {r["route"]: r for r in BA if r["variant"] == variant and r["mode"] == mode and r["k"] == 3.0 and r["survey"] == sv and r["config"] == cfg}
                mk = one(mine("ml", setup=f"{variant}_none", mode=mode, src="mine", est="Rs", qcut=5.0, survey=sv, config=cfg))
                bk = bfit(setup=f"{variant}_none", mode=mode, estimator="Rs", qcut=5.0, survey=sv, config=cfg)
                if not (bi and ii and B_):
                    continue
                s_bb = f"[{bb['ml']['S8'] - ii['ml']['S8']:+.2f}]" if bb else ""
                c.append(f"{cfg}: {bi['ml']['S8']:+.2f}+-{bi.get('boot_S8', np.nan):.2f} = {ii['ml']['S8']:+.2f} + {bi['ml']['S8'] - ii['ml']['S8']:+.2f} {s_bb} | "
                         f"{B_['blended']['b_S8']:+.2f} = {B_['isolated']['b_S8']:+.2f} + {B_['blended']['b_S8'] - B_['isolated']['b_S8']:+.2f}; "
                         f"K-analyst {mk['ml']['S8'] if mk else np.nan:+.2f} | {bk['b_S8'] if bk else np.nan:+.2f}; "
                         f"power (i) {bi['power']:.2f}|{B_['blended']['power']:.2f}, bound {bi['at_bound']}")
            print(f"  {variant:9s} {mode:8s} {sv:4s} " + "\n" + "\n".join("        " + x for x in c))

print("\n(9) the model's kernel interpolation: ML S8 bias with my model using the build's bilinear kernel interpolation ('mlb') vs "
      "my default PCHIP-in-q' model ('ml'), headline, q_cut 5: data = build R_s | my R_s; build (fits.json)")
for mode in L.MODES:
    for sv in ("SO40", "full"):
        c = []
        for cfg in ("d", "e"):
            v = {}
            for kind in ("mlb", "ml"):
                for src in ("build", "mine"):
                    r = one(mine(kind, setup="prod_none", mode=mode, src=src, est="Rs", qcut=5.0, survey=sv, config=cfg))
                    v[(kind, src)] = r["ml"]["S8"] if r else np.nan
            b = bfit(setup="prod_none", mode=mode, estimator="Rs", qcut=5.0, survey=sv, config=cfg)
            c.append(f"{cfg}: build R_s bilin {v[('mlb', 'build')]:+.3f} / pchip {v[('ml', 'build')]:+.3f}; my R_s bilin {v[('mlb', 'mine')]:+.3f} / "
                     f"pchip {v[('ml', 'mine')]:+.3f}; build {b['b_S8']:+.3f}")
        print(f"  {mode:8s} {sv:4s} " + ";  ".join(c))

print("\n(10) Fisher sigma(S8) at the truth, my model (smooth-Tinker08 grid) / build (fits.json), headline, q_cut 5 and 6")
rat = {}
for r in mine("ml", setup="prod_none", src="build", est="Rs"):
    t = r["task"]
    b = bfit(setup="prod_none", mode=t[2], estimator="Rs", qcut=t[5], survey=t[6], config=t[7])
    if b:
        rat.setdefault(t[7], []).append(r["s_S8"] / b["s_S8"])
print("  " + "; ".join(f"config {c}: {min(v):.4f}-{max(v):.4f} ({len(v)} fits)" for c, v in sorted(rat.items())))
rr = [json.loads(l) for l in open(os.path.join(L.OUT, "f04_rows_roundedgrid_partial.jsonl"))]
rat2 = {}
for r in rr:
    t = r["task"]
    if t[0] != "ml":
        continue
    b = bfit(setup=t[1], mode=t[2], estimator="Rs", qcut=t[5], survey=t[6], config=t[7])
    if b:
        rat2.setdefault(t[7], []).append(r["s_S8"] / b["s_S8"])
print("  first run with f01's grid (colossus' rounded Delta_m in Tinker08, 0.005 Om spacing; superseded): "
      + "; ".join(f"config {c}: {min(v):.4f}-{max(v):.4f} ({len(v)} fits)" for c, v in sorted(rat2.items())))
