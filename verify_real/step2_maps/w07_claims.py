"""w07: claim-by-claim comparison of the map route (w04: Delta_map with patch-bootstrap errors, linearised and ML
biases; w03: noise estimators; w06: Poisson realisations) with the build's step-2 claims.
'map lin' = linearised Fisher estimator on Delta_map (bootstrap 1 sigma); 'map ML' = ML fit on N0 + Delta_map;
'pull' = (map - build) / bootstrap sigma, with the build's value for the SAME primaries (q_A >= 0.5) where available.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/boris/GitHub/sz-poisson-blending"
OUT = os.path.join(HERE, "out")
lin = json.load(open(os.path.join(OUT, "w04_lin.json")))
fits = json.load(open(os.path.join(OUT, "w04_fits.json")))
dz = np.load(os.path.join(OUT, "w04_delta.npz"))
bfits = json.load(open(f"{ROOT}/data_real/step2/fits.json"))
f = open(os.path.join(HERE, "w07_claims.log"), "w")


def log(s=""):
    print(s, flush=True); f.write(s + "\n"); f.flush()


def L(nm, cn, sv="SO40", qc=5.0, p="S8"):
    return lin[f"{nm}|{qc}|{sv}|{cn}"][p], lin[f"{nm}|{qc}|{sv}|{cn}"].get(p + "_b05"), lin[f"{nm}|{qc}|{sv}|{cn}"].get(p + "_ball")


def ML(nm, cn, sv="SO40", qc=5.0):
    for r in fits:
        if r["setup"] == nm and r["config"] == cn and r["survey"] == sv and r["qcut"] == qc:
            return r


BMAP = {"baseline_f1": "baseline_f1", "goal_f1": "goal_f1", "white_f1": "white", "baseline_f0.5": "baseline_f0.5",
        "baseline_f2": "baseline_f2", "goal_f0.5": "goal_f0.5", "goal_f2": "goal_f2", "baseline_effi": "baseline_effi",
        "baseline_effii": "baseline_effii", "goal_effi": "goal_effi", "goal_effii": "goal_effii", "lowP_f1": "lowP_f1",
        "lowP_effi": "lowP_effi"}


def BML(nm, cn, sv="SO40", qc=5.0):
    s = BMAP.get(nm, nm)
    for r in bfits:
        if r["setup"] == s and r["config"] == cn and r["survey"] == sv and r["qcut"] == qc and r["route"] == "cat" \
                and r.get("start", "truth") == "truth":
            return r


def line(nm, cn, sv="SO40", qc=5.0, p="S8"):
    (v, e), b05, ball = L(nm, cn, sv, qc, p)
    m = ML(nm, cn, sv, qc)
    b = BML(nm, cn, sv, qc)
    ml = f"{m['b_' + p]:+.2f}" if m else "  -  "
    bm = f"{b['b_' + p]:+.2f}" if b else "  -  "
    pull = (v - b05) / e if b05 is not None else np.nan
    return (f"{nm:15s} {cn} {sv:4s} q{qc:.0f} {p:3s}: map lin {v:+.2f} +- {e:.2f}, map ML {ml} | build: ML {bm}, lin(all) "
            f"{ball if ball is None else f'{ball:+.2f}'}, lin(q_A>=0.5) {b05 if b05 is None else f'{b05:+.2f}'} | pull {pull:+.1f}"), v, e, (b05 if b05 is not None else np.nan)


def frac(nm, qc=5):
    D = dz[f"{nm}_q{qc}_D"]; Ns = dz[f"{nm}_q{qc}_Ns"]; Db = dz[f"{nm}_q{qc}_Dboot"].astype(float)
    return D, Ns, Db


log("[w07] map route vs the build's claims (C ids of the build's claim list)")
log("\n--- C1 / C2: config d, q > 5, S8 bias, SO40 and full sky, baseline / goal (fdom 1)")
for nm in ("baseline_f1", "goal_f1"):
    for sv in ("SO40", "full"):
        log("   " + line(nm, "d", sv)[0])
log("\n--- C3: old white filter (Q3 setup) through the map route, config d S8 (build +1.11 / +2.07; Q3 +1.11 / +2.08) "
    "and config a (build -3.95 / -6.24)")
for sv in ("SO40", "full"):
    for cn in ("d", "a"):
        log("   " + line("white_f1", cn, sv)[0])
log("\n--- C4: config a S8, baseline / goal, SO40 / full (build +1.33 / +1.85; +2.11 / +2.92)")
for nm in ("baseline_f1", "goal_f1"):
    for sv in ("SO40", "full"):
        log("   " + line(nm, "a", sv)[0])
log("\n--- C5: configs b, c S8, SO40 (build: baseline b +0.91, c +0.57; goal b +1.23, c +0.80)")
for nm in ("baseline_f1", "goal_f1"):
    for cn in ("b", "c"):
        log("   " + line(nm, cn, "SO40")[0])
log("\n--- C6 / C7: attribution rule fdom 0.5 / 1 / 2: S8 bias configs a, d, e")
for run in ("baseline", "goal"):
    for sv in ("SO40", "full"):
        for cn in ("a", "d", "e"):
            vals = [L(f"{run}_f{fd}", cn, sv)[0] for fd in ("0.5", "1", "2")]
            b05 = [L(f"{run}_f{fd}", cn, sv)[1] for fd in ("0.5", "1", "2")]
            bml = [BML(f"{run}_f{fd}", cn, sv) for fd in ("0.5", "1", "2")]
            log(f"   {run:8s} {cn} {sv:4s}: map lin " + " / ".join(f"{v:+.2f}+-{e:.2f}" for v, e in vals)
                + " | build ML " + " / ".join(f"{b['b_S8']:+.2f}" for b in bml)
                + " | build lin(q_A>=0.5) " + " / ".join(f"{b:+.2f}" for b in b05))
log("\n--- C8 / C9: blended - naive counts at q > 5 (full sky), Delta/N; z tilt; q shape")
ZED = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
zg = [np.nonzero((ZED[:-1] >= a - 1e-9) & (ZED[1:] <= b + 1e-9))[0] for a, b in ((0.01, 0.1), (2.0, 3.0))]
for nm in ("baseline_f1", "goal_f1", "baseline_f0.5", "baseline_f2", "goal_f0.5", "goal_f2", "white_f1",
           "baseline_effi", "baseline_effii", "goal_effi", "goal_effii", "lowP_f1", "lowP_effi"):
    D, Ns, Db = frac(nm)
    Nsb = Ns.sum()
    tot = D.sum() / Nsb; te = (Db.sum((1, 2)) / Nsb).std()
    zlo = D[zg[0]].sum() / Ns[zg[0]].sum(); zloe = (Db[:, zg[0]].sum((1, 2)) / Ns[zg[0]].sum()).std()
    zhi = D[zg[1]].sum() / Ns[zg[1]].sum(); zhie = (Db[:, zg[1]].sum((1, 2)) / Ns[zg[1]].sum()).std()
    q0 = D[:, 0].sum() / Ns[:, 0].sum(); q0e = (Db[:, :, 0].sum(1) / Ns[:, 0].sum()).std()
    qh = D[:, 6:].sum() / Ns[:, 6:].sum(); qhe = (Db[:, :, 6:].sum((1, 2)) / Ns[:, 6:].sum()).std()
    t = np.load(f"{ROOT}/data_real/step2/truth_{BMAP[nm]}.npz")
    B, BN = t["D05_q5"], t["Nn05_q5"]
    btot = B.sum() / BN.sum(); bz0 = B[zg[0]].sum() / BN[zg[0]].sum(); bz1 = B[zg[1]].sum() / BN[zg[1]].sum()
    bq0 = B[:, 0].sum() / BN[:, 0].sum(); bqh = B[:, 6:].sum() / BN[:, 6:].sum()
    log(f"   {nm:15s}: total map {100*tot:+.3f} +- {100*te:.3f}% vs build {100*btot:+.3f}% (pull {(tot-btot)/te:+.1f}) | "
        f"z<0.1 {100*zlo:+.2f}+-{100*zloe:.2f} (build {100*bz0:+.2f}) -> z 2-3 {100*zhi:+.2f}+-{100*zhie:.2f} (build {100*bz1:+.2f}) | "
        f"q 5-7.3 {100*q0:+.2f}+-{100*q0e:.2f} (build {100*bq0:+.2f}), q>47 {100*qh:+.2f}+-{100*qhe:.2f} (build {100*bqh:+.2f})")
log("\n--- C10: map-sigma variants (build r_i, r_ii): config a S8 SO40 (build -0.64, -0.83 baseline; -1.00, -1.38 goal); "
    "configs d, e SO40 / full")
for nm in ("baseline_effi", "baseline_effii", "goal_effi", "goal_effii"):
    log("   " + line(nm, "a", "SO40")[0])
    for sv in ("SO40", "full"):
        for cn in ("d", "e"):
            log("   " + line(nm, cn, sv)[0])
for nm in [k for k in sorted(set(k.split("|")[0] for k in lin)) if k.endswith(("_std", "_clip", "_rob"))]:
    log(f"   [analyst variant, r measured on the simulated maps (w03)] " + line(nm, "a", "SO40")[0])
    for cn in ("d", "e"):
        log(f"   [analyst variant] " + line(nm, cn, "SO40")[0])
        log(f"   [analyst variant] " + line(nm, cn, "full")[0])
    D, Ns, Db = frac(nm)
    log(f"   [analyst variant] {nm}: Delta/N {100*D.sum()/Ns.sum():+.3f} +- {100*(Db.sum((1,2))/Ns.sum()).std():.3f}%")
log("\n--- C13 / C14: config d nuisance shifts and goodness of fit (map ML on the central vector)")
for nm in ("baseline_f1", "goal_f1", "white_f1"):
    for sv in ("SO40", "full"):
        m = ML(nm, "d", sv); b = BML(nm, "d", sv)
        log(f"   {nm:12s} {sv:4s}: map ML Om {m['b_Om']:+.2f} s8 {m['b_s8']:+.2f} (1-b) {m['b_omb']:+.2f} alpha {m['b_alpha']:+.2f} "
            f"slnY {m['b_slnY']:+.2f} chi2min {m['dev']:.2f} power {m['power']:.3f} at bound {m['at_bound']} | build Om {b['b_Om']:+.2f} "
            f"s8 {b['b_s8']:+.2f} (1-b) {b['b_omb']:+.2f} alpha {b['b_alpha']:+.2f} slnY {b['b_slnY']:+.2f} chi2min {b['dev']:.2f} "
            f"power {b['power']:.3f} at bound {b['at_bound']}")
pw = {}
for r in fits:
    if r["setup"].startswith(("baseline", "goal")) and r["qcut"] == 5.0:
        k = (r["survey"], "a" if r["config"] == "a" else "b-e")
        pw[k] = max(pw.get(k, 0), r["power"])
log("   max goodness-of-fit power over the map MMF fits (fdom 0.5-2, map-sigma; q 5): " + ", ".join(f"{k[0]} {k[1]}: {v:.3f}" for k, v in sorted(pw.items())))
log("\n--- C15: q_cut 6, config a, SO40 (build +0.98 / +1.40; old filter -2.09)")
for nm in ("baseline_f1", "goal_f1", "white_f1"):
    (v5, e5), _, _ = L(nm, "a", "SO40", 5.0)
    (v6, e6), b6, _ = L(nm, "a", "SO40", 6.0)
    m6 = ML(nm, "a", "SO40", 6.0)
    log(f"   {nm:12s}: q5 lin {v5:+.2f} +- {e5:.2f} -> q6 lin {v6:+.2f} +- {e6:.2f} (ML {m6['b_S8'] if m6 else float('nan'):+.2f}; "
        f"build lin(q_A>=0.5) {b6:+.2f}); change {100*(v6/v5-1):+.0f}%")
log("\n--- C19: low-pressure population: Delta/N (above), config a SO40 / full (build +0.57 / +0.91), configs d-e")
for sv in ("SO40", "full"):
    for cn in ("a", "d", "e"):
        log("   " + line("lowP_f1", cn, sv)[0])
D, Ns, Db = frac("lowP_f1")
log(f"   low-pressure N_single (q_A >= 0.5, full sky, map) {Ns.sum():,.0f} vs build Nn05 "
    f"{np.load(f'{ROOT}/data_real/step2/truth_lowP_f1.npz')['Nn05_q5'].sum():,.0f}")
log("\n--- C20 / C22: white Delta total and naive totals (full sky, q > 5)")
for nm, ref in (("white_f1", "white"), ("baseline_f1", "baseline_f1"), ("goal_f1", "goal_f1")):
    D, Ns, Db = frac(nm)
    t = np.load(f"{ROOT}/data_real/step2/truth_{ref}.npz")
    log(f"   {nm:12s}: map N_single {Ns.sum():,.0f} vs build Nn05 {t['Nn05_q5'].sum():,.0f} ({Ns.sum()/t['Nn05_q5'].sum()-1:+.4f}); "
        f"map Delta {D.sum():+,.0f} +- {Db.sum((1,2)).std():,.0f} vs build D05 {t['D05_q5'].sum():+,.1f} (all primaries "
        f"{t['D_q5'].sum():+,.1f})")
q3 = np.load(f"{ROOT}/data_cosmo/truth_catalogue_v2.npz")
log(f"   Q3 (c11) white Delta {q3['D_q5'].sum():+,.1f}")
log("\n[w07] done")
