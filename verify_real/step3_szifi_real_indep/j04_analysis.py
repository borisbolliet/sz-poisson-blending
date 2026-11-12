"""j04: step 3a's SZiFi claims 1, 3, 4, 5 and 6, recomputed by the verifier (lens 'indep') with its own code from
  (b) the build's stored SZiFi runs (data_real/step3_szifi_real/runs, read as data), and
  (c) the verifier's own SZiFi runs on 6 new tiles per noise level (out/runs, made with jsz_run.py).
r = sigma_SZiFi / sigma_analytic with the verifier's own sigma_analytic of SZiFi's filter (j03); the production finder's
r = sigma / sigma_Y (step 1's MMF, j01).  Interior: >= 9' from the edges (jlib).  Production-finder and step-3a emulation
outputs are read from data_real/step3 (tiles/, noise_only/, arbiter/a01_*) and, for the new tiles, from j01.
Usage: .venv/bin/python -I -B j04_analysis.py
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np
import jlib as L

t0 = time.time()
ref = L.load(os.path.join(L.OUT, "j01_ref.npz"))
ana = L.load(os.path.join(L.OUT, "j03_analytic.npz"))
anaB = L.load(os.path.join(L.ROOT, "data_real", "step3_szifi_real", "analytic_sigma.npz"))
NEW = [int(p) for p in ref["new_pids"]]
SIGY = {m: ref[f"sigY_{m}"] for m in L.MODES}
SANA = {(c, m): ana[f"sigana_{m}"] for c in ("S0", "S1", "S2") for m in L.MODES}
for m in L.MODES:   # S3 / S3i (l_max,1d = 10000): the build's z04 value, read as data (not recomputed here)
    SANA[("S3i", m)] = SANA[("S3it0", m)] = anaB[f"{m}_S3"]
A01 = {}


def a01(kind, pid):
    key = (kind, pid)
    if key not in A01:
        A01[key] = L.load(os.path.join(L.D3, "arbiter", f"a01_{kind}_{pid:03d}.npz"))
    return A01[key]


CLAIM = {("S1", "baseline"): (1.135, 1.119, 1.050), ("S1", "goal"): (1.181, 1.172, 1.078),
         ("S2", "baseline"): (1.019, 1.044, 1.026), ("S2", "goal"): (1.038, 1.075, 1.046),
         ("prod", "baseline"): (1.039, 1.050, 1.024), ("prod", "goal"): (1.053, 1.067, 1.035)}
BUILD_R1 = {("S1", "baseline"): (1.168, 1.142, 1.057), ("S1", "goal"): (1.219, 1.198, 1.088),
            ("S2", "baseline"): (1.026, 1.051, 1.030), ("S2", "goal"): (1.050, 1.083, 1.049),
            ("S3i", "baseline"): (1.043, 1.066, 1.058), ("S3i", "goal"): (1.071, 1.105, 1.085)}
SUMMARY = {}


def real_block(src, cfg, mode, pids, emul=None):
    """Iteration 1 (and 0) of one configuration on realistic tiles.  emul: 'a01' (stored arbiter emulation on the a01
    tiles) or 'j01' (the verifier's emulation on its new tiles) or None."""
    R = {k: [] for k in ("r0", "r1", "rF0", "rF1", "f", "n0", "n1", "nP1", "spec_lo", "spec_hi", "big_s", "big_f",
                         "big_s_int", "big_f_int", "nmask", "rE1", "fE", "nE0", "nE1", "spec_lo_E", "spec_hi_E")}
    for pid in pids:
        r = L.load(L.run_path(src, cfg, mode, "real", pid))
        st = L.prod_tile(pid)
        sa = SANA[(cfg, mode)]
        R["r0"].append(L.szifi_sigma(r, 0) / sa)
        R["rF0"].append(st[f"{mode}_it0_sig"] / SIGY[mode])
        R["rF1"].append(st[f"{mode}_it1_sig"] / SIGY[mode])
        c0 = L.szifi_cat(r, 0)
        fc0 = L.finder_cat(st, f"{mode}_", 0)
        fc1 = L.finder_cat(st, f"{mode}_", 1)
        R["n0"].append(L.n_int(c0, 5.0))
        R["nP1"].append(L.n_int(fc1, 5.0))
        big = L.TH[c0["jt"]] >= 7.2
        R["big_s"].append(int(((c0["q"] > 4) & big).sum()))
        R["big_f"].append(int(((fc0["q"] > 4) & (L.TH[fc0["jt"]] >= 7.2)).sum()))
        R["big_s_int"].append(int(((c0["q"] > 4) & big & L.inside(c0)).sum()))
        R["big_f_int"].append(int(((fc0["q"] > 4) & (L.TH[fc0["jt"]] >= 7.2) & L.inside(fc0)).sum()))
        R["nmask"].append(int((c0["q"] > 4).sum()))
        if int(r["n_it"]) > 1:
            R["r1"].append(L.szifi_sigma(r, 1) / sa)
            c1 = L.szifi_cat(r, 1)
            R["n1"].append(L.n_int(c1, 5.0))
            f = r["fsky_rec"][0] if "fsky_rec" in r else r["rec_fsky"][0]
            R["f"].append(float(f))
            ell = r["spec0_ell"]
            R["spec_lo"].append(L.auto_ratio(r["spec1"], r["spec0"], ell, 100, 3000))
            R["spec_hi"].append(L.auto_ratio(r["spec1"], r["spec0"], ell, 3000, 20001))
        if emul == "a01":
            name = {"S1": "szifi", "S2": "szifi_l12000"}.get(cfg)
            if name:
                e = a01("real", pid)
                R["rE1"].append(e[f"{mode}_{name}_sig1"] / SIGY[mode])
                R["fE"].append(float(e[f"{mode}_{name}_fsky"]))
                R["nE0"].append(float(e[f"{mode}_{name}_n0"][2]))
                R["nE1"].append(float(e[f"{mode}_{name}_n1"][2]))
        elif emul == "j01":
            name = {"S1": "szifi", "S2": "szifi_l12000"}.get(cfg)
            if name:
                p = f"real_{mode}_{pid:03d}_{name}_"
                R["rE1"].append(ref[p + "it1_sig"] / SIGY[mode])
                R["fE"].append(float(ref[p + "fsky"]))
                R["nE0"].append(float((ref[p + "it0_q"] > 5).sum()))
                R["nE1"].append(float((ref[p + "it1_q"] > 5).sum()))
                # emulation's binned spectra (finder bins of width 72, bin centres (b + 1/2) 72)
                C0, C1 = ref[p + "it0_C"], ref[p + "it1_C"]
                lc = (np.arange(C0.shape[0]) + 0.5) * 72.0
                ok = np.einsum("bii->b", C0) > 0
                R["spec_lo_E"].append(L.auto_ratio(C1[ok], C0[ok], lc[ok], 100, 3000))
                R["spec_hi_E"].append(L.auto_ratio(C1[ok], C0[ok], lc[ok], 3000, 20001))
    return {k: np.array(v) for k, v in R.items()}


def print_real(src, cfg, mode, pids, R):
    n = len(pids)
    print(f"\n  ---- {cfg} {mode}, {src} runs, {n} realistic tiles {pids if n <= 10 else ''}")
    m0, s0, e0 = L.ms(R["r0"])
    mF0, _, _ = L.ms(R["rF0"])
    d0 = R["r0"] - R["rF0"]
    md0, sd0, _ = L.ms(d0)
    print(f"    r it0, SZiFi (0.5 / 4.45 / 15'):  {L.fmt(m0[[0, 9, 14]])}  (tile sd {L.fmt(s0[[0, 9, 14]])})")
    print(f"    r it0, production finder:         {L.fmt(mF0[[0, 9, 14]])}")
    print(f"    r_SZiFi - r_finder at it0: {md0[0]:+.4f} (tile sd {sd0[0]:.4f}) at 0.5'; largest r_SZiFi/r_finder - 1 above 2': "
          f"{np.nanmax(np.nanmean(R['r0'] / R['rF0'], 0)[L.TH > 2.0] - 1):+.4f}")
    out = {"r0": m0, "d0": md0, "n": n}
    if R["r1"].size:
        m1, s1, e1 = L.ms(R["r1"])
        mP1, sP1, _ = L.ms(R["rF1"])
        print(f"    r it1, SZiFi (0.5 / 4.45 / 15'):  {L.fmt(m1[[0, 9, 14]])}  (tile sd {L.fmt(s1[[0, 9, 14]], '{:.4f}')}; "
              f"error of the mean {L.fmt(e1[[0, 9, 14]], '{:.4f}')})")
        if (cfg, mode) in CLAIM:
            print(f"       step 3a's claim (a01 emulation, 30 tiles): {CLAIM[(cfg, mode)]}; the build's value: {BUILD_R1.get((cfg, mode))}")
        print(f"    r it1, production finder (stored): {L.fmt(mP1[[0, 9, 14]])} (tile sd {L.fmt(sP1[[0, 9, 14]])}); "
              f"claim {CLAIM[('prod', mode)]}")
        out.update(r1=m1, s1=s1, e1=e1, rP1=mP1, sP1=sP1)
        if R["rE1"].size:
            mE, sE, _ = L.ms(R["rE1"])
            dd = R["r1"] - R["rE1"]
            mdd, sdd, edd = L.ms(dd)
            print(f"    emulation of the same iteration on the same tiles, r it1: {L.fmt(mE[[0, 9, 14]])}; "
                  f"SZiFi - emulation (paired): {L.fmt(mdd[[0, 9, 14]], '{:+.4f}')} (tile sd {L.fmt(sdd[[0, 9, 14]])})")
            df = R["f"] - R["fE"]
            cc = np.corrcoef(dd[:, 0], df)[0, 1] if len(df) > 2 else np.nan
            print(f"    unmasked fraction f: SZiFi {np.mean(R['f']):.3f} (range {np.min(R['f']):.3f}-{np.max(R['f']):.3f}); "
                  f"emulation {np.mean(R['fE']):.3f}; paired SZiFi - emulation {np.mean(df):+.3f} +- {np.std(df, ddof=1):.3f}; "
                  f"correlation of (r_SZiFi - r_emul) at 0.5' with (f_SZiFi - f_emul): {cc:+.2f}")
            out.update(rE1=mE, dE=mdd, sdE=sdd, eE=edd, fE=np.mean(R["fE"]), corr=cc)
            nE0, nE1 = R["nE0"].sum(), R["nE1"].sum()
            print(f"    emulation, whole periodic tile: q > 5 per tile {nE0 / n:.1f} -> {nE1 / n:.1f} ({100 * (nE1 / nE0 - 1):+.1f}%)")
            if R["spec_lo_E"].size:
                print(f"    emulation's auto spectra after / before, l 100-3000 / 3000-20000: {np.mean(R['spec_lo_E']):.3f} / {np.mean(R['spec_hi_E']):.3f}")
        else:
            print(f"    unmasked fraction f: SZiFi {np.mean(R['f']):.3f} (range {np.min(R['f']):.3f}-{np.max(R['f']):.3f})")
        print(f"    auto spectra after / before (6 channels, all bins), l 100-3000 / 3000-20000: {np.mean(R['spec_lo']):.3f} / "
              f"{np.mean(R['spec_hi']):.3f} (tile range {np.min(R['spec_hi']):.3f}-{np.max(R['spec_hi']):.3f})")
        rat, err = L.boot_ratio(R["n1"], R["n0"])
        pt = 100 * (R["n1"] / R["n0"] - 1)
        cf = np.corrcoef(pt, R["f"])[0, 1] if pt.size > 2 else np.nan
        print(f"    q > 5 per tile in the interior, it0 -> it1: {R['n0'].mean():.1f} -> {R['n1'].mean():.1f} "
              f"({100 * (rat - 1):+.1f} +- {100 * err:.1f}%, tile bootstrap); per tile {pt.mean():+.1f} (sd {np.std(pt, ddof=1) if pt.size > 1 else np.nan:.1f}, "
              f"range {pt.min():+.1f} to {pt.max():+.1f}; {int((pt > 0).sum())} of {pt.size} tiles gain); correlation with f {cf:+.2f}")
        out.update(f=np.mean(R["f"]), spec_hi=np.mean(R["spec_hi"]), spec_lo=np.mean(R["spec_lo"]), dn=(rat - 1, err),
                   n0=R["n0"], n1=R["n1"], nP1=R["nP1"])
        rp, ep = L.boot_ratio(R["n1"], R["nP1"])
        print(f"    final q > 5 count, SZiFi / production finder (interior, same tiles): {rp:.4f} +- {ep:.4f} "
              f"({R['n1'].sum()} / {R['nP1'].sum()})")
        out["vsprod"] = (rp, ep)
    print(f"    q > 4 detections with theta >= 7.2' per tile at it0: SZiFi {R['big_s'].mean():.1f} (interior {R['big_s_int'].mean():.1f}), "
          f"production finder {R['big_f'].mean():.1f} (interior {R['big_f_int'].mean():.1f}); SZiFi q > 4 detections masked per tile "
          f"{R['nmask'].mean():.1f}")
    out["big"] = (R["big_s_int"].mean(), R["big_f_int"].mean())
    SUMMARY[(src, cfg, mode, "real")] = out
    return out


def noise_block(src, cfg, mode, pids):
    sa = SANA[(cfg, mode)]
    s0, s1, sP1, nf, nf0, nfP = [], [], [], [], [], []
    fS, fE, sE = [], [], []
    name = {"S1": "szifi", "S2": "szifi_l12000"}.get(cfg)
    for pid in pids:
        r = L.load(L.run_path(src, cfg, mode, "noise", pid))
        s0.append(L.szifi_sigma(r, 0) / sa)
        if int(r["n_it"]) > 1:
            fS.append(float((r["fsky_rec"] if "fsky_rec" in r else r["rec_fsky"])[0]))
        if name and src == "build":
            e = a01("noise", pid)
            sE.append(e[f"{mode}_{name}_sig1"] / e[f"{mode}_sig_ana"])
            fE.append(float(e[f"{mode}_{name}_fsky"]))
        elif name:
            pre = f"noise_{mode}_{pid:03d}_{name}_"
            sE.append(ref[pre + "it1_sig"] / SIGY[mode])
            fE.append(float(ref[pre + "fsky"]))
        pn = L.prod_noise(pid)
        sP1.append(pn[f"{mode}_whitened_it1_sig"] / SIGY[mode])
        c0 = L.szifi_cat(r, 0)
        nf0.append([L.n_int(c0, t) for t in (4.0, 4.5, 5.0)])
        cP = L.finder_cat(pn, f"{mode}_whitened_", 1)
        nfP.append([L.n_int(cP, t) for t in (4.0, 4.5, 5.0)])
        if int(r["n_it"]) > 1:
            s1.append(L.szifi_sigma(r, 1) / sa)
            c1 = L.szifi_cat(r, 1)
            nf.append([L.n_int(c1, t) for t in (4.0, 4.5, 5.0)])
        else:
            nf.append([L.n_int(c0, t) for t in (4.0, 4.5, 5.0)])
    n = len(pids)
    s0, sP1 = np.array(s0), np.array(sP1)
    nf, nf0, nfP = np.array(nf).sum(0), np.array(nf0).sum(0), np.array(nfP).sum(0)
    area = n * L.AREA
    print(f"\n  ---- {cfg} {mode}, {src} runs, {n} noise-only tiles {pids}")
    print(f"    sigma/analytic it0 at 0.50 1.04 2.15 4.45 9.23 15.00': {L.fmt(np.nanmean(s0, 0)[L.SC6])}")
    out = {"n": n}
    if s1:
        s1 = np.array(s1)
        m1, sd1, e1 = L.ms(s1)
        print(f"    sigma/analytic it1:                                  {L.fmt(m1[L.SC6])} (tile sd {L.fmt(sd1[L.SC6])})")
        out.update(s1=m1, sd1=sd1, e1=e1)
    if sE and len(s1) == len(sE):
        dd = np.asarray(s1) - np.asarray(sE)
        md, sdd, ed = L.ms(dd)
        print(f"    emulation on the same tiles ({'a01' if src == 'build' else 'j01'}): sigma/analytic it1 {L.fmt(np.mean(sE, 0)[L.SC6])}; "
              f"SZiFi - emulation (paired) at 0.5 / 15': {md[0]:+.4f} +- {ed[0]:.4f} / {md[14]:+.4f} +- {ed[14]:.4f}; "
              f"unmasked fraction f SZiFi {np.mean(fS):.3f}, emulation {np.mean(fE):.3f}")
        out.update(dE=md, eE=ed)
    elif fS:
        print(f"    unmasked fraction f: SZiFi {np.mean(fS):.3f}")
    print(f"    false detections per 100 deg^2 of interior, q > 4 / 4.5 / 5, final: {L.fmt(100 * nf / area, '{:.2f}')} "
          f"[{nf[0]}, {nf[1]}, {nf[2]}]; it0 {L.fmt(100 * nf0 / area, '{:.2f}')}; interior {area:.0f} deg^2")
    mP, _, _ = L.ms(sP1)
    print(f"    production finder, same tiles: sigma/analytic it1 {L.fmt(mP[L.SC6])}; false q > 4 / 4.5 / 5: "
          f"{L.fmt(100 * nfP / area, '{:.2f}')} [{nfP[0]}, {nfP[1]}, {nfP[2]}]")
    out.update(nf=nf, area=area, nfP=nfP)
    SUMMARY[(src, cfg, mode, "noise")] = out
    return out


def it0_block(src, mode, pids, cfg="S1"):
    """Claim 1: SZiFi's iteration 0 vs the production finder's (stored), matched both ways in the interior."""
    print(f"\n  ---- iteration 0, {cfg} ({src} runs) vs production finder, {mode}, {len(pids)} realistic tiles")
    acc = {}
    offs = []
    for pid in pids:
        r = L.load(L.run_path(src, cfg, mode, "real", pid))
        st = L.prod_tile(pid)
        S = L.szifi_cat(r, 0)
        Fc = L.finder_cat(st, f"{mode}_", 0)
        S_raw = L.szifi_cat(r, 0, shift=0.0)
        # position offset without correction: finder q > 6 at >= 60' from the edges, nearest SZiFi within 1.5'
        deep = (L.edge_dist(Fc["py"], Fc["px"]) >= 60) & (Fc["q"] > 6)
        Fd = {k: v[deep] for k, v in Fc.items()}
        Fd_all = dict(Fd)
        nn, ok, dq, dj, off = L.match(dict(Fd_all), S_raw, 6.0, rad=1.5)
        offs.append(off)
        for th in (4.5, 5.0):
            for lab, A, B, ss in (("F->S", Fc, S, None), ("S->F", S, Fc, None), ("F->S step", Fc, S, 1), ("S->F step", S, Fc, 1)):
                n, okk, dq, dj, of = L.match(A, B, th, scale_step=ss)
                a = acc.setdefault((th, lab), {"n": 0, "m": 0, "dq": [], "dj": [], "off": []})
                a["n"] += n
                a["m"] += int(okk.sum())
                a["dq"].append(dq)
                a["dj"].append(dj)
                a["off"].append(of)
            n, okk, _, _, _ = L.match(Fc, S, th, thB=th)
            a = acc.setdefault((th, "F->S cut"), {"n": 0, "m": 0})
            a["n"] += n
            a["m"] += int(okk.sum())
    off = np.concatenate(offs)
    print(f"    position offset, SZiFi pixel - finder pixel (finder q > 6 at >= 60' from the edges, nearest within 1.5', "
          f"no correction): ({off[:, 0].mean():+.3f} +- {off[:, 0].std(ddof=1) / np.sqrt(len(off)):.3f}, "
          f"{off[:, 1].mean():+.3f} +- {off[:, 1].std(ddof=1) / np.sqrt(len(off)):.3f}) pixels, {len(off)} pairs; "
          f"row offsets {dict(zip(*np.unique(np.rint(off[:, 0]).astype(int), return_counts=True)))}")
    res = {"off": off.mean(0)}
    for th in (4.5, 5.0):
        for lab in ("F->S", "S->F", "F->S step", "S->F step"):
            a = acc[(th, lab)]
            dq = np.concatenate(a["dq"])
            dj = np.concatenate(a["dj"])
            of = np.concatenate(a["off"])
            frac = a["m"] / a["n"]
            same = np.mean(dj == 0) if dj.size else np.nan
            print(f"    q > {th}: {lab:9s}: N {a['n']:5d}, matched within 1' {100 * frac:6.2f}%; dq = q_S - q_F "
                  f"{(dq.mean() if lab.startswith('F') else -dq.mean()):+.3f} +- {dq.std(ddof=1):.3f}; same scale {100 * same:.1f}%"
                  + (f"; mean offset after the -1/2 shift ({of[:, 0].mean():+.3f}, {of[:, 1].mean():+.3f}) px" if lab == "F->S" else ""))
            res[(th, lab)] = (frac, dq.mean() if lab.startswith("F") else -dq.mean(), dq.std(ddof=1))
        a = acc[(th, "F->S cut")]
        print(f"    q > {th}: finder -> SZiFi when SZiFi is cut at the same q: matched {100 * a['m'] / a['n']:.2f}%")
    SUMMARY[(src, "it0", mode)] = res
    return res


# =====================================================================================================================
print("j04: SZiFi's iteration, recomputed by the verifier (own code)")
print(f"  interior >= {L.INT}' from the edges: {L.AREA:.2f} deg^2 per tile; SZiFi positions shifted by -1/2 pixel")
print(f"  sigma_analytic of SZiFi's filter: the verifier's own (j03) for S1/S2; for S3i the build's z04 value (data)")
print(f"  new tiles of the verifier: {NEW}")

print("\n" + "=" * 110 + "\n(b) THE BUILD'S STORED RUNS, recomputed\n" + "=" * 110)
pb = {}
for mode in L.MODES:
    for cfg in ("S1", "S2", "S3i"):
        pids = L.run_pids("build", cfg, mode, "real")
        pb[(cfg, mode)] = pids
        R = real_block("build", cfg, mode, pids, emul="a01" if cfg in ("S1", "S2") else None)
        print_real("build", cfg, mode, pids, R)
    pids = L.run_pids("build", "S3it0", mode, "real")
    for kind in ("real", "noise"):
        pk = L.run_pids("build", "S3it0", mode, kind)
        nq4, nq5, med = [], [], []
        for pid in pk:
            c = L.szifi_cat(L.load(L.run_path("build", "S3it0", mode, kind, pid)), 0)
            nq4.append(L.n_int(c, 4.0))
            nq5.append(L.n_int(c, 5.0))
            med.append(np.median(c["q"][L.inside(c)]))
        print(f"\n  ---- S3it0 (mixed_gaussian, iteration 0) {mode} {kind}, build, tiles {pk}: q > 4 / 5 per tile in the interior "
              f"{np.mean(nq4):.0f} / {np.mean(nq5):.0f}; median q {L.fmt(med, '{:.1f}')}")
for mode in L.MODES:
    for cfg in ("S1", "S2", "S3i"):
        noise_block("build", cfg, mode, L.run_pids("build", cfg, mode, "noise"))
for mode in L.MODES:
    it0_block("build", mode, pb[("S1", mode)])
# paired count ratios (tile bootstrap)
print("\n  ---- paired final q > 5 counts (interior), tile bootstrap (2000)")
for mode in L.MODES:
    cnt = {}
    for cfg in ("S1", "S2", "S3i"):
        for pid in pb[(cfg, mode)]:
            c1 = L.szifi_cat(L.load(L.run_path("build", cfg, mode, "real", pid)), 1)
            cnt[(cfg, pid)] = L.n_int(c1, 5.0)
    for pid in sorted({p for (_, p) in cnt}):
        cnt[("prod", pid)] = L.n_int(L.finder_cat(L.prod_tile(pid), f"{mode}_", 1), 5.0)
    for a, b in (("S1", "prod"), ("S2", "prod"), ("S3i", "prod"), ("S3i", "S2"), ("S2", "S1")):
        ps = sorted(set(pb[(a, mode)]) & set(pb[(b, mode)] if b != "prod" else pb[(a, mode)]))
        rr, ee = L.boot_ratio([cnt[(a, p)] for p in ps], [cnt[(b, p)] for p in ps])
        print(f"    {mode:8s} {a:4s} / {b:4s} ({len(ps)} tiles): {rr:.4f} +- {ee:.4f} "
              f"({sum(cnt[(a, p)] for p in ps)} / {sum(cnt[(b, p)] for p in ps)})")
        SUMMARY[("build", "ratio", mode, a, b)] = (rr, ee)

print("\n" + "=" * 110 + "\n(c) THE VERIFIER'S OWN RUNS ON NEW TILES\n" + "=" * 110)
for mode in L.MODES:
    for cfg in ("S1", "S2", "S3i"):
        pids = L.run_pids("mine", cfg, mode, "real")
        pids = [p for p in pids if p in NEW]
        if pids:
            R = real_block("mine", cfg, mode, pids, emul="j01" if cfg in ("S1", "S2") else None)
            print_real("mine", cfg, mode, pids, R)
    for kind in ("real", "noise"):
        pk = [p for p in L.run_pids("mine", "S3it0", mode, kind) if p in NEW]
        if not pk:
            continue
        nq4, nq5, med = [], [], []
        for pid in pk:
            c = L.szifi_cat(L.load(L.run_path("mine", "S3it0", mode, kind, pid)), 0)
            nq4.append(L.n_int(c, 4.0))
            nq5.append(L.n_int(c, 5.0))
            med.append(np.median(c["q"][L.inside(c)]))
        print(f"\n  ---- S3it0 (mixed_gaussian, iteration 0) {mode} {kind}, verifier, tiles {pk}: q > 4 / 5 per tile in the "
              f"interior {np.mean(nq4):.0f} / {np.mean(nq5):.0f} (range {min(nq4)}-{max(nq4)}); median q {L.fmt(med, '{:.1f}')}")
for mode in L.MODES:
    for cfg in ("S1", "S2"):
        pk = [p for p in L.run_pids("mine", cfg, mode, "noise") if p in NEW]
        if pk:
            noise_block("mine", cfg, mode, pk)
for mode in L.MODES:
    pk = [p for p in L.run_pids("mine", "S1", mode, "real") if p in NEW]
    if pk:
        it0_block("mine", mode, pk)

print("\n" + "=" * 110 + "\n(c2) NEW TILES AGAINST THE BUILD AND STEP 3a: differences in units of their combined error\n" + "=" * 110)
for mode in L.MODES:
    for cfg in ("S1", "S2", "S3i"):
        a, b = SUMMARY.get(("mine", cfg, mode, "real")), SUMMARY.get(("build", cfg, mode, "real"))
        if not a or "r1" not in a:
            continue
        for lab, idx in (("0.5'", 0), ("4.45'", 9), ("15'", 14)):
            d = a["r1"][idx] - b["r1"][idx]
            e = np.hypot(a["e1"][idx], b["e1"][idx])
            line = (f"  {cfg} {mode:8s} r it1 at {lab:5s}: new {a['r1'][idx]:.4f} +- {a['e1'][idx]:.4f} ({a['n']} tiles; tile sd "
                    f"{a['s1'][idx]:.4f}); build {b['r1'][idx]:.4f} +- {b['e1'][idx]:.4f} ({b['n']}); new - build {d:+.4f} = {d / e:+.1f} sigma")
            if (cfg, mode) in CLAIM:
                dc = a["r1"][idx] - CLAIM[(cfg, mode)][[0, 9, 14].index(idx)]
                line += (f"; new - step 3a claim {dc:+.4f} = {dc / a['e1'][idx]:+.1f} x (new error of the mean) = "
                         f"{dc / a['s1'][idx]:+.2f} x (tile sd)")
            print(line)
            if "dE" in a and "dE" in b:
                dd = a["dE"][idx] - b["dE"][idx]
                ee = np.hypot(a["eE"][idx], b["eE"][idx])
                print(f"       paired SZiFi - emulation: new {a['dE'][idx]:+.4f} +- {a['eE'][idx]:.4f}; build {b['dE'][idx]:+.4f} +- "
                      f"{b['eE'][idx]:.4f}; difference {dd:+.4f} = {dd / ee:+.1f} sigma")
        if "dn" in a:
            d = a["dn"][0] - b["dn"][0]
            print(f"  {cfg} {mode:8s} q > 5 change through the iteration: new {100 * a['dn'][0]:+.1f} +- {100 * a['dn'][1]:.1f}%; build "
                  f"{100 * b['dn'][0]:+.1f} +- {100 * b['dn'][1]:.1f}%; difference {d / np.hypot(a['dn'][1], b['dn'][1]):+.1f} sigma")
            d = a["vsprod"][0] - b["vsprod"][0]
            print(f"  {cfg} {mode:8s} final q > 5 count / production: new {a['vsprod'][0]:.3f} +- {a['vsprod'][1]:.3f}; build "
                  f"{b['vsprod'][0]:.3f} +- {b['vsprod'][1]:.3f}; difference {d / np.hypot(a['vsprod'][1], b['vsprod'][1]):+.1f} sigma")
            print(f"  {cfg} {mode:8s} f: new {a['f']:.3f}, build {b['f']:.3f}; spectra l > 3000 after/before: new {a['spec_hi']:.3f}, "
                  f"build {b['spec_hi']:.3f}")
    for cfg in ("S1", "S2"):
        a, b = SUMMARY.get(("mine", cfg, mode, "noise")), SUMMARY.get(("build", cfg, mode, "noise"))
        if not a or "s1" not in a:
            continue
        for idx, lab in ((0, "0.5'"), (14, "15'")):
            d = a["s1"][idx] - b["s1"][idx]
            print(f"  {cfg} {mode:8s} noise-only sigma/analytic it1 at {lab:4s}: new {a['s1'][idx]:.4f} +- {a['e1'][idx]:.4f} "
                  f"({a['n']} tiles); build {b['s1'][idx]:.4f} +- {b['e1'][idx]:.4f} ({b['n']}); difference "
                  f"{d / np.hypot(a['e1'][idx], b['e1'][idx]):+.1f} sigma")
        fa, fb = a["nf"][2], b["nf"][2]
        print(f"  {cfg} {mode:8s} false q > 5 per 100 deg^2: new {100 * fa / a['area']:.2f} ({fa}), build {100 * fb / b['area']:.2f} ({fb})")
print(f"\ndone [{time.time() - t0:.0f} s]")
