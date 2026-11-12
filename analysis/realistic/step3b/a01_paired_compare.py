"""Step 3b arbitration / a01: the baseline paired count change (claims N1, N3) -- build vs the sims verifier.

The sims verifier's 400 own tiles give R_p = N_det / N_iso - 1 = -1.71 +- 0.13% at q > 5 (baseline) against the build's
-1.36 +- 0.06% (1,548 tiles), and no extra deficit at z < 0.1 (R_p -0.1 +- 1.2% against -3.3 +- 0.6%).  Both use the
same definitions: credited halos (merge none) at their true z vs isolated copies (same position, same q', halo-free tile,
virtual insertion).  This script puts both data sets through ONE code:
  (A) totals at e = 5 and per tile (N_det, N_iso, halos with q' in 4-6), build Q2 / new / all and sims;
  (B) R_p in step 3a's six coarse z bins;
  (C) the paired change in cells of theta500 x q' (where does each data set lose clusters?) and the two flip rates
      P(credited, copy not) and P(copy credited, halo not);
  (D) random 400-tile subsets of the build's 1,548 tiles: how often do they give the sims verifier's values?
  (E) halo population per tile (q' 3-6 by z; z < 0.1 halos by theta500);
  (F) z < 0.1: per-tile distribution of N_iso - N_det (heavy tails?), contribution of the largest tiles.
Both noise levels.  Usage: python a01_paired_compare.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arblib as A
A.threads(1)
import numpy as np

E = 5.0
THB = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 1e3])
QPB = np.array([2.0, 4.0, 5.0, 6.0, 8.0, 15.0, 1e4])
NB = 2000


def stats(D, label, W=None):
    nt = D["ntiles"]
    det = D["qo_none"] > E; iso = D["qiso"] > E
    nd = A.per_tile(D["t"], det.astype(float), nt); ni = A.per_tile(D["t"], iso.astype(float), nt)
    W = A.boot_w(nt, NB, 11) if W is None else W
    v, e, _ = A.ratio_boot(nd, ni, W)
    m46 = (D["qp"] > 4) & (D["qp"] < 6)
    n46 = A.per_tile(D["t"], m46.astype(float), nt)
    print(f"  {label:22s} tiles {nt:5d}: per tile N_det {nd.mean():7.2f}, N_iso {ni.mean():7.2f}, N_iso - N_det "
          f"{(ni - nd).mean():5.2f} +- {(ni - nd).std(ddof=1) / np.sqrt(nt):.2f}; halos with q' 4-6 {n46.mean():7.2f} "
          f"+- {n46.std(ddof=1) / np.sqrt(nt):.2f};  R_p = {A.fmt(v, e)}", flush=True)
    return dict(nd=nd, ni=ni, W=W, det=det, iso=iso)


def zbins(D, S, label):
    nt = D["ntiles"]
    out = []
    for a, b in zip(A.ZC[:-1], A.ZC[1:]):
        m = (D["z"] >= a) & (D["z"] < b)
        nd = A.per_tile(D["t"][m], S["det"][m].astype(float), nt); ni = A.per_tile(D["t"][m], S["iso"][m].astype(float), nt)
        v, e, _ = A.ratio_boot(nd, ni, S["W"])
        out.append((v, e, nd.sum(), ni.sum()))
    print(f"  {label:22s} R_p by z: " + ", ".join(f"{100 * v:+.1f}+-{100 * e:.1f} ({int(n)})" for v, e, n, _ in out), flush=True)
    return out


def cells(D, S):
    """Paired change (N_det - N_iso) / N_iso and flip rates per (theta500, q') cell, with bootstrap errors."""
    nt = D["ntiles"]
    it = np.clip(np.searchsorted(THB, D["th"], side="right") - 1, 0, THB.size - 2)
    iq = np.searchsorted(QPB, D["qp"], side="right") - 1
    ok = (iq >= 0) & (iq < QPB.size - 1)
    nc = (THB.size - 1) * (QPB.size - 1)
    c = it * (QPB.size - 1) + iq
    res = {}
    for lab, w in (("det", S["det"]), ("iso", S["iso"]), ("n", np.ones(D["t"].size, bool)),
                   ("lost", (~S["det"]) & S["iso"]), ("gain", S["det"] & ~S["iso"])):
        idx = (D["t"][ok].astype(np.int64) * nc + c[ok])
        res[lab] = np.bincount(idx, weights=w[ok].astype(float), minlength=nt * nc).reshape(nt, nc)
    Wb = S["W"]
    Nd, Ni = res["det"].sum(0), res["iso"].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = Nd / Ni - 1
        fb = (Wb @ res["det"]) / (Wb @ res["iso"]) - 1
    return f.reshape(THB.size - 1, -1), np.nanstd(fb, 0).reshape(THB.size - 1, -1), Ni.reshape(THB.size - 1, -1), \
        (res["lost"].sum(0) / np.maximum(res["n"].sum(0), 1)).reshape(THB.size - 1, -1), \
        (res["gain"].sum(0) / np.maximum(res["n"].sum(0), 1)).reshape(THB.size - 1, -1), fb


if __name__ == "__main__":
    t00 = time.time()
    for mode in ("baseline", "goal"):
        print(f"=================== {mode} ===================", flush=True)
        Bq = A.load_build(mode, "q2"); Bn = A.load_build(mode, "new"); Ba = A.load_build(mode, "all"); Ss = A.load_sims(mode)
        print(f"(A) totals at q_obs > {E:.0f} (paired: N_det = credited halos, merge none; N_iso = credited isolated copies); "
              f"tile bootstrap {NB}", flush=True)
        S = {}
        for lab, D in (("build Q2", Bq), ("build new tiles", Bn), ("build all", Ba), ("sims verifier", Ss)):
            S[lab] = stats(D, lab)
        dlt = (S["sims verifier"]["nd"].sum() / S["sims verifier"]["ni"].sum()) - (S["build all"]["nd"].sum() / S["build all"]["ni"].sum())
        print(f"  sims - build (all): {100 * dlt:+.2f} pp", flush=True)
        print("(B) R_p by z (0.01-0.1, 0.1-0.5, 0.5-1, 1-1.5, 1.5-2, 2-3) [%] (credited halos in brackets)", flush=True)
        Z = {}
        for lab, D in (("build Q2", Bq), ("build new tiles", Bn), ("build all", Ba), ("sims verifier", Ss)):
            Z[lab] = zbins(D, S[lab], lab)
        print(f"(C) paired change (N_det - N_iso) / N_iso [%] at e = 5 per cell of theta500 [arcmin] x q'; +- tile bootstrap; "
              f"[N_iso]", flush=True)
        C = {}
        for lab, D in (("build all", Ba), ("sims verifier", Ss)):
            C[lab] = cells(D, S[lab])
        hdr = "  theta \\ q'      " + "".join(f"{f'{a:g}-{b:g}':>24s}" for a, b in zip(QPB[:-1], QPB[1:]))
        for lab in C:
            f, e, n, lost, gain, _ = C[lab]
            print(f"  -- {lab}\n" + hdr, flush=True)
            for i, (a, b) in enumerate(zip(THB[:-1], THB[1:])):
                print(f"  {f'{a:g}-{b:g}':>14s}  " + "".join(f"{100 * f[i, k]:+7.2f}+-{100 * e[i, k]:4.2f} [{int(n[i, k]):6d}]" for k in range(f.shape[1])), flush=True)
        fa, ea, *_ = C["build all"]; fs, es, *_ = C["sims verifier"]
        pull = (fs - fa) / np.sqrt(ea ** 2 + es ** 2)
        ok = np.isfinite(pull)
        print(f"  pulls sims - build per cell: chi^2 {np.sum(pull[ok] ** 2):.1f} for {ok.sum()} cells; largest |pull| "
              f"{np.nanmax(np.abs(pull)):.2f} at theta {THB[np.unravel_index(np.nanargmax(np.abs(pull)), pull.shape)[0]]:g}'", flush=True)
        for lab in C:
            f, e, n, lost, gain, _ = C[lab]
            print(f"  flip rates per halo, {lab}: lost (copy credited, halo not) / gained, q' 4-5, 5-6, 6-8 by theta: " +
                  "; ".join(f"{a:g}-{b:g}': " + ", ".join(f"{100 * lost[i, k]:.1f}/{100 * gain[i, k]:.1f}" for k in (1, 2, 3))
                            for i, (a, b) in enumerate(zip(THB[:-1], THB[1:]))), flush=True)
        # (D) random 400-tile subsets of the build
        rng = np.random.default_rng(5)
        nd_a, ni_a = S["build all"]["nd"], S["build all"]["ni"]
        mz = (Ba["z"] < 0.1)
        ndz = A.per_tile(Ba["t"][mz], S["build all"]["det"][mz].astype(float), Ba["ntiles"])
        niz = A.per_tile(Ba["t"][mz], S["build all"]["iso"][mz].astype(float), Ba["ntiles"])
        tot, low = [], []
        for _ in range(4000):
            s = rng.choice(Ba["ntiles"], 400, replace=False)
            tot.append(nd_a[s].sum() / ni_a[s].sum() - 1); low.append(ndz[s].sum() / niz[s].sum() - 1)
        tot, low = np.array(tot), np.array(low)
        sv = Z["sims verifier"][0][0]; st = S["sims verifier"]["nd"].sum() / S["sims verifier"]["ni"].sum() - 1
        print(f"(D) 4000 random 400-tile subsets of the build's 1,548 tiles: R_p total mean {100 * tot.mean():+.2f}%, std "
              f"{100 * tot.std():.2f}%, fraction <= sims' {100 * st:+.2f}%: {np.mean(tot <= st):.4f}; R_p at z < 0.1 mean "
              f"{100 * low.mean():+.2f}%, std {100 * low.std():.2f}%, fraction >= sims' {100 * sv:+.2f}%: {np.mean(low >= sv):.4f}; "
              f"both: {np.mean((tot <= st) & (low >= sv)):.4f}", flush=True)
        # (E) population
        print("(E) halos per tile with q' in 3-6, by coarse z bin; and z < 0.1 halos with q' > 4 by theta500 (<4, 4-8, 8-16, >16')", flush=True)
        for lab, D in (("build Q2", Bq), ("build new tiles", Bn), ("sims verifier", Ss)):
            nt = D["ntiles"]
            row = []
            for a, b in zip(A.ZC[:-1], A.ZC[1:]):
                m = (D["z"] >= a) & (D["z"] < b) & (D["qp"] > 3) & (D["qp"] < 6)
                row.append(m.sum() / nt)
            m = (D["z"] < 0.1) & (D["qp"] > 4)
            hth = np.histogram(D["th"][m], [0, 4, 8, 16, 1e3])[0] / nt
            print(f"  {lab:18s}: " + ", ".join(f"{x:.3f}" for x in row) + " | z < 0.1, q' > 4: " + ", ".join(f"{x:.3f}" for x in hth)
                  + f" | halos per tile (rows q' >= 1.9) {np.sum(D['qp'] >= 1.9) / nt:.1f}", flush=True)
        # (F) z < 0.1 per-tile distribution
        print("(F) z < 0.1: per-tile N_iso - N_det (credited copies minus credited halos)", flush=True)
        for lab, D in (("build Q2", Bq), ("build new tiles", Bn), ("sims verifier", Ss)):
            nt = D["ntiles"]
            m = D["z"] < 0.1
            dd = A.per_tile(D["t"][m], (D["qiso"][m] > E).astype(float) - (D["qo_none"][m] > E).astype(float), nt)
            ni = A.per_tile(D["t"][m], (D["qiso"][m] > E).astype(float), nt)
            srt = np.sort(dd)[::-1]
            print(f"  {lab:18s}: mean {dd.mean():.4f} +- {dd.std(ddof=1) / np.sqrt(nt):.4f} per tile (N_iso {ni.mean():.3f}); "
                  f"values {dict(zip(*np.unique(dd, return_counts=True)))}; largest 5: {srt[:5]}; share of the net from the top 1% "
                  f"of tiles {srt[:max(1, nt // 100)].sum() / max(dd.sum(), 1e-9):.2f}", flush=True)
        del Bq, Bn, Ba, Ss
    print(f"done [{time.time() - t00:.0f} s]; peak RSS {A.peak_rss_gb():.2f} GB", flush=True)
