"""j08: claim 1g, does step 3a's SZiFi-check emulation (verify_real/step3_szifi, runs v04_lin: SZiFi's pixel-sampled
template, zero-padded filtering; 10 tiles per level, pid 0, 33, ..., 297) reproduce SZiFi's own iteration 0?
Compared with the build's stored S1 runs on the same tiles (iteration 0) and with the verifier's own sigma_analytic (j03).
Both catalogues use SZiFi's pixel convention (the S/N at pixel n belongs to n - 1/2); interior as in jlib.
Usage: .venv/bin/python -I -B j08_emul.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np
import jlib as L

V04 = os.path.join(L.ROOT, "verify_real", "step3_szifi", "out")
ana = L.load(os.path.join(L.OUT, "j03_analytic.npz"))
print("j08: SZiFi's iteration 0 (the build's S1 runs) vs step 3a's emulation v04_lin, same tiles")
for mode in L.MODES:
    pids = [p for p in range(0, 324, 33) if os.path.exists(os.path.join(V04, f"v04_lin_{mode}_tile{p:03d}.npz"))]
    dsig, dana, acc, ident, ncom, dq_com = [], [], {}, 0, 0, []
    for pid in pids:
        e = L.load(os.path.join(V04, f"v04_lin_{mode}_tile{pid:03d}.npz"))
        r = L.load(L.run_path("build", "S1", mode, "real", pid))
        s = L.szifi_sigma(r, 0)
        dsig.append(s / e["sig0"] - 1)
        dana.append(ana[f"sigana_{mode}"] / e["sig_ana"] - 1)
        S = L.szifi_cat(r, 0)
        E = dict(q=np.asarray(e["det0_q"], float), py=e["det0_n0"] - 0.5, px=e["det0_n1"] - 0.5,
                 jt=np.asarray(e["det0_j"], np.int64))
        for th in (4.5, 5.0):
            for lab, A, B in (("emul->SZiFi", E, S), ("SZiFi->emul", S, E)):
                n, ok, dq, dj, off = L.match(A, B, th)
                a = acc.setdefault((th, lab), [0, 0, [], []])
                a[0] += n
                a[1] += int(ok.sum())
                a[2].append(dq)
                a[3].append(dj)
        # identical (pixel, scale) entries in the interior, q > 4
        si, ei = L.inside(S) & (S["q"] > 4), L.inside(E) & (E["q"] > 4)
        ks = {(int(a), int(b), int(c)): q for a, b, c, q in zip(r["it0_i"][si], r["it0_j"][si], S["jt"][si], S["q"][si])}
        ke = {(int(a), int(b), int(c)): q for a, b, c, q in zip(e["det0_n0"][ei], e["det0_n1"][ei], E["jt"][ei], E["q"][ei])}
        com = set(ks) & set(ke)
        ident += len(com)
        ncom += len(set(ks) | set(ke))
        dq_com += [ks[k] - ke[k] for k in com]
    dsig, dana = np.array(dsig), np.array(dana)
    print(f"\n  {mode}: {len(pids)} tiles {pids}")
    print(f"    sigma_SZiFi(it0) / sigma_emulation(it0) - 1: max |.| over tiles and scales {np.abs(dsig).max():.1e} "
          f"(mean at 0.5 / 4.45 / 15': {L.fmt(dsig.mean(0)[[0, 9, 14]], '{:+.1e}')})")
    print(f"    sigma_analytic: verifier's (j03, SZiFi's own filter) / emulation's - 1: {L.fmt(dana.mean(0)[[0, 4, 9, 14]], '{:+.1e}')} "
          f"at 0.5, 1.32, 4.45, 15' (max |.| {np.abs(dana).max():.1e})")
    for (th, lab), (n, m, dq, dj) in sorted(acc.items()):
        dq, dj = np.concatenate(dq), np.concatenate(dj)
        print(f"    q > {th}: {lab}: N {n}, matched within 1' {100 * m / n:.2f}%; dq (partner - entry) {dq.mean():+.3f} +- "
              f"{dq.std(ddof=1):.3f}; same scale {100 * np.mean(dj == 0):.1f}%")
    print(f"    interior q > 4 (pixel, scale) entries in both catalogues: {ident} of {ncom}; max |dq| over them "
          f"{np.abs(dq_com).max():.2e}")
