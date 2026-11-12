"""x13 (indep lens): near-threshold completeness of three independent injection sets of the same design: the build's
(data_real/step3/injections), own set 1 (x03, out/inj), own set 2 (x03b, out/inj2).  P(credited q_obs > 5 | q_inj in
[4, 6)) per size with tile-bootstrap errors, and the pairwise chi^2 over the six sizes.  Also the own binned-table
variant of the count change is NOT recomputed here (see x06).  Usage: python x13_injection_sets.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vlib3 as V

log = V.Tee(os.path.join(V.HERE, "x13_injection_sets.log"))
TI = [0.3, 0.6, 1.2, 2.4, 4.8, 9.6]
rng = np.random.default_rng(3)


def load(d, own, mode):
    out = {t: ([], [], []) for t in TI}
    for f in sorted(os.listdir(d)):
        if not f.startswith(f"inj_{mode}_"):
            continue
        r = dict(np.load(os.path.join(d, f)))
        th = float(r["theta_inj"])
        if own:
            qb = r["qbest"]
        else:
            sig = r["sig"]; cred = r["d_S_tsz"] / sig[r["d_j"]] >= 2.0
            qb = np.full(r["q_inj"].size, -np.inf)
            for k in np.flatnonzero(cred):
                h = r["d_near"][k]
                qb[h] = max(qb[h], r["d_q"][k])
        out[th][0].append(r["q_inj"]); out[th][1].append(qb); out[th][2].append(np.full(r["q_inj"].size, int(r["k"])))
    return {t: tuple(np.concatenate(v) for v in out[t]) for t in TI}


def comp(S, t):
    q, qb, tl = S[t]
    m = (q >= 4.0) & (q < 6.0)
    d = (qb[m] > 5).astype(float); tl = tl[m]
    ut = np.unique(tl)
    idx = {u: np.flatnonzero(tl == u) for u in ut}
    bs = [d[np.concatenate([idx[u] for u in rng.choice(ut, ut.size)])].mean() for _ in range(300)]
    return d.mean(), np.std(bs)


log("x13: P(credited q_obs > 5 | q_inj 4-6) per size, three independent injection sets (tile-bootstrap errors)")
for mode in V.MODES:
    sets = {"build": load(os.path.join(V.BUILD_DATA, "injections"), False, mode), "own1": load(os.path.join(V.OUT, "inj"), True, mode),
            "own2": load(os.path.join(V.OUT, "inj2"), True, mode)}
    R = {k: np.array([comp(S, t) for t in TI]) for k, S in sets.items()}
    log(f"\n{mode}")
    for i, t in enumerate(TI):
        log(f"  theta {t:4}': " + "  ".join(f"{k} {R[k][i, 0]:.4f} +- {R[k][i, 1]:.4f}" for k in R))
    for a, b in (("build", "own1"), ("build", "own2"), ("own1", "own2")):
        z = (R[a][:, 0] - R[b][:, 0]) / np.hypot(R[a][:, 1], R[b][:, 1])
        w = 1 / (R[a][:, 1] ** 2 + R[b][:, 1] ** 2)
        log(f"  {a} - {b}: per size " + ", ".join(f"{v:+.1f}" for v in z) + f" sigma; chi2 {np.sum(z ** 2):.1f} for 6; weighted mean difference "
            f"{np.sum(w * (R[a][:, 0] - R[b][:, 0])) / w.sum():+.4f} +- {w.sum() ** -0.5:.4f}")
