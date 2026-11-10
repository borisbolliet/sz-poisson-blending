"""Own pure-positional Poisson skies (no szsim): rare-config P(exists), pair-count variance, cap vs flat area.
Positions only, since for a marked Poisson process with position-independent marks any thinning is again Poisson."""
import numpy as np, time
from scipy.spatial import cKDTree
from scipy import stats
ARC = np.pi/180/60
rng = np.random.default_rng(31337)
def sky(nbar):
    n = rng.poisson(nbar); u = rng.uniform(-1, 1, n); ph = rng.uniform(0, 2*np.pi, n)
    s = np.sqrt(1-u*u); return np.column_stack([s*np.cos(ph), s*np.sin(ph), u])
ch = lambda t: 2*np.sin(t*ARC/2)
mu = lambda N, t: N*np.sin(t*ARC/2)**2   # = N(1-cos)/2
p3 = 1-3*np.sqrt(3)/(4*np.pi)
t0 = time.time()
# ---------- 1. rare configs, Nbar = N(M>3e14) ----------
N3 = 21840.4; NS = 6000
TP = [0.05, 0.2, 0.5, 1.0]; TT = [5., 10.]
pc = np.zeros((NS, len(TP)), int); tc = np.zeros((NS, len(TT)), int)
for s in range(NS):
    X = sky(N3); tr = cKDTree(X)
    for i, t in enumerate(TP): pc[s, i] = tr.count_neighbors(tr, ch(t)) - len(X)
    for i, t in enumerate(TT):
        pr = tr.query_pairs(ch(t), output_type='ndarray')
        if len(pr) < 3: continue
        # triangles via adjacency
        adj = {}
        for a, b in pr: adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
        tc[s, i] = sum(len(adj[a] & adj[b]) for a, b in pr) // 3
pc //= 2
print(f"[rare] {NS} skies Nbar={N3}  ({time.time()-t0:.0f}s)")
for i, t in enumerate(TP):
    E = N3*mu(N3, t)/2; Pa = -np.expm1(-E); Pm = (pc[:, i] > 0).mean(); e = np.sqrt(Pa*(1-Pa)/NS)
    print(f" pairs<{t}': E={E:.4f} MC mean={pc[:, i].mean():.4f}+-{pc[:, i].std()/np.sqrt(NS):.4f} | P(exists) 1-exp(-E)={Pa:.4f} MC={Pm:.4f}+-{e:.4f} z={(Pm-Pa)/e:+.2f}")
for i, t in enumerate(TT):
    E = N3*mu(N3, t)**2*p3/6; Pa = -np.expm1(-E); Pm = (tc[:, i] > 0).mean(); e = np.sqrt(Pa*(1-Pa)/NS)
    v = tc[:, i]
    print(f" triples<{t}': E={E:.4f} MC mean={v.mean():.4f}+-{v.std()/np.sqrt(NS):.4f} (z={(v.mean()-E)/(v.std()/np.sqrt(NS)):+.2f}) var/mean={v.var()/v.mean():.3f} | P(exists) 1-exp(-E)={Pa:.4f} MC={Pm:.4f}+-{e:.4f} z={(Pm-Pa)/e:+.2f}")
# ---------- 2. pair count variance, Nbar = det5 ----------
N5 = 85913.4; NS2 = 400; TH = [1., 5., 10.]
P = np.zeros((NS2, len(TH))); Ns = np.zeros(NS2)
for s in range(NS2):
    X = sky(N5); tr = cKDTree(X); Ns[s] = len(X)
    for i, t in enumerate(TH): P[s, i] = (tr.count_neighbors(tr, ch(t)) - len(X))/2
print(f"[pairs] {NS2} skies Nbar={N5} ({time.time()-t0:.0f}s)")
for i, t in enumerate(TH):
    m = mu(N5, t); E = N5*m/2; V = E*(1+2*m); f = np.sin(t*ARC/2)**2
    resid = P[:, i] - f*Ns*(Ns-1)/2   # remove the part driven by the realised N
    ve = P[:, i].var(ddof=1); se = ve*np.sqrt(2/(NS2-1))
    print(f" {t}': mu={m:.4f} E={E:.1f} MC={P[:, i].mean():.1f}+-{P[:, i].std()/np.sqrt(NS2):.1f} | Var_th={V:.1f} (naive {E:.1f}) Var_MC={ve:.1f}+-{se:.1f} ratio={ve/V:.3f}+-{se/V:.3f} naive-ratio={ve/E:.3f} | Var(P|N)~{resid.var(ddof=1):.1f} vs E={E:.1f}")
# ---------- 3. cap vs flat, Nbar = M>1e14 ----------
N14 = 396387.8; NS3 = 200; TC = [1., 5., 10., 30.]; NC = 1000
mc = np.zeros((NS3, len(TC)))
for s in range(NS3):
    X = sky(N14); tr = cKDTree(X)
    u = rng.uniform(-1, 1, NC); ph = rng.uniform(0, 2*np.pi, NC); R = np.column_stack([np.sqrt(1-u*u)*np.cos(ph), np.sqrt(1-u*u)*np.sin(ph), u])
    for i, t in enumerate(TC):
        c = 2*np.sin(np.radians(t)/2)
        mc[s, i] = tr.query_ball_point(R, c, return_length=True).mean()
print(f"[caps] {NS3} skies Nbar={N14} ({time.time()-t0:.0f}s)")
for i, t in enumerate(TC):
    th = np.radians(t); cap = N14*(1-np.cos(th))/2; flat = N14*th**2/4
    m = mc[:, i].mean(); e = mc[:, i].std(ddof=1)/np.sqrt(NS3)
    print(f" {t} deg: cap={cap:.3f} flat={flat:.3f} MC={m:.3f}+-{e:.3f} z_cap={(m-cap)/e:+.2f} z_flat={(m-flat)/e:+.2f}")
