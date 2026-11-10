"""Independent MC of aperture counts using fresh szsim full-sky catalogues (M>1e13 floor), own neighbour code.
Seeds 900001.. (disjoint from analysis seeds). Analytic Nbar taken from v1 (own colossus integral)."""
import os, sys, time, json, resource
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
from szsim.sampler import draw_catalog
ARC = np.pi/180/60
NB = dict(det5=85913.4, M3e13=4598099.4, M1e14=396387.8, det4=132126.5)  # from v1_analytic.log
NSKY = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = sys.argv[2] if len(sys.argv) > 2 else 'v2_mc_szsim_results.npz'
SEED0 = int(sys.argv[3]) if len(sys.argv) > 3 else 900001
TH = np.array([1., 2., 5., 10.])
NMAX = 6
def xyz(ra, dec):
    ra = np.radians(ra.astype(np.float64)); dec = np.radians(dec.astype(np.float64))
    return np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
def chord(t_arcmin): return 2*np.sin(t_arcmin*ARC/2)
def counts(tree, pts, t, exclude_self):
    # exclude_self=True: centre is in the neighbour set by construction.
    # Otherwise the centre MAY be a member of the neighbour population (e.g. det5 centre with M>1e14):
    # remove it by exact-position match (Palm: centre never counted).
    c = tree.query_ball_point(pts, chord(t), return_length=True)
    if exclude_self:
        return c - 1
    return c - tree.query_ball_point(pts, 1e-12, return_length=True)
res = {}
def add(key, arr):
    res.setdefault(key, []).append(arr)
t0 = time.time()
rng_c = np.random.default_rng(424242)
for s in range(NSKY):
    cat = draw_catalog(SEED0+s, Mmin=1e13, fields=("ra", "dec", "M500", "q_obs", "z"))
    M = cat['M500']; qo = cat['q_obs']
    sel = {'det5': qo > 5, 'det4': qo > 4, 'M3e13': M >= 3e13, 'M1e14': M >= 1e14}
    P = {k: xyz(cat['ra'][v], cat['dec'][v]) for k, v in sel.items()}
    z5 = cat['z'][sel['det5']]
    add('Ntot_det5', np.array([len(P['det5'])])); add('Ntot_M3e13', np.array([len(P['M3e13'])])); add('Ntot_M1e14', np.array([len(P['M1e14'])])); add('Ntot_det4', np.array([len(P['det4'])]))
    del cat, M, qo, sel
    T = {k: cKDTree(v) for k, v in P.items()}
    # random centres
    u = rng_c.uniform(-1, 1, 100000); ph = rng_c.uniform(0, 2*np.pi, 100000)
    R = np.column_stack([np.sqrt(1-u*u)*np.cos(ph), np.sqrt(1-u*u)*np.sin(ph), u])
    for pop in ['det5', 'M3e13', 'M1e14']:
        for t in TH:
            c = tree_len = T[pop].query_ball_point(R, chord(t), return_length=True)
            add(f'rand|{pop}|{t}', np.bincount(np.minimum(c, NMAX), minlength=NMAX+1)/len(c))
    # Palm: detected-centred
    for cen, pop in [('det5', 'det5'), ('det4', 'det4'), ('det5', 'M3e13'), ('det5', 'M1e14')]:
        for t in TH:
            c = counts(T[pop], P[cen], t, cen == pop)
            assert c.min() >= 0
            add(f'palm|{cen}|{pop}|{t}', np.bincount(np.minimum(c, NMAX), minlength=NMAX+1)/len(c))
            add(f'ncen|{cen}|{pop}|{t}', np.array([len(c)]))
    # Palm with centre split by own z (marks independent of positions)
    for lab, m in [('zc>1', z5 > 1), ('zc<0.5', z5 < 0.5)]:
        for t in [5., 10.]:
            c = counts(T['M1e14'], P['det5'][m], t, False)
            add(f'palm|det5{lab}|M1e14|{t}', np.bincount(np.minimum(c, NMAX), minlength=NMAX+1)/len(c))
            add(f'ncen|det5{lab}|M1e14|{t}', np.array([len(c)]))
    # pair and triple counts det5
    for t in [0.5, 1., 2., 5., 10.]:
        pr = T['det5'].query_pairs(chord(t), output_type='ndarray')
        add(f'pairs|{t}', np.array([len(pr)]))
        if t >= 2:
            # triples with all three separations < t: for each pair (i<j), count k>j adjacent to both
            nb = T['det5'].query_ball_point(P['det5'], chord(t))
            S = [set(x) for x in nb]
            ntri = 0
            for i, j in pr:
                a, b = (i, j) if i < j else (j, i)
                ntri += sum(1 for k in (S[a] & S[b]) if k > b)
            add(f'triples|{t}', np.array([ntri]))
            # number of det5 with >= n neighbours
            nn = np.array([len(x)-1 for x in nb])
            add(f'geqn|{t}', np.array([(nn >= n).sum() for n in range(1, 6)]))
    del P, T
    print(f"sky {s} done {time.time()-t0:.0f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB", flush=True)
np.savez(OUT, **{k: np.array(v) for k, v in res.items()})
