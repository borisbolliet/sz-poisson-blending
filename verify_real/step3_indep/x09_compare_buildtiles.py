"""x09 (indep lens): one-to-one comparison of the own pipeline run on the build's tiles (x04c -> out/buildtiles) with the
build's stored arrays (data_real/step3/tiles): catalogue, primary, c_p, delta, q_alone, forced-photometry values and the
class labels of every q > 5 detection and of every forced halo.  Usage: python x09_compare_buildtiles.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vlib3 as V

D = os.path.join(V.OUT, "buildtiles")
pids = sorted(int(f[6:9]) for f in os.listdir(D) if f.startswith("fresh_"))
log = V.Tee(os.path.join(V.HERE, "x09_compare_buildtiles.log"))
log(f"x09: own pipeline on build tiles {pids} vs the build's stored arrays")
tot = {}


def add(k, v):
    tot.setdefault(k, []).append(v)


for pid in pids:
    m = dict(np.load(os.path.join(D, f"fresh_{pid:03d}.npz")))
    b = V.build_tile(pid); h = V.build_truth(pid)
    for mode in V.MODES:
        p = mode + "_"
        qt = h["Y500"] / V.sigma_Y(mode, h["theta500"])
        # catalogue
        km = {(int(a), int(c), int(d)): i for i, (a, c, d) in enumerate(zip(m[p + "j"], m[p + "iy"], m[p + "ix"]))}
        kb = {(int(a), int(c), int(d)): i for i, (a, c, d) in enumerate(zip(b[p + "it1_j"], b[p + "it1_iy"], b[p + "it1_ix"]))}
        com = sorted(set(km) & set(kb))
        im = np.array([km[k] for k in com]); ib = np.array([kb[k] for k in com])
        add("cat_unmatched", len(km) + len(kb) - 2 * len(com))
        add("dq", np.abs(m[p + "q"][im] - b[p + "it1_q"][ib]).max())
        # q > 5 detections: own d5 rows <-> build rows
        d5 = m[p + "d5"]
        rowm = {int(i): a for a, i in enumerate(d5)}
        sig = b[p + "it1_sig"]
        pairs = [(rowm[i], j) for i, j in zip(im, ib) if i in rowm]
        am = np.array([a for a, _ in pairs]); bb = np.array([j for _, j in pairs])
        primm = m[p + "top_idx"][am, 0]; primb = b[p + "d_top_idx"][bb, 0]
        cpm = m[p + "top_ex"][am, 0]; cpb = b[p + "d_top_ex"][bb, 0]
        sd = sig[b[p + "it1_j"][bb]]
        dm = m[p + "S_tsz"][d5][am] - cpm; db = b[p + "d_S_tsz"][bb] - cpb
        add("n5", am.size); add("prim_diff", int(np.sum(primm != primb)))
        add("dcp", np.max(np.abs(cpm - cpb) / sd)); add("ddelta", np.max(np.abs(dm - db) / sd))
        realm, realb = cpm / sd >= 2, cpb / sd >= 2
        add("real_flip", int(np.sum(realm != realb)))
        add("b10_flip", int(np.sum((dm > 0.1 * cpm) != (db > 0.1 * cpb))))
        # q_alone of the primaries (own isolated search vs build's)
        fim, fqm = m[p + "f_idx"], m[p + "f_q_alone"]
        qam = fqm[np.searchsorted(fim, primm)]
        qab = b[p + "d_q_alone"][bb]
        add("dqa_prim", np.abs(qam - qab))
        add("needs_flip", int(np.sum((qam < 5) != (qab < 5))))
        # forced halos common to both
        fib = b[p + "f_idx"]
        cm = np.intersect1d(fim, fib)
        ia = np.searchsorted(fim, cm); ibb = np.searchsorted(fib, cm)
        fs = sig[b[p + "f_j"][ibb]]
        add("nf", cm.size)
        add("dqa", np.abs(fqm[ia] - b[p + "f_q_alone"][ibb]))
        add("alone5_flip", int(np.sum((fqm[ia] > 5) != (b[p + "f_q_alone"][ibb] > 5))))
        add("down", np.abs(m[p + "f_own"][ia] - b[p + "f_own"][ibb]) / fs)
        add("dnb", np.abs(m[p + "f_nb_val"][ia, 0] - b[p + "f_nb_val"][ibb, 0]) / fs)
        add("dfst", np.abs(m[p + "f_S_tsz"][ia] - b[p + "f_S_tsz"][ibb]) / fs)
        add("dfcn", np.abs(m[p + "f_S_cn"][ia] - (b[p + "f_S_cmb"][ibb] + b[p + "f_S_noise"][ibb])) / fs)
        add("fj_diff", int(np.sum(m[p + "f_j"][ia] != b[p + "f_j"][ibb])))
        log(f"  tile {pid:3d} {mode:8s}: catalogue unmatched {tot['cat_unmatched'][-1]}, max |dq| {tot['dq'][-1]:.1e}; q > 5: {am.size}, "
            f"primary differs {tot['prim_diff'][-1]}, max |d c_p| {tot['dcp'][-1]:.1e} sigma, max |d delta| {tot['ddelta'][-1]:.1e} sigma; "
            f"q_alone of primaries max |d| {tot['dqa_prim'][-1].max():.1e}; forced common {cm.size}: max |d q_alone| {tot['dqa'][-1].max():.1e}, "
            f"own {tot['down'][-1].max():.1e}, nb {tot['dnb'][-1].max():.1e}, tsz {tot['dfst'][-1].max():.1e}, cn {tot['dfcn'][-1].max():.1e} sigma")
dqa = np.concatenate(tot["dqa"]); dqap = np.concatenate(tot["dqa_prim"])
log(f"\nsummary over {len(pids)} tiles x 2 levels: catalogue unmatched {sum(tot['cat_unmatched'])}; q > 5 detections {sum(tot['n5'])}: "
    f"primary differs {sum(tot['prim_diff'])}, max |d c_p| {max(tot['dcp']):.1e} sigma, max |d delta| {max(tot['ddelta']):.1e} sigma, "
    f"label flips: real {sum(tot['real_flip'])}, Y boosted > 10% {sum(tot['b10_flip'])}, only thanks to neighbours {sum(tot['needs_flip'])}")
log(f"  isolated search q_alone, own vs build: primaries median |d| {np.median(dqap):.1e}, max {dqap.max():.1e}; all common forced halos "
    f"({sum(tot['nf'])}): median {np.median(dqa):.1e}, 99% {np.percentile(dqa, 99):.1e}, max {dqa.max():.1e}; q_alone > 5 flips {sum(tot['alone5_flip'])}")
log(f"  forced photometry (sigma units), max over all: own response {max(np.max(v) for v in tot['down']):.1e}, largest neighbour "
    f"{max(np.max(v) for v in tot['dnb']):.1e}, tSZ {max(np.max(v) for v in tot['dfst']):.1e}, CMB + noise {max(np.max(v) for v in tot['dfcn']):.1e}; "
    f"nearest-scale differences {sum(tot['fj_diff'])}")
