"""w04c: Delta per q_obs bin (summed over z), map vs build (same primaries q_A >= 0.5), q_cut 5, full sky, with the
bootstrap sigma: shows that w04b's q-profile chi2 needs bins with more than a handful of objects (q > 100 holds ~8)."""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = "/Users/boris/GitHub/sz-poisson-blending/data_real/step2"
d = np.load(os.path.join(HERE, "out", "w04_delta.npz"))
out = open(os.path.join(HERE, "w04c_qprofile.log"), "w")
def log(s):
    print(s); out.write(s + "\n")
for nm, b in (('baseline_f1', 'baseline_f1'), ('goal_f1', 'goal_f1'), ('baseline_f2', 'baseline_f2'), ('white_f1', 'white'),
              ('baseline_effi', 'baseline_effi')):
    D = d[nm + '_q5_D']; Db = d[nm + '_q5_Dboot'].astype(float); Ns = d[nm + '_q5_Ns']
    t = np.load(f'{R}/truth_{b}.npz'); B = t['D05_q5']; BN = t['Nn05_q5']
    qp = D.sum(0); qb = B.sum(0); qs = Db.sum(1).std(0)
    log(f"[{nm}] q bin | N_single (map) | Delta map +- boot | Delta build | pull | Delta/N map, build")
    for k in range(9):
        log(f"   {k} | {Ns.sum(0)[k]:9.1f} | {qp[k]:+8.1f} +- {qs[k]:6.1f} | {qb[k]:+8.1f} | {(qp[k]-qb[k])/max(qs[k],1e-12):+5.1f} | "
            f"{100*qp[k]/Ns.sum(0)[k]:+.2f}%  {100*qb[k]/BN.sum(0)[k]:+.2f}%")
out.close()
