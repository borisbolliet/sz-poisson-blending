"""Step 3b arbitration / a06b: a06's careful re-optimisation for the other fof 10' fits that end at (or near) the
sigma_lnY bound or have the largest ML / linearised ratios (fits verifier, f05 (6)): goal SO40 d and e, goal q_cut 6
SO40 e, baseline SO40 e and full-sky e (all R_s, q_cut 5 unless stated).  Same procedure and output format as a06.
Usage: python a06b_fit_checks_fof.py NPROC
"""
import os, sys, json, time
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a06_fit_checks as A6                        # noqa: E402
import s3blib as B                                 # noqa: E402

CASES = [("prod_fof10", "goal", "SO40", "d"), ("prod_fof10", "goal", "SO40", "e"), ("prod_fof10", "baseline", "SO40", "e"),
         ("prod_fof10", "baseline", "full", "e")]

if __name__ == "__main__":
    from multiprocessing import get_context
    t00 = time.time()
    with get_context("spawn").Pool(NPROC) as p:
        out = list(p.imap(A6.job, CASES))
    for res in out:
        s = res["case"]
        print(f"  {s[0]:10s} {s[1]:8s} {s[2]:4s} {s[3]}: build ML S8 {res['b_S8_build']:+.4f} (lnL {res['lnL_build']:.6f}, at bound "
              f"{res['at_bound']}); re-optimised: " + "; ".join(f"from {a['start']}: lnL {a['lnL']:.6f} (+{a['lnL'] - res['lnL_build']:.2e}), "
                                                                   f"S8 {a['b_S8']:+.4f}" for a in res["starts"])
              + f" -> best S8 {res['b_S8_best']:+.4f}, theta {[round(v, 4) for v in res['theta_best']]} [{res['sec']:.0f} s]", flush=True)
    json.dump(out, open(os.path.join(B.OUT, "arbiter", "a06b_fit_checks_fof.json"), "w"), indent=1, default=float)
    print(f"done [{time.time() - t00:.0f} s]", flush=True)
