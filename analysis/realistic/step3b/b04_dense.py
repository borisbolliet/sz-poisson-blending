"""Step 3b / b04: do dense injection tiles select isolated clusters differently from virtual clusters on noise-only
tiles, and why?  (Production finder.)

Step 3a found that injection tiles (t03 / a02: 64-225 clusters per tile, 40'-75' apart) expect 0.8% (baseline) and 1.2%
(goal) fewer isolated detections than virtual clusters on noise-only tiles.  Here the dense design is repeated on the
noise-only skies themselves, so that every comparison is paired (same CMB + noise realisation, same clusters):
  per noise-only tile pid = 0..NP-1 (CMB seed 300000 + pid, noise seed 400000 + pid) and noise level: a02's layout
  (a02_injections_hi.layout with k = 5000 + pid: one size per tile, cycling 0.3', 0.6', 1.2', 2.4', 4.8', 9.6';
  225 / 100 / 64 clusters on a jittered grid 40' / 60' / 75' apart; Y500 = q_inj sigma_Y(theta), q_inj uniform 1.8-8),
  painted into the tile's CMB + noise.
  (P)  the production finder run end to end on the dense tile; a final detection is credited to its nearest injected
       cluster when the filtered tSZ there is >= 2 sigma (a02's rule); q_obs = highest credited q;
  (Vd) the same clusters inserted virtually (s3blib.virtual_credit) into the dense tile's own final filter applied to the
       CMB + noise alone;
  (Vn) the same clusters inserted virtually into the noise-only tile's final filter (the kernel's reference).
  P vs Vd isolates painting, the cluster's own SZ in the covariance and its own mask at the iteration (filter fixed);
  Vd vs Vn isolates the filter change: the injected clusters' SZ power in the covariance and their masks.
Outputs: data_real/step3b/dense/d_MODE_PID.npz.  Usage: python b04_dense.py PID0 PID1 NPROC
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3blib as B                                 # noqa: E402
B.threads(1)
import numpy as np                                 # noqa: E402
from multiprocessing import Pool                   # noqa: E402
sys.path.insert(0, os.path.join(B.ROOT, "analysis", "realistic", "step3"))
import a02_injections_hi as A2                     # noqa: E402
import b03_virtual as B3                           # noqa: E402
F = B.F
K0 = 5000


def work(job):
    pid, mode = job
    fn = os.path.join(B.OUT, "dense", f"d_{mode}_{pid:03d}.npz")
    if os.path.exists(fn):
        return job, 0.0, B.peak_rss_gb()
    t0 = time.time()
    G = F.grid()
    bas, tables = B.basis()
    its_n, X = B.noise_only_filter(pid, mode, "prod", G)
    fn_, qn_n = its_n[-1]["filt"], its_n[-1]["q"]
    sig_n = fn_.sigma_arcmin2.copy()
    tabs_n = B.filter_tables(fn_, bas, G)
    th = float(A2.TH_INJ[pid % 6])
    y, x, q = A2.layout(th, K0 + pid)
    Y = q * B.sigma_Y(mode, th)
    ths = np.full(y.size, th)
    Xt = F.tsz_channels(F.paint_y(y, x, Y, ths, tables, G), G)
    its_d = F.run_finder(X + Xt, G)
    fd, dd = its_d[-1]["filt"], its_d[-1]["det"]
    sig_d = fd.sigma_arcmin2.copy()
    qo, nspur = B3.credit_injected(fd, dd, Xt, y, x)
    del Xt
    qn_d = fd.q_cube(fd.CX(X))
    tabs_d = B.filter_tables(fd, bas, G)
    W = B.window_pix(ths, B.TH_K, B3.RW)
    vd = B.run_virtual(qn_d, sig_d, tabs_d, y, x, Y, ths, W)
    vn = B.run_virtual(qn_n, sig_n, tabs_n, y, x, Y, ths, W)
    np.savez(fn, pid=pid, mode=mode, theta=th, y=y, x=x, Y=Y, q_inj=q, sig_d=sig_d, sig_n=sig_n, fsky_d=float(its_d[-1]["fsky"]),
             fsky_n=float(its_n[-1]["fsky"]), P=qo["none"], Vd=vd["q"], Vn=vn["q"], Vd_j=vd["j"], Vn_j=vn["j"], nspur=nspur,
             t=time.time() - t0)
    return job, time.time() - t0, B.peak_rss_gb()


if __name__ == "__main__":
    P0, P1, NPROC = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(os.path.join(B.OUT, "dense"), exist_ok=True)
    jobs = [(p, m) for p in range(P0, P1) for m in B.MODES]
    t00 = time.time()
    print(f"b04: dense injections (a02 design) on noise-only tiles {P0}..{P1 - 1} x {B.MODES}, {NPROC} processes", flush=True)
    rss_max = 0.0
    with Pool(NPROC, maxtasksperchild=40) as pool:
        for i, (job, tt, rss) in enumerate(pool.imap_unordered(work, jobs)):
            rss_max = max(rss_max, rss)
            if i % 40 == 0:
                print(f"  {job}: {tt:.1f} s, worker peak RSS {rss:.2f} GB [elapsed {time.time() - t00:.0f} s]", flush=True)
    print(f"b04 done: {len(jobs)} jobs in {time.time() - t00:.0f} s wall; largest worker peak RSS {rss_max:.2f} GB", flush=True)
