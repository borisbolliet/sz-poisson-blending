"""v03: check (a), part 2 -- S/N maps, scale by scale, with the SAME covariance (the build's final, iteration-1,
binned covariance of each tile, the build's bin assignment), SZiFi conventions (our emulation) vs the build.

Per tile and scale:
  q_ref  : the build's S/N map evaluated at n - 1/2 (Fourier shift), i.e. at the position SZiFi's pixel n refers to;
  q_lb   : SZiFi's filtering (template centred at N/2 - 1/2, LINEAR 'same' convolution) of the BUILD's template;
  q_ls   : the same with SZiFi's own sampled template;
  q_cs   : SZiFi's template but periodic convolution (separates the template effect from the tile-edge effect);
  q_det  : SZiFi's template, periodic, centred, read at the build's final q > 4.5 detection pixels and scales.
Reported per scale: max |q_lb - q_ref| by distance from the tile edge (edge effect only); rms and max of q_cs - q_ref
beyond 30' from the edges, their correlation (template effect only); q_det - q_build at the detections.
Usage: python -I -B v03_maps.py MODE PID [PID ...]   (writes out/v03_MODE_PID.npz)
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "2"
sys.path.insert(0, HERE)
import numpy as np
import vsz_lib as V

F = V.F
mode = sys.argv[1]
pids = [int(a) for a in sys.argv[2:]]
g = V.szgrid()
G = F.grid()
N = V.N
TH = V.THETAS
EDGES = np.array([0.0, 2.5, 10.0, 30.0, 60.0, 301.0])       # arcmin from the nearest tile edge
ar = np.arange(N)
dpix = np.minimum(ar, N - 1 - ar)
dedge = (np.minimum(dpix[:, None], dpix[None, :]) + 0.5) * V.PIXA  # SZiFi pixel n refers to position n - 1/2
band_id = np.digitize(dedge, EDGES) - 1
inner = dedge >= 30.0
t0 = time.time()
for pid in pids:
    fn = os.path.join(V.OUTD, f"v03_{mode}_{pid:03d}.npz")
    if os.path.exists(fn):
        print(f"tile {pid}: exists", flush=True)
        continue
    st = V.load_stored_tile(pid)
    X, _ = V.make_tile(pid, mode)
    C1 = st[f"{mode}_it1_C"]
    filt = V.SZFilter(Cbin=C1, mbin=G.m_bin, g=g)
    Xm = V.to_modes(X, g)
    maps = V.lfiltered_maps(X, g)
    del X
    lin = V.LinearConv(maps)
    del maps
    nb_ = len(EDGES) - 1
    edge_max = np.zeros((len(TH), nb_)); edge_rms = np.zeros((len(TH), nb_)); edge_p999 = np.zeros((len(TH), nb_))
    tot_rms_b = np.zeros((len(TH), nb_)); tot_max_b = np.zeros((len(TH), nb_))
    tmpl_rms = np.zeros(len(TH)); tmpl_max = np.zeros(len(TH)); tmpl_corr = np.zeros(len(TH))
    tot_rms = np.zeros(len(TH)); tot_max_in = np.zeros(len(TH))
    sig_b = np.zeros(len(TH)); sig_s = np.zeros(len(TH))
    dj, diy, dix, dq = (st[f"{mode}_it1_{k}"] for k in ("j", "iy", "ix", "q"))
    sel = dq > 4.5
    q_det = np.full(dq.size, np.nan)
    for j, th in enumerate(TH):
        Tc = V.template_modes(th, "build", g)
        T0 = Tc * np.conj(g.m_phase_c)[None, :]
        F0, D0 = filt.weights(T0)
        q_ref = V.q_circular(F0, D0, Xm, shift_half=True, g=g)
        del F0
        Fc, Dc = filt.weights(Tc)
        q_lb = lin.q(Fc, Dc, g)
        del Fc, Tc, T0
        d = np.abs(q_lb - q_ref)
        for b in range(nb_):
            db = d[band_id == b]
            edge_max[j, b] = db.max(); edge_rms[j, b] = np.sqrt(np.mean(db ** 2)); edge_p999[j, b] = np.quantile(db, 0.999)
        del q_lb, d
        Ts = V.szifi_T(j, g)
        Fs, Ds = filt.weights(Ts)
        sig_b[j], sig_s[j] = 1 / np.sqrt(D0), 1 / np.sqrt(Ds)
        q_cs = np.roll(V.q_circular(Fs, Ds, Xm, g=g), (-(N // 2 - 1), -(N // 2 - 1)), axis=(0, 1))
        dd = (q_cs - q_ref)[inner]
        tmpl_rms[j] = np.sqrt(np.mean(dd ** 2)); tmpl_max[j] = np.abs(dd).max()
        tmpl_corr[j] = np.corrcoef(q_cs[inner], q_ref[inner])[0, 1]
        q_ls = lin.q(Fs, Ds, g)
        dl = (q_ls - q_ref)[inner]
        tot_rms[j] = np.sqrt(np.mean(dl ** 2)); tot_max_in[j] = np.abs(dl).max()
        dfull = np.abs(q_ls - q_ref)
        for b in range(nb_):
            db = dfull[band_id == b]
            tot_rms_b[j, b] = np.sqrt(np.mean(db ** 2)); tot_max_b[j, b] = db.max()
        del dfull
        del q_ls, q_cs, dd, dl, Fs
        Fs0, Ds0 = filt.weights(Ts * np.conj(g.m_phase_c)[None, :])
        q0 = V.q_circular(Fs0, Ds0, Xm, g=g)
        m = sel & (dj == j)
        q_det[m] = q0[diy[m], dix[m]]
        del Fs0, q0, Ts, q_ref
    np.savez(fn, pid=pid, edges=EDGES, edge_max=edge_max, edge_rms=edge_rms, edge_p999=edge_p999, tot_rms_b=tot_rms_b,
             tot_max_b=tot_max_b, tmpl_rms=tmpl_rms, tmpl_max=tmpl_max, tmpl_corr=tmpl_corr,
             tot_rms=tot_rms, tot_max_in=tot_max_in, sig_b=sig_b, sig_s=sig_s, q_det=q_det, q_build=dq, det_j=dj,
             det_sel=sel)
    print(f"tile {pid:3d} {mode}: edge effect max|dq| (build template) by edge band {EDGES.tolist()}': "
          + ", ".join(f"{x:.1e}" for x in edge_max.max(0)) + "; rms " + ", ".join(f"{x:.1e}" for x in edge_rms.max(0))
          + f"; template effect rms (>=30') "
          f"{tmpl_rms.min():.3f}-{tmpl_rms.max():.3f}; at detections q>4.5 mean dq {np.nanmean(q_det[sel] - dq[sel]):+.3f}"
          f"  [{time.time() - t0:.0f} s, RSS {V.peak_rss_gb():.2f} GB]", flush=True)
    del lin, Xm, filt
print(f"done [{time.time() - t0:.0f} s]; peak RSS {V.peak_rss_gb():.2f} GB")
