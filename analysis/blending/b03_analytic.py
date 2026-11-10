"""Analytic (compound-Poisson / P(D)) blending predictions.

Neighbour field around a halo A (Slivnyak: same as around a random point) is a Poisson process.
Each neighbour B adds x_B = rho(thA, thB, s) Y_B / sigma_Y(thA) to A's matched-filter SNR.
delta_A = sum_B x_B has characteristic function  exp( int dN [e^{ik x} - 1] ) -> FFT.

Conventions for the map mean (the white-noise matched filter is NOT compensated: its DC response is
int rho d^2s = sigma_A^2 RATIO^2 / N_w, e.g. 43 arcmin^2 at thA = 3'):
  c = 0 'raw'      : the filter sees the absolute y sky, q_b = q + n + delta
  c = 1 'monopole' : the map mean is removed (as in any real map), q_b = q + n + delta - mu(thA),
                     mu = <delta> = ybar * DC / sigma_A.

Detection in the blended catalogue (catalogue-level proxy for one detection per filter peak):
   det_b(A) = [q_b > q_cut] AND [A dominant: x_B < q_A for every neighbour B]
Dominance -> truncated compound Poisson:  P(dom) = exp(-Lambda(>q_A)), and conditional on dom the
sum is compound Poisson with the rate restricted to x < q_A.
Blending classes among det_b:
   B2(eps)   : (delta - c mu) / q_A >  eps   (Y_A over-estimated by > eps)
   B2neg(eps): (delta - c mu) / q_A < -eps   (only possible for c=1)
   B3        : q_A + n < q_cut  (would not be detected in isolation: boosted)
plus  'lost' (q+n > q_cut but q_b < q_cut, dominant) and 'satellite' (q+n > q_cut, not dominant).
Neighbour mass floors 1e13 / 3e13 / 1e14 (classes summed accordingly).
Output: data/blending/analytic.npz
"""
import os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blendlib as bl
from szsim import signal as sg, cosmo_hmf as ch

QCUTS = np.array([4.0, 5.0, 6.0, 7.0])
EPS = np.array([0.05, 0.1, 0.2, 0.5, 1.0])
DMIN = np.array([0.5, 1.0, 2.0])  # B3 restricted to boosts delta' > DMIN (SNR units)
QNODES = np.concatenate([np.arange(0.1, 20.0, 0.1), np.geomspace(20.0, 3000.0, 60)])
FLOORS = ["1e13", "3e13", "1e14"]
CCDF_T = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
Q_PRIM = 0.5


def primary_population(tab, qmin, qmax=np.inf, zsub=2, msub=2, ngh=24):
    """Expected full-sky counts of halos (M>1e13) CIC-binned on (lnth node, q node),
    restricted to qmin < q_A < qmax."""
    m = sg.SZModel()
    T = ch.default_table()
    z = T.z[::zsub]; lnM = T.lnM[::msub]; d2N = T.d2N[::zsub, ::msub]
    wz = np.full(len(z), z[1] - z[0]); wz[0] *= .5; wz[-1] *= .5
    wM = np.full(len(lnM), lnM[1] - lnM[0]); wM[0] *= .5; wM[-1] *= .5
    x, wg = np.polynomial.hermite_e.hermegauss(ngh); wg /= wg.sum()
    H = np.zeros((tab.NT, len(QNODES)))
    M = np.exp(lnM)
    for iz, zz in enumerate(z):
        th = m.theta500(M, zz)
        q0 = m.Y500_med(M, zz) / m.sigma_Y(th)
        q = (q0[:, None] * np.exp(m.sigma_lnY * x)[None, :]).ravel()
        w = (d2N[iz] * wz[iz] * wM)[:, None] * wg[None, :]
        w = w.ravel()
        tt = np.repeat(th, ngh)
        s = (q > qmin) & (q < qmax)
        q, w, tt = q[s], w[s], tt[s]
        fa = np.clip((np.log(tt) - tab.lth0) / tab.dlth, 0, tab.NT - 1.000001)
        ia = fa.astype(int); wa = fa - ia
        # quadratic (3-point Lagrange) weights in q: removes the O(dq^2) bias of linear CIC
        k = np.clip(np.searchsorted(QNODES, q) - 1, 0, len(QNODES) - 2)
        j0 = np.clip(np.where(q - QNODES[k] < QNODES[k + 1] - q, k - 1, k), 0, len(QNODES) - 3)
        x0, x1, x2 = QNODES[j0], QNODES[j0 + 1], QNODES[j0 + 2]
        L = [(q - x1) * (q - x2) / ((x0 - x1) * (x0 - x2)), (q - x0) * (q - x2) / ((x1 - x0) * (x1 - x2)),
             (q - x0) * (q - x1) / ((x2 - x0) * (x2 - x1))]
        for da, fa_ in ((0, 1 - wa), (1, wa)):
            for db in range(3):
                np.add.at(H, (ia + da, j0 + db), w * fa_ * L[db])
    return H


def main():
    t0 = time.time()
    tab = bl.load_rho_table()
    m = sg.SZModel()
    sigA = m.sigma_Y(tab.th)                     # same sigma_Y as used for catalogue q
    dc = bl.dc_response(sigA, m)
    pop = bl.neighbour_population(zsub=1, msub=6, ngh=16)  # full z resolution: low-z tail converged (b08)
    thB, YB, cB, wB = pop
    Tb = ch.default_table()
    print("neighbour population check (counts per class, full sky):")
    for c in range(3):
        lo, hi = bl.MASS_EDGES[c], min(bl.MASS_EDGES[c + 1], 1e16)
        print(f"   class {c}: grid {wB[cB == c].sum() * bl.ARCMIN2_FULLSKY:,.1f}  table {Tb.N_total(lo, hi):,.1f}")
    ybar = np.array([(wB * YB)[cB == c].sum() for c in range(3)])
    print("   ybar_c (sum Y500 per arcmin^2) =", ybar, " total", ybar.sum())

    # ---- rate histograms for every theta_A node
    NT = tab.NT
    nbins = len(bl.XLOG_EDGES) - 1
    W = np.zeros((NT, 3, nbins)); WX = np.zeros_like(W); WX2 = np.zeros_like(W)
    CAP = np.zeros((NT, 3)); CAPX = np.zeros((NT, 3))
    for i in range(NT):
        W[i], WX[i], WX2[i], CAP[i], CAPX[i] = bl.rate_histogram(tab.th[i], sigA[i], pop, tab)
    mu_hist = WX.sum(2) + CAPX                    # (NT, 3)
    mu_formula = ybar[None, :] * dc[:, None] / sigA[:, None]
    # monopole shift used everywhere = mean of delta under the SAME (interpolated) kernel as the sums
    mu_exact = mu_hist
    print(f"rate histograms {time.time()-t0:.1f}s; mean check  sum(w x) / (ybar DC/sigma) - 1: "
          f"max |.| = {np.abs(mu_hist / mu_formula - 1).max():.2e} (per class), total: "
          f"{np.abs(mu_hist.sum(1) / mu_formula.sum(1) - 1).max():.2e}")
    for i in np.searchsorted(tab.th, [0.5, 1, 2, 4, 8, 16]):
        mu = mu_hist[i]; lam = W[i].sum(1) + CAP[i]
        var = WX2[i].sum(1)
        print(f"   thA={tab.th[i]:6.2f}': mu_c = {np.round(mu, 4)} (tot {mu.sum():.3f}); "
              f"neighbours/filter = {np.round(lam, 1)}; var_c (x<500) = {np.round(var, 3)}; "
              f"Lambda(x>1) = {bl.PD.lam_above(W[i].sum(0), WX[i].sum(0), 1.0) + CAP[i].sum():.4f}")

    pdc = bl.PD(N=2 ** 20, dd=0.001, x_small=0.002)        # untruncated P(D): range 1048
    pdc_small = bl.PD(N=2 ** 18, dd=0.001, x_small=0.002)  # truncated at x < 20: range 262
    pdc_big = bl.PD(N=2 ** 19, dd=0.002, x_small=0.004)    # truncated at x >= 20: range 1048
    g = pdc.grid
    # ---- untruncated P(D): CCDF and quantiles at every node, each floor
    CCDF = np.zeros((len(FLOORS), NT, len(CCDF_T)))
    QUANT = np.zeros((len(FLOORS), NT, 5))
    for fi, fl in enumerate(FLOORS):
        cl = list(bl.FLOORS[fl])
        for i in range(NT):
            Wf, WXf, WX2f = W[i, cl].sum(0), WX[i, cl].sum(0), WX2[i, cl].sum(0)
            p = pdc.pdf(Wf, WXf, WX2f)
            lam_cap = CAP[i, cl].sum()
            p = p * np.exp(-lam_cap)              # mass beyond the grid cap (x>500) -> tail
            cdf = np.cumsum(p)
            tail = 1.0 - cdf                      # P(delta > grid_j)
            # [fix after verification] cumsum(p)[j] is P(delta <= g_j + dd/2): read at the cell edge
            CCDF[fi, i] = np.interp(CCDF_T, g + 0.5 * pdc.dd, tail)
            QUANT[fi, i] = np.interp([0.16, 0.5, 0.84, 0.977, 0.9987], cdf, g + 0.5 * pdc.dd)
            # [fix after verification] the dd=1e-3 grid (+2dd smoothing) cannot resolve medians ~0.006
            # (floor 1e14, small thA): high-resolution CDF below x_c (blendlib.pd_cdf_fine; same as r02_pd_fine.py)
            ddf = float(np.clip(QUANT[fi, i, 1] / 500.0, 1e-5, 5e-4))
            te, cdf_f, xc = bl.pd_cdf_fine(Wf, WXf, WX2f, lam_cap, ddf)
            qf = np.interp([0.16, 0.5, 0.84, 0.977, 0.9987], cdf_f, te)
            QUANT[fi, i] = np.where(qf < xc, qf, QUANT[fi, i])
            sm = CCDF_T < xc
            CCDF[fi, i, sm] = 1.0 - np.interp(CCDF_T[sm], te, cdf_f)
    print(f"untruncated P(D) done {time.time()-t0:.1f}s")

    # ---- primary population
    H = primary_population(tab, Q_PRIM)
    Hlow = primary_population(tab, 0.1, Q_PRIM)
    N_single_cic = [(H * ndtr(QNODES[None, :] - qc)).sum() for qc in QCUTS]
    N_lib = [m.expected_detections(qc) for qc in QCUTS]
    print("single-cluster detections: CIC grid (q>0.5) vs library expected_detections:")
    for qc, a, b in zip(QCUTS, N_single_cic, N_lib):
        print(f"   q_cut={qc}: {a:,.1f} vs {b:,.1f}  (rel {a/b-1:+.2e})")
    print(f"primary population done {time.time()-t0:.1f}s, total H = {H.sum():,.0f}, low-q H = {Hlow.sum():,.0f}")

    # ---- per-node blending probabilities
    nq = len(QNODES); nF = len(FLOORS); nC = len(QCUTS); nE = len(EPS)
    shp = (nF, 2, nC, NT, nq)
    Pdom = np.zeros((nF, NT, nq))
    Pdet = np.zeros(shp); Pb3 = np.zeros(shp); Plost = np.zeros(shp)
    Pb2 = np.zeros(shp + (nE,)); Pb2n = np.zeros(shp + (nE,)); Pb3d = np.zeros(shp + (len(DMIN),))
    for fi, fl in enumerate(FLOORS):
        cl = list(bl.FLOORS[fl])
        for i in range(NT):
            if np.abs(H[i]).sum() + np.abs(Hlow[i]).sum() == 0:
                continue
            Wf, WXf, WX2f = W[i, cl].sum(0), WX[i, cl].sum(0), WX2[i, cl].sum(0)
            mu_tot = mu_exact[i, cl].sum()
            for k, qa in enumerate(QNODES):
                if H[i, k] == 0 and Hlow[i, k] == 0:
                    continue
                xcut = min(qa, bl.X_CAP)
                lam_above = bl.PD.lam_above(Wf, WXf, xcut) + CAP[i, cl].sum()
                Pdom[fi, i, k] = np.exp(-lam_above)
                pd_ = pdc_small if xcut < 20 else pdc_big
                p = pd_.pdf(Wf, WXf, WX2f, xcut=xcut)
                g = pd_.grid; jmax = len(g)
                p /= p.sum()
                tailsum = np.concatenate([np.cumsum(p[::-1])[::-1], [0.0]])
                for c in (0, 1):
                    sh = c * mu_tot
                    for ic, qc in enumerate(QCUTS):
                        # I(j) = sum_{j'>=j} p_j' Phi(q_A + g_j' - sh - q_cut)
                        J = min(jmax, int(np.ceil((qc - qa + sh + 8.5 + pd_.off) / pd_.dd)) + 1)
                        J = max(J, 1)
                        pg = p[:J] * ndtr(qa + g[:J] - sh - qc)
                        part = np.concatenate([np.cumsum(pg[::-1])[::-1], [0.0]])

                        def I(jt):
                            jt = int(min(max(jt, 0), jmax))
                            return tailsum[jt] if jt >= J else part[jt] + tailsum[J]

                        def jidx(d):  # first grid index with g_j > d
                            return int(np.floor((d + pd_.off) / pd_.dd)) + 1

                        Iall = I(0)
                        Pdet[fi, c, ic, i, k] = Iall
                        j0 = jidx(sh)
                        Ipos = I(j0)
                        phi0 = ndtr(qa - qc)
                        Pb3[fi, c, ic, i, k] = max(Ipos - phi0 * tailsum[min(j0, jmax)], 0.0)
                        Plost[fi, c, ic, i, k] = max(phi0 * (1 - tailsum[min(j0, jmax)]) - (Iall - Ipos), 0.0)
                        for idm, dm in enumerate(DMIN):
                            jd = jidx(sh + dm)
                            Pb3d[fi, c, ic, i, k, idm] = max(I(jd) - phi0 * tailsum[min(jd, jmax)], 0.0)
                        for ie, e in enumerate(EPS):
                            Pb2[fi, c, ic, i, k, ie] = I(jidx(sh + e * qa))
                            Pb2n[fi, c, ic, i, k, ie] = Iall - I(jidx(sh - e * qa))
        print(f"floor {fl} done {time.time()-t0:.1f}s", flush=True)
    for arr in (Pdet, Pb3, Plost):
        arr *= Pdom[:, None, None, :, :]
    for arr in (Pb2, Pb2n, Pb3d):
        arr *= Pdom[:, None, None, :, :, None]
    Phi_single = ndtr(QNODES[None, :] - QCUTS[:, None])           # (nC, nq)
    Psat = (1 - Pdom)[:, None, :, :] * Phi_single[None, :, None, :]  # (nF, nC, NT, nq)

    np.savez(os.path.join(bl.DATA, "analytic.npz"), QCUTS=QCUTS, EPS=EPS, QNODES=QNODES, th=tab.th,
             sigA=sigA, dc=dc, ybar=ybar, mu_exact=mu_exact, mu_formula=mu_formula, W=W, WX=WX, WX2=WX2, CAP=CAP, CAPX=CAPX,
             CCDF_T=CCDF_T, CCDF=CCDF, QUANT=QUANT, H=H, Hlow=Hlow, Pdom=Pdom, Pdet=Pdet, Pb3=Pb3,
             Plost=Plost, Pb2=Pb2, Pb2n=Pb2n, Pb3d=Pb3d, DMIN=DMIN, Psat=Psat, Phi_single=Phi_single, FLOORS=np.array(FLOORS))

    # ---- summary table
    for c, cname in ((0, "raw"), (1, "monopole-removed")):
        print(f"\n=== convention c={c} ({cname}); primaries q_A>{Q_PRIM}; full-sky expected numbers ===")
        for fi, fl in enumerate(FLOORS):
            print(f" neighbour floor M>{fl}")
            for ic, qc in enumerate(QCUTS):
                Ns = (H * Phi_single[ic][None, :]).sum()
                Nd = (H * Pdet[fi, c, ic]).sum()
                b2 = [(H * Pb2[fi, c, ic, :, :, ie]).sum() / Nd for ie in range(nE)]
                b2n = [(H * Pb2n[fi, c, ic, :, :, ie]).sum() / Nd for ie in range(nE)]
                b3 = (H * Pb3[fi, c, ic]).sum() / Nd
                b3d = [(H * Pb3d[fi, c, ic, :, :, j]).sum() / Nd for j in range(len(DMIN))]
                lost = (H * Plost[fi, c, ic]).sum() / Ns
                sat = (H * Psat[fi, ic]).sum() / Ns
                b3low = (Hlow * Pb3[fi, c, ic]).sum()
                print(f"  q_cut={qc:.0f}: N_single={Ns:,.0f} N_det_b={Nd:,.0f} ({Nd/Ns-1:+.4f}); "
                      f"B2(eps={list(EPS)})={np.round(b2, 5)}; B2neg={np.round(b2n, 5)}; B3={b3:.5f}; "
                      f"lost/N_single={lost:.5f}; sat/N_single={sat:.5f}; B3 from 0.1<q<0.5: {b3low:.2f}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
