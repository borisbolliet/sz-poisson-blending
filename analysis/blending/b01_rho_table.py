"""Build and validate the matched-filter cross-response table rho(theta_A, theta_B, s).

rho_AB(s) = Y_hat of A's matched filter (centred on A, tuned to theta_A) per unit Y500 of a
neighbour B of size theta_B at separation s (szsim.signal.SZModel.cross_response, validated in
analysis/build against a 2D pixel filter to 2.2e-4).

Support: tau_A and tau_B are truncated at 5 theta500 and the filtered beam kernel (B_l^2) is a
Gaussian of width sqrt(2) sigma_b = 0.84'; we take rho = 0 for s > S_AB = 5 thA + 5 thB + R_BEAM,
R_BEAM = 5 sqrt(2) sigma_b (= 4.20').

Table: lnthA, lnthB on a common log grid (0.25'-80', NT nodes); s = S_AB * t^2, t uniform (NU nodes).
Symmetry used: rho_AB * sigma_A^-2 is symmetric in (A,B) -> rho_BA = rho_AB * sigma_B^2 / sigma_A^2.

Checks printed:
 (1) table integral  int 2 pi s rho ds  vs exact  sigma_A^2 RATIO^2 / N_w  (DC response)
 (2) trilinear interpolation vs exact cross_response at random (thA, thB, s)
 (3) tail: |rho| at s = S_AB relative to rho(0)
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from szsim import signal as sg, cosmo_hmf as ch

# RHO_NT / RHO_OUT env overrides: used by r05_kernel_convergence (NT=79 nests the NT=40 nodes)
NT, NU = int(os.environ.get("RHO_NT", 40)), 241
OUT = os.path.join(ch.DATA_DIR, "blending", os.environ.get("RHO_OUT", "rho_table.npz"))
NL = 80000  # l-integration nodes (20000 gave 1.5% DC-response error at theta=80')
TH = np.geomspace(0.25, 80.0, NT)
T = np.linspace(0.0, 1.0, NU)
m = sg.SZModel()
R_BEAM = 5 * np.sqrt(2) * m.beam_fwhm_arcmin / np.sqrt(8 * np.log(2))


def S_of(a, b):
    return 5 * a + 5 * b + R_BEAM


def job(ij):
    i, j = ij
    a, b = TH[i], TH[j]
    s = S_of(a, b) * T ** 2
    return i, j, m.cross_response(a, b, s, nl=NL)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time()
    sig = m.sigma_Y_exact(TH)
    pairs = [(i, j) for i in range(NT) for j in range(i, NT)]
    rho = np.zeros((NT, NT, NU))
    with Pool(2) as p:
        for i, j, r in p.imap_unordered(job, pairs, chunksize=4):
            rho[i, j] = r
            if j != i:
                rho[j, i] = r * sig[j] ** 2 / sig[i] ** 2
    print(f"table built: {len(pairs)} Hankel pairs, {time.time()-t0:.1f}s")
    np.savez(OUT, th=TH, t=T, rho=rho, sigY=sig, R_BEAM=R_BEAM)

    # (1) DC response check
    dc_exact = (sig * ch.ARCMIN ** 2) ** 2 * sg.RATIO_5R500 ** 2 / m.N_white_sr / ch.ARCMIN ** 2
    worst = 0
    for i in range(NT):
        for j in range(NT):
            S = S_of(TH[i], TH[j])
            s = S * T ** 2
            I = np.trapezoid(2 * np.pi * s * rho[i, j], s)
            worst = max(worst, abs(I / dc_exact[i] - 1))
    print(f"(1) DC response int 2pi s rho ds / (sigma_A^2 RATIO^2/N_w) - 1 : max over {NT*NT} pairs = {worst:.2e}")
    # [fix after verification] evaluate the DC response exactly at the quoted theta_A (the previous
    # np.interp in ln(theta) of the node values biased this convex function high by up to 1%)
    th_q = np.array([0.25, 1, 3, 10, 30])
    dc_q = (m.sigma_Y_exact(th_q) * ch.ARCMIN ** 2) ** 2 * sg.RATIO_5R500 ** 2 / m.N_white_sr / ch.ARCMIN ** 2
    print(f"    DC response [arcmin^2] at thA = 0.25, 1, 3, 10, 30' (exact): ", np.round(dc_q, 3))

    # (3) tail
    tail = np.max(np.abs(rho[:, :, -1]) / np.abs(rho[:, :, 0]))
    neg = (rho.min(axis=2) / rho[:, :, 0]).min()
    print(f"(3) max |rho(S_AB)|/rho(0) = {tail:.2e};  most negative rho(s)/rho(0) over pairs = {neg:.2e}")

    # (2) random interpolation check (python trilinear, same as numba kernel in blendlib)
    import blendlib as bl
    tab = bl.RhoTable(np.load(OUT))
    rng = np.random.default_rng(7)
    errs, errs_rel0 = [], []
    for k in range(60):
        a = float(np.exp(rng.uniform(np.log(0.26), np.log(70))))
        b = float(np.exp(rng.uniform(np.log(0.26), np.log(70))))
        S = S_of(a, b)
        s = np.sort(rng.uniform(0, 1, 8) ** 2 * S)
        ex = m.cross_response(a, b, s, nl=NL)
        ip = np.array([bl.rho_interp_py(tab, a, b, ss) for ss in s])
        r0 = m.cross_response(a, b, [0.0], nl=NL)[0]
        errs_rel0.append(np.max(np.abs(ip - ex)) / r0)
        # integral-weighted error: int 2pi s rho
        sg_ = S * T ** 2
        I_ex = np.trapezoid(2 * np.pi * sg_ * m.cross_response(a, b, sg_, nl=NL), sg_)
        I_ip = np.trapezoid(2 * np.pi * sg_ * np.array([bl.rho_interp_py(tab, a, b, x) for x in sg_]), sg_)
        errs.append(I_ip / I_ex - 1)
    errs_rel0 = np.array(errs_rel0); errs = np.array(errs)
    print(f"(2) 60 random (thA,thB) x 8 s: max|rho_interp - rho_exact|/rho(0): median {np.median(errs_rel0):.2e}, "
          f"95% {np.percentile(errs_rel0, 95):.2e}, max {errs_rel0.max():.2e}")
    print(f"    area integral of interpolated vs exact rho: median |rel| {np.median(np.abs(errs)):.2e}, max {np.abs(errs).max():.2e}")
    print(f"total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
