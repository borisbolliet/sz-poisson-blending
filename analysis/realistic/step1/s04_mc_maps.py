"""Step 1 / s04: map-level Monte Carlo validation of the MMF (independent of the l-integral code path).

Flat patch 1024 x 1024, 0.5' pixels (8.5 deg; fundamental l = 42, Nyquist l = 21600).  Six SO LAT channels
are simulated COMPONENT BY COMPONENT (not from the total covariance):
   * CMB: one Gaussian field with the CAMB lensed C_l, convolved with each channel's beam (fully correlated);
   * white noise: independent pixel noise, sigma_pix = w [uK-arcmin] / 0.5';
   * atmosphere: per optics tube two unit fields xi_a, xi_b; channel i = sqrt(Nred_i(l)) xi_a,
     channel j = sqrt(Nred_j(l)) (r xi_a + sqrt(1-r^2) xi_b), r = 0.9.
The MMF is built on the FFT grid from the per-mode 6x6 covariance (np.linalg.inv per mode, explicit
Psi_i = sigma^2 tau(l) [C^-1 a]_i form of Melin+2006) with the same l cut [100, 20000].
Tests:
 (a) filtered-map variance (all pixels, nsim maps) vs the discrete-mode prediction and vs the library
     sigma_Y_mmf (continuum Hankel integral)  [for theta500 = 1, 2, 4, 8, 16'];
 (b) injected cluster (Y = 20 sigma) recovered without bias;
 (c) noiseless cross-response: cluster B (theta_B) at separation s, filter A at the origin, vs
     MMFModel.cross_response; uniform y offset -> Y_hat = 0 (compensation).
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np
from szsim import instrument as ins, signal as sg

NPIX, PIX = 1024, 0.5           # pixels, arcmin
AR = ins.ARCMIN
LMIN, LMAX = 100.0, 20000.0
THS = [1.0, 2.0, 4.0, 8.0, 16.0]
NSIM = int(os.environ.get("NSIM", 60))


def grid():
    k = np.fft.fftfreq(NPIX, d=PIX * AR) * 2 * np.pi
    lx, ly = np.meshgrid(k, np.fft.rfftfreq(NPIX, d=PIX * AR) * 2 * np.pi, indexing="ij")
    return lx, ly, np.hypot(lx, ly)


def main(mode):
    t0 = time.time()
    rng = np.random.default_rng(2026 if mode == "baseline" else 2027)
    lx, ly, L = grid()
    Omega_p = (PIX * AR) ** 2
    Npix = NPIX * NPIX
    fw, freqs = ins.SO_LAT["fwhm"], ins.SO_LAT["freqs"]
    Bl = np.exp(-0.5 * (L[None] * (fw[:, None, None] * AR / np.sqrt(8 * np.log(2)))) ** 2)   # (6, nx, ny)
    f = ins.tsz_f_uK(freqs)
    Ccmb = ins.cmb_cl_TT(L.ravel()).reshape(L.shape)
    A = 4 * np.pi * ins.SO_LAT["fsky"]
    with np.errstate(divide="ignore"):
        shape = np.where(L > 0, (L / 1000.0) ** -3.5, 0.0)
    red = ins.SO_LAT["N_red"] * A / ins.so_t_eff()
    wpix = ins.SO_LAT["white"][mode] / PIX                                   # uK per pixel
    r = ins.SO_LAT["atm_corr"]

    # ---- per-mode covariance of the (rfft, unnormalised) Fourier coefficients: E|X|^2 = P * Npix / Omega_p
    C = np.zeros(L.shape + (6, 6))
    for i in range(6):
        C[..., i, i] += (ins.SO_LAT["white"][mode][i] * AR) ** 2 + red[i] * shape
        for j in range(6):
            C[..., i, j] += Bl[i] * Bl[j] * Ccmb
    for (i, j) in ins.SO_LAT["tubes"]:
        C[..., i, j] += r * np.sqrt(red[i] * red[j]) * shape
        C[..., j, i] += r * np.sqrt(red[i] * red[j]) * shape
    C *= Npix / Omega_p
    sel = (L >= LMIN) & (L <= LMAX)
    Cinv = np.zeros_like(C)
    Cinv[sel] = np.linalg.inv(C[sel])
    a = np.moveaxis(Bl * f[:, None, None], 0, -1)                           # (nx, ny, 6)
    Cia = np.einsum("...ij,...j->...i", Cinv, a)                             # C^-1 a
    aCa = np.einsum("...i,...i->...", a, Cia)
    # rfft weights: columns ky=0 and ky=Nyquist appear once, others twice (Hermitian partner)
    wcol = np.full(L.shape[1], 2.0); wcol[0] = 1.0; wcol[-1] = 1.0
    wgt = np.broadcast_to(wcol[None, :], L.shape)

    filt, sig_disc = {}, {}
    m_lib = ins.MMFModel(ins.Instrument.so_lat(mode, lmin=LMIN, lmax=LMAX))
    for th in THS:
        tau = sg.template_hankel(L * th * AR) / Omega_p                     # FT of unit-Y500 template, per pixel units
        # Y_hat = sum_modes w Re[conj(tau a)^T C^-1 X] / sum_modes w |tau|^2 a^T C^-1 a
        norm = np.sum(wgt * tau ** 2 * aCa)
        filt[th] = (tau[..., None] * Cia / norm)                            # Psi on the grid (nx, ny, 6)
        sig_disc[th] = np.sqrt(1.0 / norm)                                  # discrete prediction [Y500 units = sr]
    print(f"[{mode}] grid set-up {time.time()-t0:.1f}s; nsim = {NSIM}")

    def apply(X, th):
        """Y_hat map (all positions) from channel rffts X (6, nx, ny)."""
        Yk = np.einsum("xyi,ixy->xy", filt[th], X)
        return np.fft.irfft2(Yk, s=(NPIX, NPIX)) * Npix                    # sum over modes incl. partners

    # ---------------------------------------------------------------- (a) + (b)
    var = {th: [] for th in THS}
    rec = {th: [] for th in THS}
    for n in range(NSIM):
        X = np.zeros((6,) + L.shape, complex)
        g = np.fft.rfft2(rng.standard_normal((NPIX, NPIX))) * np.sqrt(np.where(L > 0, Ccmb, 0) / Omega_p)
        X += Bl * g[None]
        for i in range(6):
            X[i] += np.fft.rfft2(rng.standard_normal((NPIX, NPIX)) * wpix[i])
        for (i, j) in ins.SO_LAT["tubes"]:
            xa = np.fft.rfft2(rng.standard_normal((NPIX, NPIX))) / np.sqrt(Omega_p)
            xb = np.fft.rfft2(rng.standard_normal((NPIX, NPIX))) / np.sqrt(Omega_p)
            X[i] += np.sqrt(red[i] * shape) * xa
            X[j] += np.sqrt(red[j] * shape) * (r * xa + np.sqrt(1 - r * r) * xb)
        for th in THS:
            Ym = apply(X, th)
            var[th].append(Ym.var())
            # (b) inject a Y = 20 sigma cluster at the patch centre into the same noise: Y_hat at the centre
            Yinj = 20 * sig_disc[th]
            ph = np.exp(-1j * (lx + ly) * (NPIX // 2) * PIX * AR)
            Xc = a.transpose(2, 0, 1) * (Yinj * sg.template_hankel(L * th * AR) / Omega_p * ph)[None]
            Ys = apply(X + Xc, th)[NPIX // 2, NPIX // 2]
            rec[th].append((Ys - Yinj) / sig_disc[th])
    print(f"[{mode}] (a) filtered-map variance vs predictions; (b) injected Y=20 sigma recovery  [{time.time()-t0:.0f}s]")
    for th in THS:
        v = np.array(var[th]); sd = np.sqrt(v.mean())
        lib = m_lib.sigma_Y_exact(th)[0] * AR ** 2
        rr = np.array(rec[th])
        print(f"   theta500={th:5.1f}': MC std {sd:.4e} (+- {0.5*v.std()/v.mean()/np.sqrt(len(v)):.2e} rel. from sim scatter)  "
              f"discrete-mode {sig_disc[th]:.4e}  library(Hankel) {lib:.4e} | MC/disc-1 = {sd/sig_disc[th]-1:+.4f}, "
              f"disc/lib-1 = {sig_disc[th]/lib-1:+.4f} | injected: mean (Yhat-Y)/sigma = {rr.mean():+.3f} +- {rr.std()/np.sqrt(len(rr)):.3f}, "
              f"std {rr.std():.3f}")

    # ---------------------------------------------------------------- (c) noiseless cross-response
    print(f"[{mode}] (c) noiseless cross-response: 2D FFT grid vs library Hankel")
    worst = 0.0
    for thA, thB in [(1.0, 1.0), (2.0, 4.0), (4.0, 2.0), (8.0, 8.0), (2.0, 16.0)]:
        seps = np.array([0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0])
        # filter A at origin; cluster B at (s, 0): Y_hat_A = sum Psi_A^* X_B
        tauB = sg.template_hankel(L * thB * AR) / Omega_p
        out = []
        for s in seps:
            XB = a * (tauB * np.exp(-1j * lx * s * AR))[..., None]
            out.append(np.sum(wgt * np.real(np.einsum("xyi,xyi->xy", filt[thA], XB))))
        out = np.array(out)
        lib = m_lib.cross_response(thA, thB, seps)
        d = np.abs(out - lib).max()
        worst = max(worst, d)
        print(f"   thA={thA:4.1f} thB={thB:4.1f}: grid {np.array2string(out, precision=4)}\n"
              f"                         lib  {np.array2string(lib, precision=4)}   max|diff| = {d:.1e}")
    print(f"   worst |grid - library| = {worst:.1e} (rho(0) = 1 for thA = thB)")
    Xu = a * 0.0
    Xu[0, 0, :] = a[0, 0, :] * 1e-6 * Npix              # uniform y = 1e-6: only the l=0 mode
    print(f"   uniform y = 1e-6 -> Y_hat = {np.sum(wgt * np.real(np.einsum('xyi,xyi->xy', filt[2.0], Xu))):.2e} "
          f"(library dc_response(2') = {m_lib.dc_response(2.0)[0]:.1e})")
    print(f"[{mode}] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    for mode in sys.argv[1:] or ["baseline", "goal"]:
        main(mode)
