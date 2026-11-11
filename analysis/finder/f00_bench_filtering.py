"""Step-3 planning: core-cost benchmark (random inputs, not our sky). SZiFi-style per scale (szifi/mmf.py get_mmf_q_map): template FFT per channel, filter
= C^-1 (a B tau) per mode, inverse FFT of the filter per channel, scipy.signal.fftconvolve(map_i, filter_i, 'same') per
channel, real-space norm.  Lean: the template factorises, T_i(l) = a_i B_i(l) tau_theta(l), so
   num_theta(l) = tau_theta(l) * [ (C^-1 aB)^T X ](l)   and   norm_theta = sum_l tau^2 [aB^T C^-1 aB](l):
the bracket is computed ONCE for all scales; each scale costs one multiply and one inverse real FFT.
Check: both S/N maps agree in the tile interior (linear vs circular convolution differ only near the edges)."""
import os, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np, scipy.signal as sg, scipy.fft as sfft

N, NF, NS = 1200, 6, 12
dx = 0.5 / 60 * np.pi / 180
rng = np.random.default_rng(1)
t = rng.standard_normal((N, N, NF)).astype(np.float32)
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
l = np.hypot(kx[:, None], kx[None, :])
A = rng.standard_normal((NF, NF)); base = A @ A.T + NF * np.eye(NF)
icov = base[None, None] / (1 + (l[..., None, None] / 3000.0) ** 2)          # (N, N, 6, 6)
a = rng.standard_normal(NF)
fw = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9]) / 2.355 / 60 * np.pi / 180
B = np.exp(-0.5 * (l[..., None] * fw) ** 2)                                  # (N, N, 6)
prof = lambda th: np.exp(-0.5 * (l * th / 60 * np.pi / 180) ** 2)           # stand-in isotropic profile FT
ths = np.geomspace(0.5, 10, NS)
CHECK = [1, 8]

t0 = time.time(); q_sz = {}
for k, th in enumerate(ths):
    tem = np.real(np.fft.ifft2(prof(th)[..., None] * B, axes=(0, 1)))       # beam-convolved template, real space
    tem_fft = np.fft.fft2(tem, axes=(0, 1)) * a
    filt = np.fft.fftshift(np.real(np.fft.ifft2(np.einsum("dhi,dhij->dhj", tem_fft, icov), axes=(0, 1))), axes=(0, 1))
    y = np.zeros((N, N))
    for i in range(NF):
        y += sg.fftconvolve(t[:, :, i], filt[:, :, i].astype(np.float32), mode="same")
    norm = np.sum(np.fft.fftshift(tem, axes=(0, 1)) * a * filt)
    if k in CHECK:
        q_sz[k] = y / np.sqrt(norm)
t_sz = time.time() - t0

t0 = time.time()
Nr = N // 2 + 1
X = sfft.rfft2(t.astype(np.float64), axes=(0, 1), workers=2)
aB = a * B[:, :Nr]
CiaB = np.einsum("dhij,dhj->dhi", icov[:, :Nr], aB)
CX = np.einsum("dhi,dhi->dh", CiaB, X)
aCa = np.einsum("dhi,dhi->dh", CiaB, aB)
wcol = np.full(Nr, 2.0); wcol[0] = 1.0; wcol[-1] = 1.0
t_once = time.time() - t0
t0 = time.time(); q_ln = {}
for k, th in enumerate(ths):
    p = prof(th)[:, :Nr]
    norm = np.sum(wcol * p * p * aCa)
    q = sfft.irfft2(p * CX, s=(N, N), workers=2) * (N * N) / np.sqrt(norm * N * N)
    if k in CHECK:
        q_ln[k] = q
t_ln = time.time() - t0
print(f"SZiFi-style: {t_sz:.1f} s for {NS} scales ({t_sz / NS:.2f} s per scale)")
print(f"lean:        {t_once:.2f} s once + {t_ln:.2f} s for {NS} scales ({t_ln / NS:.3f} s per scale) -> x{t_sz / (t_once + t_ln):.0f} faster")
m = 100
for k in CHECK:
    A_, B_ = q_sz[k][m:-m, m:-m], q_ln[k][m:-m, m:-m]
    print(f"theta {ths[k]:.2f}': interior S/N rms {A_.std():.3f} vs {B_.std():.3f}; max |difference| {np.abs(A_ - B_).max():.2e}")
for k in CHECK:
    best = min(((np.abs(np.roll(q_sz[k], (sx, sy), (0, 1))[m:-m, m:-m] - q_ln[k][m:-m, m:-m]).max(), sx, sy)
                for sx in (-1, 0, 1) for sy in (-1, 0, 1)))
    print(f"theta {ths[k]:.2f}': best alignment shift {best[1:]} px -> max |difference| {best[0]:.2e}")
