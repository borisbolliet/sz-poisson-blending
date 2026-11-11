"""V5: ingredient sizes recomputed with my population + my G interpolation: (i) share of q_obs>5 detections from
primaries with q_A<5 (and <3), blended vs naive; (iv) raw sky (map mean not removed): q_obs = q + n + delta_tr,
i.e. kernel CDF shifted by mu(theta): F_raw(t) = G(t - mu); (iii) arithmetic of the noise-peak term."""
import numpy as np, vlib as v
pop = v.population()
K = v.kernel(); mu_nodes = np.load(f"{v.ROOT}/data_cosmo/G_fid_b03rates.npz")["mu"]
Nb = v.counts_blended(pop); Nn = v.counts_gauss(pop)
for qc in (5.0, 3.0):
    s = pop["q"] < qc
    sub = {k: (val[s] if isinstance(val, np.ndarray) else val) for k, val in pop.items()}
    print(f"(i) share of q_obs>5 detections with q_A<{qc:g}: blended {100*v.counts_blended(sub).sum()/Nb.sum():.2f}%  naive {100*v.counts_gauss(sub).sum()/Nn.sum():.2f}%")
# raw sky: P(q_obs in [a,b]) = G(b - q - mu) - G(a - q - mu)  <=> evaluate counts with q -> q + mu(theta) in the t-argument only
mu_p = np.interp(np.log(pop["th"]), K["lth"], mu_nodes)
# trick: counts_blended uses t = e - q and kernel nodes at q; build a shifted call by passing edges per point is not
# supported, so replicate the interpolation here
G = K["G"]; qe = v.QE
ia, fa = v._lin(np.log(pop["th"]), K["lth"]); ik, fk = v._lin(pop["q"], K["q"])
F = np.empty((pop["q"].size, len(qe)))
for j, e in enumerate(qe):
    t = np.full(pop["q"].size, K["t"][-1]) if not np.isfinite(e) else e - pop["q"] - mu_p
    it, ft = v._lin(t, K["t"]); val = 0.0
    for da, wa in ((0, 1 - fa), (1, fa)):
        for dk, wk in ((0, 1 - fk), (1, fk)):
            val = val + wa * wk * ((1 - ft) * G[ia + da, ik + dk, it] + ft * G[ia + da, ik + dk, it + 1])
    F[:, j] = np.where(t < K["t"][0], 0.0, val)
Nraw = v.bin_counts(np.diff(F, axis=1), pop["zb"], pop["w"])
print(f"(iv) raw sky (mean not removed): total {Nraw.sum():,.1f} = {100*(Nraw.sum()/Nb.sum()-1):+.2f}% vs blended; "
      f"Cash(D=blend, mu=raw) full sky {v.cash(Nb, Nraw):.0f}, sum pull^2 {np.sum((Nraw-Nb)**2/Nb):.0f}")
full = 4 * np.pi * (180 / np.pi) ** 2
rate = 192 / 32400.0
n5 = Nb.sum(0)[0]
for fs in (1.0, 0.4):
    nn = rate * full * fs
    print(f"(iii) f_sky={fs}: noise peaks {nn:.1f}, +{100*nn/(n5*fs):.2f}% of 5<q<6, {nn/np.sqrt(n5*fs):.2f} sigma (q-marginal)")
