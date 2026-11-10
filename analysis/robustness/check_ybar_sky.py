"""Sky-realised mean y, ybar_sky = RATIO * sum(Y500) / 4pi, vs the analytic ensemble mean (Campbell).
The probe-mean of dY at random points is exactly sigma_A^2 RATIO ybar_sky / N_w (int rho d^2sep is
independent of theta_B), so this separates sampler fluctuations from kernel errors."""
import numpy as np
from szsim import sampler, signal as sg, cosmo_hmf as ch
import blendlib as bl
model = sg.SZModel(); table = ch.default_table()
yan = bl.mean_ybar(model, table, 1e13)
print(f"analytic ybar(M>1e13) = {yan:.5e}")
vals = []
for seed in [1001, 1002, 1003, 1004, 1005, 1006]:
    c = sampler.draw_catalog(seed, Mmin=1e13, model=model, fields=("Y500",))
    yb = sg.RATIO_5R500 * c["Y500"].astype(np.float64).sum() * bl.ARCMIN ** 2 / (4 * np.pi)
    vals.append(yb)
    print(f"seed {seed}: ybar_sky = {yb:.5e}  ratio to analytic = {yb/yan:.4f}")
vals = np.array(vals)
print(f"mean ratio = {vals.mean()/yan:.4f} +- {vals.std(ddof=1)/np.sqrt(len(vals))/yan:.4f}; sky-to-sky rms = {vals.std(ddof=1)/yan:.4f}")
# predicted sky-to-sky rms: sqrt(sum <Y^2>)*RATIO/4pi
M = np.exp(table.lnM)
Y2 = np.array([model.Y500_med(M, zz) ** 2 for zz in table.z]) * np.exp(2 * model.sigma_lnY ** 2) * bl.ARCMIN ** 4
wM = table._mass_weights(1e13, None); wz = table._weights(table.z, table.zmin, table.zmax)
print(f"predicted sky-to-sky rms of ybar_sky / ybar = {np.sqrt(wz @ (table.d2N * Y2) @ wM) * sg.RATIO_5R500 / (4*np.pi) / yan:.4f}")
