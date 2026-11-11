"""V1: independent recomputation of naive and blended dN/dz/dq at the fiducial; comparison with c02."""
import time, json
import numpy as np
import vlib as v
t0 = time.time()
pop = v.population()
print(f"population: {pop['q'].size:,} points, {time.time()-t0:.1f}s")
Nn = v.counts_gauss(pop); t1 = time.time(); Nb = v.counts_blended(pop)
print(f"naive+blended counts {time.time()-t1:.1f}s")
# resolution test
pop2 = v.population(nd=v.nodes(nzper=8, nsub=10, nM=300, ngh=32))
Nn2 = v.counts_gauss(pop2); Nb2 = v.counts_blended(pop2)
print(f"resolution test (80 z/bin, 300 M, 32 GH): Nn rel {Nn2.sum()/Nn.sum()-1:+.2e}, Nb rel {Nb2.sum()/Nb.sum()-1:+.2e},"
      f" max per-bin |rel| naive {np.nanmax(np.abs(Nn2/Nn-1)):.2e}")
C = np.load(f"{v.ROOT}/data_cosmo/counts_fid.npz")
print(f"N_naive(q>5): mine {Nn.sum():,.1f}  c02 {C['Nn'].sum():,.1f}  rel {Nn.sum()/C['Nn'].sum()-1:+.2e}")
print(f"N_blend(q>5): mine {Nb.sum():,.1f}  c02 {C['Nb'].sum():,.1f}  rel {Nb.sum()/C['Nb'].sum()-1:+.2e}")
print(f"ratio blended/naive: mine {Nb.sum()/Nn.sum():.5f}  c02 {C['Nb'].sum()/C['Nn'].sum():.5f}")
print(f"max per-bin rel diff mine vs c02: naive {np.max(np.abs(Nn/C['Nn']-1)):.2e} (excluding N<1: {np.max(np.abs(Nn/C['Nn']-1)[C['Nn']>1]):.2e}),"
      f" blended {np.max(np.abs(Nb/C['Nb']-1)[C['Nb']>1]):.2e}")
R = 100 * (Nb / Nn - 1)
print("blended/naive-1 [%%] (mine): z0.01-0.1,q5-6 = %.2f ; z2-3,q7-8 = %.2f ; min %.2f max %.2f" % (R[0, 0], R[-1, 2], R.min(), R.max()))
print("diff of my ratio table vs c02 ratio table (pct points): max |d| = %.3f" % np.max(np.abs(R - 100 * (C['Nb'] / C['Nn'] - 1))[C['Nn'] > 1]))
for fs in (1.0, 0.4):
    for lab, den in (("sqrt(N_naive)", Nn), ("sqrt(N_blend)", Nb)):
        P = (Nb - Nn) * fs / np.sqrt(den * fs)
        print(f"f_sky={fs}: sum pull^2 with {lab} = {np.sum(P**2):.1f}; max|pull| {np.abs(P).max():.2f}; "
              f"Cash(D=blend, mu=naive) = {v.cash(Nb*fs, Nn*fs):.1f}")
qm_n, qm_b = Nn.sum(0), Nb.sum(0)
print("q marginal rel diff [%]:", np.round(100 * (qm_b / qm_n - 1), 2))
print("q 5-6 marginal pull full / 40%%: %.2f / %.2f" % ((qm_b[0] - qm_n[0]) / np.sqrt(qm_n[0]), (qm_b[0] - qm_n[0]) * 0.4 / np.sqrt(0.4 * qm_n[0])))
zm_n, zm_b = Nn.sum(1), Nb.sum(1)
print("z marginal rel diff [%]:", np.round(100 * (zm_b / zm_n - 1), 2))
np.savez("/private/tmp/claude-501/-Users-boris-GitHub/0935154e-1cde-4eab-ab64-4515ca671a83/scratchpad/v1_counts.npz", Nn=Nn, Nb=Nb)
print(f"done {time.time()-t0:.0f}s")
