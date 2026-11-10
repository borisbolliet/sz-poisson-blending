import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paircount import count_within
rng = np.random.default_rng(1)
n = 4000
ra = 360 * rng.random(n); dec = np.degrees(np.arcsin(2 * rng.random(n) - 1))
# add clustered points near poles and wrap
ra = np.concatenate([ra, rng.random(300) * 0.3, 359.9 + rng.random(300) * 0.1, rng.random(200) * 360]); dec = np.concatenate([dec, rng.random(300) * 0.5, rng.random(300) * 0.5, 89.5 + rng.random(200) * 0.5])
th = np.array([1.0, 5.0, 30.0, 120.0])
c = count_within(ra, dec, th)
x = np.column_stack([np.cos(np.radians(dec)) * np.cos(np.radians(ra)), np.cos(np.radians(dec)) * np.sin(np.radians(ra)), np.sin(np.radians(dec))])
ang = np.degrees(np.arccos(np.clip(x @ x.T, -1, 1))) * 60
np.fill_diagonal(ang, 1e9)
bf = np.stack([(ang < t).sum(1) for t in th], 1)
print("paircount vs brute force: max |diff| =", np.abs(c - bf).max(), " total counts", bf.sum(0))
