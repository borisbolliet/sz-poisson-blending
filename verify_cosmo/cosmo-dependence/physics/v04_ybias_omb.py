"""Order-of-magnitude cross-check: a fractional Y overestimate b_Y on detections maps to a (1-b) shift of
(1-b) * b_Y / alpha if the naive model absorbs it in the Y-M normalisation."""
import numpy as np
G = dict(np.load("/Users/boris/GitHub/sz-poisson-blending/data_cosmo/c02_grid.npz"))
j = int(np.argmin(np.abs(G["params"] - np.array([0.81, 0.31, 0.8, 0.17])).sum(1)))
by = G["ybias"][j]
print(f"stored ybias at fiducial {by:.4f}; implied d(1-b) = 0.8*ybias/1.79 = {0.8*by/1.79:+.4f}  (fits: +0.0124 case B, +0.0190 case C)")
