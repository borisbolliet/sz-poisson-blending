"""V3b: check int rho d^2s = RATIO^2 sigma_A^2 / N_w (Campbell mean kernel) with own Hankel numerics."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import j0
import own_model as om
print(f"g(0)/RATIO = {om.g_of(np.array([0.0]))[0]/om.RATIO:.6f}")
sb = 1.4 / np.sqrt(8 * np.log(2))
for tA, tB in [(1, 0.6), (3, 0.6), (3, 3)]:
    l = np.linspace(0, 12 / sb, 20000); B2 = np.exp(-(l * sb) ** 2)
    CAA = np.trapezoid(l * B2 * om.g_of(l * tA) ** 2, l) / (2 * np.pi)
    s = np.linspace(0, 5 * (tA + tB) + 6, 6001)
    C = np.array([np.trapezoid(l * B2 * om.g_of(l * tA) * om.g_of(l * tB) * j0(l * ss), l) for ss in s]) / (2 * np.pi)
    I = 2 * np.pi * np.trapezoid(C * s, s) / CAA
    pred = om.RATIO ** 2 * om.sigmaY(tA, 1.4, 4e-6)[0] ** 2 / (4e-6) ** 2
    print(f"thA={tA} thB={tB}: int rho d2s = {I:.5f} arcmin^2; RATIO^2 sigma^2/N_w = {pred:.5f}; ratio {I/pred:.5f}")
