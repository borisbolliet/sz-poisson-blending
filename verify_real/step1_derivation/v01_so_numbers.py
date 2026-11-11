"""v01: SO LAT numbers re-derived from the paper text and the SO v3.0.0 code (both fetched by the verifier),
the effective noise knees, and the tSZ SED by an independent (occupation-number Kompaneets) route.
Usage: python -I v01_so_numbers.py <so19.txt> <SO_v300.py.txt>     (downloaded files are read as data only)
"""
import os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import brentq
import vlib as V

txt = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
code = open(sys.argv[2], encoding="utf-8", errors="replace").read()

print("=== (1) paper text (arXiv:1808.07445v2, pdftotext -layout), LAT columns of Table 1 and Table 2 ===")
for k, line in enumerate(txt):
    if re.match(r"^\s+(27|39|93|145|225|280)\s+\d", line):
        print(f"   L{k+1}: {line.strip()}")
for key in ["5-year survey with 20% of the total observing time", "discarding 15% of the maps", "we fix `knee = 1000",
            "αknee = −3.5", "correlation coefficient of 0.9", "mistakenly used N", "divide\nthe extrapolated"]:
    hits = [k for k, l in enumerate(txt) if key.split("\n")[0] in l]
    print(f"   '{key.splitlines()[0]}': lines {[h+1 for h in hits][:3]}")
print("   -> transcription used (vlib): FWHM", V.FWHM_AM, "white baseline", V.WHITE["baseline"], "goal", V.WHITE["goal"],
      "N_red", V.NRED_PAPER, "l_knee", V.L_KNEE, "alpha", V.ALPHA_KNEE, "r_atm", V.R_ATM)

claim = dict(fwhm=[7.4, 5.1, 2.2, 1.4, 1.0, 0.9], wb=[71, 36, 8.0, 10, 22, 54], wg=[52, 27, 5.8, 6.3, 15, 37],
             nred=[100, 39, 230, 1500, 17000, 31000])
print("   builder claim == my transcription:",
      np.allclose(claim["fwhm"], V.FWHM_AM), np.allclose(claim["wb"], V.WHITE["baseline"]),
      np.allclose(claim["wg"], V.WHITE["goal"]), np.allclose(claim["nred"], V.NRED_PAPER))

print("\n=== (2) SO v3.0.0 code constants (grep of the downloaded source) ===")
for pat in [r"C_\d+\s*=\s*[\d.]+", r"S_LA_\d+\s+=\s*np.array\(\[[^\]]+\]\)", r"t = .*", r"A_SR = .*", r"r_atm = .*",
            r"AN_T_27\s+=.*", r"alpha_temp = .*", r"ell_pivot = .*"]:
    for m in re.findall(pat, code)[:6]:
        print("   ", m.strip())
print("   code N_red_eff = C/(2 N_tubes) =", V.nred_from_code(), " paper Table 2:", V.NRED_PAPER,
      " ratio paper/code", np.round(V.NRED_PAPER / V.nred_from_code(), 4))
for mode in ("baseline", "goal"):
    wc = V.white_from_code(mode)
    print(f"   [{mode}] code white [uK-arcmin] = {np.round(wc, 3)}; paper {V.WHITE[mode]}; ratio paper/code "
          f"{np.round(V.WHITE[mode]/wc, 4)}; max |ratio^2-1| (power) = {np.max(np.abs((V.WHITE[mode]/wc)**2-1)):.4f}")
print(f"   t_eff = {V.T_EFF:.6e} s (5 yr x 365.25 d x 0.2 x 0.85); A = 4 pi 0.4 = {V.A_SR:.5f} sr")

print("\n=== (3) noise power: paper-table route vs code route, and effective knee (red = white) ===")
ell = np.array([100, 300, 1000, 2000, 3000, 5000, 8000, 20000], float)
for mode in ("baseline", "goal"):
    wp = (V.WHITE[mode] * V.AM) ** 2; wc = (V.white_from_code(mode) * V.AM) ** 2
    rp = V.NRED_PAPER * V.A_SR / V.T_EFF; rc = V.nred_from_code() * V.A_SR / V.T_EFF
    sh = (ell / V.L_KNEE) ** V.ALPHA_KNEE
    Np = wp[:, None] + rp[:, None] * sh[None, :]
    Nc = wc[:, None] + rc[:, None] * sh[None, :]
    print(f"   [{mode}] N_ii(paper)/N_ii(code) - 1 at l={ell.astype(int).tolist()}:")
    for i in range(6):
        print(f"      {V.FREQ[i]:5.0f} GHz: {np.round(Np[i]/Nc[i]-1, 4)}")
    for (i, j) in V.TUBES:
        print(f"      pair {V.FREQ[i]:.0f}x{V.FREQ[j]:.0f}: N_ij ratio paper/code = {np.sqrt(rp[i]*rp[j])/np.sqrt(rc[i]*rc[j]):.4f} (all l)")
    allr = np.concatenate([(Np / Nc - 1).ravel(), [np.sqrt(rp[i]*rp[j])/np.sqrt(rc[i]*rc[j]) - 1 for i, j in V.TUBES]])
    print(f"      max |paper/code - 1| over all N_ij(l) (l <= 20000) = {np.max(np.abs(allr)):.4f}")
    lk = V.L_KNEE * (rp / wp) ** (1 / 3.5)
    print(f"      effective knee l where red = white: {np.round(lk).astype(int).tolist()}")
claim_knee = {"baseline": [409, 461, 1809, 2721, 3469, 2466], "goal": [489, 544, 2174, 3543, 4318, 3060]}
print("   builder claim knees:", claim_knee)

print("\n=== (4) tSZ SED: closed form vs occupation-number Kompaneets route (finite differences) ===")
def g_kompaneets(nu):
    x = V.HPL * nu * 1e9 / (V.KB * V.TCMB)
    n = lambda x: 1.0 / np.expm1(x)
    h = 1e-4 * x
    dn = lambda x: (n(x + h) - n(x - h)) / (2 * h)
    # Delta n / y = x^-2 d/dx [x^4 dn/dx]  ; a temperature shift dT/T gives Delta n = -x dn/dx dT/T
    F = lambda x: x ** 4 * dn(x)
    dn_y = (F(x + h) - F(x - h)) / (2 * h) / x ** 2
    return dn_y / (-x * dn(x))
nus = np.array([27, 39, 93, 145, 150, 217, 225, 280, 353], float)
for nu in nus:
    a, b = V.gx(nu), g_kompaneets(nu)
    print(f"   {nu:5.0f} GHz: g closed = {a:+.5f}  Kompaneets(n) = {b:+.5f}  diff {a-b:+.1e};  f = {V.f_uK(nu):+.4e} uK per y")
nu0 = brentq(lambda nu: V.gx(nu), 150, 300, xtol=1e-10)
print(f"   null frequency = {nu0:.3f} GHz")
print("   builder claim: g = -1.963, -1.922, -1.572, -1.016, +0.116, +1.002 ; null 217.51 GHz")
gb = np.array([-1.963, -1.922, -1.572, -1.016, 0.116, 1.002])
print("   max |mine - claim| at band centres:", np.max(np.abs(V.gx(V.FREQ) - gb)))
