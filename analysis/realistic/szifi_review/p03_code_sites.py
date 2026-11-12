"""p03: check the SZiFi source locations that ../../../SZIFI_REVIEW.md cites, in the clone at /Users/boris/GitHub/szifi
(read only; SZiFi is not imported or run here).

For each citation the script checks that the cited line holds the expected identifier and prints OK or FAIL with
file:line and the identifier, never the line itself (SZiFi has no licence, so none of its code is reproduced here).
This is a token check: it confirms the location, not what the code does there.  Labels are the review's item numbers.
It also checks the commit, that the clone is unchanged, a few absences, and counts the Fourier modes and SZiFi's
spectral bins that lie wholly outside the l range, on our tiles and on the tiles of SZiFi's Planck survey file.

Usage (from the repository root): .venv/bin/python -I -B analysis/realistic/szifi_review/p03_code_sites.py
"""
import os, subprocess
import numpy as np

SZ = "/Users/boris/GitHub/szifi"
COMMIT = "73626fb084adc4e664f7322aaff573e6f0df21ea"

# (review item, file, line, identifier expected on that line)
SITES = [
    ("E4", "szifi/mmf.py", 179, "[lmin,lmax]"),
    ("E4", "szifi/mmf.py", 181, "indices_filter"),
    ("E4", "szifi/mmf.py", 185, "filter_tmap"),
    ("E4", "szifi/mmf.py", 186, "filter_tmap"),
    ("E4", "szifi/spec.py", 24, "l0_bins"),
    ("E4", "szifi/spec.py", 25, "lf_bins"),
    ("E4", "szifi/spec.py", 398, "extrapolate"),
    ("E4", "szifi/spec.py", 317, "np.linalg.inv"),
    ("E4", "szifi/mmf.py", 664, "filter_tmap"),
    ("E4", "szifi/mmf.py", 1155, "fftconvolve"),
    ("E4", "szifi/params.py", 24, "[100,2500]"),
    ("E4", "szifi/params.py", 13, "decouple_type"),
    ("E4", "test_files/run_szifi_so_car.py", 30, "300000"),
    ("E4", "surveys/data_planck.py", 40, "1024"),
    ("E4", "surveys/data_planck.py", 41, "14.8"),
    ("E4", "surveys/data_planck.py", 75, "get_apodised_mask"),
    ("E4", "surveys/data_planck.py", 76, "get_apodised_mask"),
    ("E5", "szifi/spec.py", 333, "n_freq"),
    ("E5", "szifi/spec.py", 363, "/ps_isotropic"),
    ("E5", "szifi/spec.py", 373, "gaussian_filter"),
    ("E5", "szifi/spec.py", 382, "*ps_isotropic"),
    ("E5", "szifi/mmf.py", 1162, "norm"),
    ("E5", "szifi/mmf.py", 1168, "np.sqrt(norm)"),
    ("E6", "szifi/mmf.py", 465, "identify_clusters"),
    ("E6", "szifi/mmf.py", 467, "np.all"),
    ("E6", "szifi/mmf.py", 442, "max_it"),
    ("E6", "szifi/cat.py", 301, "deepcopy(catalogue_1)"),
    ("E6", "szifi/cat.py", 373, "indices_catalogue_2==0."),
    ("E6", "szifi/cat.py", 384, "np.array([-1])"),
    ("E6", "szifi/cat.py", 392, "catalogue_1_new.append"),
    ("E3", "szifi/cat.py", 140, "get_angdist"),
    ("E3", "szifi/sphere.py", 94, "hp.rotator.angdist"),
    ("E3", "szifi/cat.py", 176, "AgglomerativeClustering"),
    ("E3", "szifi/cat.py", 178, "affinity="),
    ("E3", "szifi/cat.py", 291, "hp.rotator.angdist"),
    ("E3", "szifi/sphere.py", 86, "hp.rotator.angdist"),
    ("E3", "szifi/cat.py", 323, "get_angdist"),
    ("E3", "szifi/cat.py", 327, "get_angdist"),
    ("E2", "szifi/params.py", 22, "snr_weigthing"),
    ("E2", "szifi/mmf.py", 95, "snr_weighting"),
    ("E2", "szifi/mmf.py", 687, "snr_weighting"),
    ("E2", "test_files/run_szifi_planck.py", 14, "params_szifi_default"),
    ("E2", "test_files/run_szifi_planck_with_fixed.py", 10, "params_szifi_default"),
    ("E2", "test_files/run_szifi_planck_with_fixed_rsz.py", 10, "params_szifi_default"),
    ("E2", "test_files/run_szifi_planck_constrained.py", 11, "params_szifi_default"),
    ("E7", "szifi/mmf.py", 628, "theta_cart"),
    ("E7", "szifi/mmf.py", 629, "0.5*self.pix.nx"),
    ("E7", "szifi/maps.py", 58, "+0.5"),
    ("E7", "szifi/maps.py", 65, "+0.5"),
    ("E7", "szifi/mmf.py", 1155, 'mode="same"'),
    ("E7", "szifi/maps.py", 72, "+0.5"),
    ("E7", "szifi/maps.py", 79, "+0.5"),
    ("E1", "szifi/params.py", 47, "maxima_lomem"),
    ("E1", "szifi/mmf.py", 578, "n_theta > 3"),
    ("E1", "szifi/mmf.py", 711, "continue"),
    ("E1", "szifi/mmf.py", 748, "sigma_vec[j]"),
    ("E1", "szifi/params.py", 6, "theta_500_vec_arcmin"),
    ("E1", "test_files/run_szifi_so_car.py", 14, "[2.,3.]"),
    ("D1", "szifi/params.py", 24, "None"),
    ("D1", "szifi/mmf.py", 179, "[lmin,lmax]"),
    ("D2", "szifi/params.py", 20, "anisotropic_gaussian"),
    ("D2", "szifi/spec.py", 292, '"isotropic"'),
    ("D2", "szifi/spec.py", 303, "anisotropic_gaussian"),
    ("D2", "szifi/spec.py", 307, "mixed_boxcar"),
    ("D2", "szifi/spec.py", 312, "return cov_tensor"),
    ("D3", "szifi/mmf.py", 5, "filterwarnings"),
    ("D4", "szifi/params.py", 49, "/Users/"),
    ("D4", "szifi/params.py", 50, "/Users/"),
    ("D4", "szifi/params.py", 51, "/Users/"),
    ("D4", "szifi/params.py", 53, "/Users/"),
    ("D5", "szifi/params.py", 52, "save_and_load_template"),
    ("D5", "szifi/mmf.py", 611, "RuntimeError"),
    ("D5", "szifi/mmf.py", 490, "os.remove"),
    ("D5", "szifi/mmf.py", 491, "os.remove"),
    ("D5", "szifi/mmf.py", 613, "save_and_load_template"),
    ("D6", "szifi/mmf.py", 238, "mask_cluster"),
    ("D6", "szifi/mmf.py", 242, "inpaint"),
    ("D6", "szifi/mmf.py", 253, "circular_cosine_blend_adaptive"),
    ("D6", "szifi/params.py", 31, "inpaint"),
    ("A2", "test_files/run_szifi_planck.py", 58, "radius_arcmin = 10."),
    ("A2", "test_files/run_szifi_planck.py", 60, "fof"),
    ("A2", "test_files/run_szifi_planck.py", 61, "fof"),
    ("S4", "szifi/params.py", 28, "nearest"),
    ("S4", "szifi/spec.py", 220, "ell_eff[1:]"),
    ("P4", "szifi/car.py", 7, "pkg_resources"),
    ("P5", "szifi/__init__.py", 3, "from .maps import *"),
]

head = subprocess.run(["git", "-C", SZ, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", SZ, "status", "--short"], capture_output=True, text=True).stdout.strip()
pyc = [r for r, d, f in os.walk(SZ) if "__pycache__" in d and ".git" not in r]
print(f"SZiFi clone at commit {head} (expected {COMMIT}): {'OK' if head == COMMIT else 'FAIL'}; "
      f"unchanged (git status empty): {'OK' if not dirty else 'FAIL'}; no __pycache__: {'OK' if not pyc else 'FAIL'}\n")


def lines_of(rel):
    with open(os.path.join(SZ, rel), encoding="utf-8") as fh:
        return fh.read().splitlines()


n_ok = 0
for item, rel, line, tok in SITES:
    L = lines_of(rel)
    ok = line <= len(L) and tok in L[line - 1]
    n_ok += ok
    print(f"  {item:3s} {rel}:{line:<5d} holds '{tok}': {'OK' if ok else 'FAIL'}")
n_lines = len({(rel, line) for _, rel, line, _ in SITES})
print(f"\n{n_ok} of {len(SITES)} checks pass ({n_lines} distinct lines)")


def has(rel, tok):
    return tok in "\n".join(lines_of(rel))


print("\nAbsences:")
print(f"  D2 szifi/params.py:20 lists the covariance types without 'mixed': {'OK' if 'mixed' not in lines_of('szifi/params.py')[19] else 'FAIL'}")
guards = sum(has(f"szifi/{f}", 'lrange"] is None') or has(f"szifi/{f}", 'lrange"] == None')
             for f in os.listdir(os.path.join(SZ, "szifi")) if f.endswith(".py"))
print(f"  D1 no file of the package tests the parameter lrange for None: {'OK' if guards == 0 else 'FAIL'} ({guards} files do)")
print(f"  E4 SZiFi's Planck script sets no l range (uses the default): {'OK' if not has('test_files/run_szifi_planck.py', 'lrange') else 'FAIL'}")
planck = [f for f in os.listdir(os.path.join(SZ, "test_files")) if f.startswith("run_szifi_planck") and f.endswith(".py")]
print(f"  E2 none of SZiFi's {len(planck)} Planck example scripts sets 'snr_weighting': "
      f"{'OK' if not any(has('test_files/' + f, 'snr_weighting') for f in planck) else 'FAIL'}")
print(f"  P2 setup.py declares no dependencies (no install_requires): {'OK' if not has('setup.py', 'install_requires') else 'FAIL'}")
lic = [f for f in os.listdir(SZ) if f.lower().startswith(("licen", "copying"))]
print(f"  P1 no licence file in the clone's top folder: {'OK' if not lic else 'FAIL'} {lic}")


def outside(n, side_deg, lo, hi, bin_fac=4):
    """Share of the Fourier modes outside [lo, hi], and SZiFi's spectral bins (width bin_fac*pi/L, up to the axis
    Nyquist l, the first bin dropped) that hold no mode inside [lo, hi]."""
    L = np.deg2rad(side_deg)
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    ell = (2 * np.pi / L * np.hypot(kx, ky)).ravel()
    inb = (ell >= lo) & (ell <= hi)
    w = bin_fac * np.pi / L
    idx = np.floor(ell / w).astype(int)
    kept = range(1, int(n / bin_fac))
    empty = [b for b in kept if not np.any(inb[idx == b])]
    return (~inb).mean(), len(kept), len(empty), ell.max(), np.pi / L * n


print("\nE4: Fourier modes and spectral bins outside SZiFi's l range")
for label, args in (("our tiles (1200 px of 0.5'), l range [100, 20000]", (1200, 10.0, 100, 20000)),
                    ("our tiles, l range [100, 30600]", (1200, 10.0, 100, 30600)),
                    ("our tiles, l range [1, 300000] (SO example)", (1200, 10.0, 1, 300000)),
                    ("SZiFi's Planck tiles (1024 px over 14.8 deg), default l range [100, 2500]", (1024, 14.8, 100, 2500))):
    f, nk, ne, lmax, lnyq = outside(*args)
    print(f"  {label}: {f * 100:.1f}% of the modes outside; {ne} of the {nk} kept bins hold no mode inside; "
          f"highest l {lmax:.0f}, axis Nyquist l {lnyq:.0f}")
