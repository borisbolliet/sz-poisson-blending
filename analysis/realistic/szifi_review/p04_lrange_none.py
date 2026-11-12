"""p04: SZiFi's default dictionary says that lrange = None uses all the modes (szifi/params.py, line 24).  Run SZiFi
itself (unchanged, through the build's driver) with lrange = None and otherwise configuration S1, on the tile of
p00_map.py, and report where it stops.  Prints the exception and the SZiFi file:line where it is raised, not SZiFi's code.

Usage (from the repository root):
  MallocLargeCache=0 /Users/boris/GitHub/szifi-venv/bin/python -I -B analysis/realistic/szifi_review/p04_lrange_none.py
"""
import os, sys, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "data_real", "szifi_review", "out")
sys.path.insert(0, os.path.join(ROOT, "analysis", "realistic", "step3_szifi_real"))
import z_szlib as Z                                   # noqa: E402

MAP = os.path.join(OUT, "maps", "baseline_real_000.npy")
Z.STDD = os.path.join(OUT, "stdout")
Z.RUND = os.path.join(OUT, "runs")
Z.CONFIGS["S1_lrange_None"] = dict(Z.CONFIGS["S1"], lrange=None)
try:
    Z.run_tile("S1_lrange_None", "baseline", "real", 0, 8.0, map_file=MAP, save=False, tag="p04")
    print("lrange = None: the run finished")
except Exception as e:
    fr = [f for f in traceback.extract_tb(e.__traceback__) if "/szifi/szifi/" in f.filename]
    where = f"{fr[-1].filename.split('/szifi/', 1)[1]}:{fr[-1].lineno} ({fr[-1].name})" if fr else "outside SZiFi"
    print(f"lrange = None: {type(e).__name__}: {e}; raised in SZiFi at {where}")
