"""c00: environment of the config verifier (lens 'config') and the state of the SZiFi clone before any run.

Prints: Python and package versions of the SZiFi venv, the clone's HEAD and 'git status --short', any __pycache__ or
.pyc inside the clone, and where 'import szifi' resolves.  Read-only.
Usage: /Users/boris/GitHub/szifi-venv/bin/python -I -B c00_env.py
"""
import os, sys, subprocess
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SZIFI = "/Users/boris/GitHub/szifi"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(HERE, "out", "numba_cache")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def pycache_in_clone():
    hits = []
    for root, dirs, files in os.walk(SZIFI):
        if ".git" in root.split(os.sep):
            continue
        hits += [os.path.join(root, d) for d in dirs if d == "__pycache__"]
        hits += [os.path.join(root, f) for f in files if f.endswith(".pyc")]
    return hits


print(f"python {sys.version.split()[0]} ({sys.executable})")
print("SZiFi clone HEAD:", sh(["git", "-C", SZIFI, "log", "-1", "--format=%h %ad %s"]).strip())
st = sh(["git", "-C", SZIFI, "status", "--short"])
print(f"'git status --short' output: {len(st)} characters{(': ' + st) if st else ''}")
print(f"__pycache__ / .pyc inside the clone (outside .git): {pycache_in_clone()}")
sys.path.insert(0, SZIFI)
import numpy, scipy, astropy, healpy, numba, pixell, pymaster, sklearn  # noqa: E402,F401
for m in (numpy, scipy, astropy, healpy, numba, pixell, sklearn):
    print(f"  {m.__name__:10s} {m.__version__}")
try:
    from importlib.metadata import version
    print(f"  pymaster   {version('pymaster')}")
    print(f"  orphics    {version('orphics')}")
except Exception as e:  # pragma: no cover
    print("  version lookup failed:", e)
import szifi  # noqa: E402
print("import szifi ->", szifi.__file__)
print(f"sys.dont_write_bytecode = {sys.dont_write_bytecode}; NUMBA_CACHE_DIR = {os.environ['NUMBA_CACHE_DIR']}")
print(f"__pycache__ / .pyc inside the clone after the import: {pycache_in_clone()}")
