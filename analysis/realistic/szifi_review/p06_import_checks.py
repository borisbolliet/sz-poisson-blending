"""p06: what importing SZiFi does (review items D3, P3, P4, P5).
Imports SZiFi (unchanged; -I -B, no bytecode written) and reports which optional packages the import loads, whether it
switches warnings off for the whole process, what szifi.utils is, and which warnings importing szifi.car (the CAR /
pixell interface) emits once all warnings are switched back on.

Usage (from the repository root):
  /Users/boris/GitHub/szifi-venv/bin/python -I -B analysis/realistic/szifi_review/p06_import_checks.py
"""
import sys, warnings, importlib
sys.dont_write_bytecode = True
sys.path.insert(0, "/Users/boris/GitHub/szifi")
import numpy as _np
with warnings.catch_warnings(record=True) as w_ctrl:
    _np.sqrt(_np.array([-1.0]))
print(f"before import szifi, numpy's 'invalid value' warning for sqrt(-1) reaches the user: {len(w_ctrl) > 0}")
before = set(sys.modules)
n_filters_before = len(warnings.filters)
import szifi                                          # noqa: E402
new = warnings.filters[:len(warnings.filters) - n_filters_before]
print(f"import szifi adds {len(new)} warnings filter(s): " + "; ".join(f"action '{f[0]}', category {f[2].__name__}, "
      f"message {'any' if f[1] is None else repr(f[1].pattern)}, module {'any' if f[3] is None else repr(f[3].pattern)}" for f in new))
with warnings.catch_warnings(record=True) as w0:
    _np.sqrt(_np.array([-1.0]))
print(f"after import szifi, numpy's 'invalid value' warning for sqrt(-1) reaches the user: {len(w0) > 0}")
loaded = set(sys.modules) - before
for pkg in ("pixell", "pymaster", "sklearn", "yaml", "numba", "healpy", "astropy", "scipy"):
    print(f"import szifi loads {pkg:9s}: {pkg in loaded}")
print(f"szifi.utils is the module '{szifi.utils.__name__}' (SZiFi's own is 'szifi.utils')")
print(f"szifi.utils is the utils that szifi/maps.py imports from pixell: {szifi.utils is sys.modules['szifi.maps'].utils}")
print(f"szifi.car imported by 'import szifi': {'szifi.car' in sys.modules}")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    importlib.import_module("szifi.car")
msgs = sorted({f"{x.category.__name__}: {str(x.message)[:300]}" for x in w if "pkg_resources" in str(x.message) or "pkg_resources" in x.filename})
print(f"import szifi.car (setuptools {importlib.import_module('setuptools').__version__}): {len(msgs)} warning(s) about pkg_resources")
for m in msgs:
    print("  " + m)
