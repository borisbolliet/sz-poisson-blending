"""NOT USED.  Written to run SZiFi's own code; the first attempt to import SZiFi was refused by this session's
permission classifier ("Code from External"), so SZiFi was never imported or run and this module was never executed.
All checks use our own emulation of SZiFi's conventions instead (vsz_lib.py).  Kept to document the attempted route.

Minimal stand-ins for the three packages SZiFi imports at import time but that are not installed here
(pymaster = NaMaster, pixell, scikit-learn).  They exist only so that `import szifi` succeeds.

Every stubbed callable raises NotImplementedError, except one: pymaster.NmtBinFlat, which SZiFi's spec.power_spectrum
constructor always builds.  Our stub only stores the bin edges and returns the bin mid-points from
get_effective_ells().  In the code path used here (spec.power_spectrum.get_power_spectrum(implementation="custom"),
SZiFi's own non-NaMaster periodogram) that value is used only for the length of an array; the multipoles that SZiFi
actually interpolates are recomputed by SZiFi's own get_binned_spec (also the bin mid-points).

Usage (before importing szifi):  import vsz_stubs; vsz_stubs.install(); import szifi
SZiFi itself is never modified: the clone is only added to the end of sys.path.
"""
import sys
import types
import numpy as np

SZIFI_ROOT = "/Users/boris/GitHub/szifi"


def _missing(name):
    def f(*a, **k):
        raise NotImplementedError(f"stub '{name}': needs a package that is not installed (pymaster/pixell/sklearn)")
    return f


class _MissingClass:
    _name = "?"

    def __init__(self, *a, **k):
        raise NotImplementedError(f"stub class '{self._name}': needs a package that is not installed")


def _missing_class(name):
    return type(name.split(".")[-1], (_MissingClass,), {"_name": name})


class NmtBinFlat:
    """Stand-in for pymaster.NmtBinFlat: stores the bin edges; effective ells = bin mid-points."""

    def __init__(self, l0, lf):
        self.l0 = np.asarray(l0, float)
        self.lf = np.asarray(lf, float)

    def get_effective_ells(self):
        return 0.5 * (self.l0 + self.lf)


def install():
    if "szifi" in sys.modules:
        return
    sys.dont_write_bytecode = True                      # never write .pyc files into the read-only clone
    # ---- pymaster
    pm = types.ModuleType("pymaster")
    pm.NmtBinFlat = NmtBinFlat
    for n in ("NmtFieldFlat", "NmtWorkspaceFlat"):
        setattr(pm, n, _missing_class("pymaster." + n))
    for n in ("compute_coupled_cell_flat", "mask_apodization_flat", "compute_full_master_flat"):
        setattr(pm, n, _missing("pymaster." + n))
    sys.modules["pymaster"] = pm
    # ---- pixell (package with enmap, utils, wcsutils, reproject, curvedsky, enplot)
    px = types.ModuleType("pixell")
    px.__path__ = []
    for sub in ("enmap", "utils", "wcsutils", "reproject", "curvedsky", "enplot"):
        m = types.ModuleType("pixell." + sub)
        setattr(px, sub, m)
        sys.modules["pixell." + sub] = m
    enm = sys.modules["pixell.enmap"]

    class ndmap(np.ndarray):                             # only used in isinstance() checks
        pass
    enm.ndmap = ndmap
    for n in ("enmap", "empty", "zeros", "sky2pix", "pix2sky", "smooth_gauss", "posmap", "read_map", "write_map"):
        setattr(enm, n, _missing("pixell.enmap." + n))
    ut = sys.modules["pixell.utils"]
    ut.arcmin = np.pi / 180.0 / 60.0
    ut.degree = np.pi / 180.0
    ut.angdist = _missing("pixell.utils.angdist")
    sys.modules["pixell.wcsutils"].explicit = _missing("pixell.wcsutils.explicit")
    sys.modules["pixell"] = px
    # ---- scikit-learn
    sk = types.ModuleType("sklearn")
    sk.__path__ = []
    skc = types.ModuleType("sklearn.cluster")
    skc.AgglomerativeClustering = _missing_class("sklearn.cluster.AgglomerativeClustering")
    skc.DBSCAN = _missing_class("sklearn.cluster.DBSCAN")
    skm = types.ModuleType("sklearn.metrics")
    skm.pairwise_distances = _missing("sklearn.metrics.pairwise_distances")
    sk.cluster, sk.metrics = skc, skm
    sys.modules.update({"sklearn": sk, "sklearn.cluster": skc, "sklearn.metrics": skm})
    # ---- the clone, at the END of sys.path (never shadows anything)
    if SZIFI_ROOT not in sys.path:
        sys.path.append(SZIFI_ROOT)
