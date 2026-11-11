"""Step 1 / s06c: blending fractions when only SUB-THRESHOLD neighbours contribute (own mean q_B < 5 in the same
instrument, as in s05's 'subthr').

This is the part of the blending that masking / deblending the detected clusters cannot remove (the setting of
Raghunathan 2022's 'remove the detected clusters and re-run' test).  Same code as s06b (imported), with:
   neighbour population filtered to Y_B / sigma_Y(theta_B) < 5, one floor (all masses >= 1e13),
   missing-variance term from s05's sub-threshold Parseval variance.
Output: data_real/pd_fractions_{run}_subthr.npz and s06c_pd_subthreshold_{run}.log.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s06b_pd_fractions as S          # noqa: E402


def only_subthreshold(pop, model):
    thB, YB, cB, wB = pop
    keep = YB / model.sigma_Y(thB) < 5.0
    return thB[keep], YB[keep], cB[keep], wB[keep]


if __name__ == "__main__":
    S.TAG = "_subthr"
    S.FLOORS = {"1e13": (0, 1, 2)}
    S.VAR_KEY = {"1e13": "subthr"}
    S.POP_FILTER = only_subthreshold
    S.main(sys.argv[1])
