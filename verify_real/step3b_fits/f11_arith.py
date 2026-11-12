"""f11 (step-3b verification, lens 'fits'): arithmetic checks of quoted numbers (A6 floor, B9 differences)."""
import numpy as np
from scipy.special import ndtr
print("A6: analytic kernel of a halo with q' -> 0 at e = 5: Phi(sqrt(k) - 5) for k = 0, 2, 3: "
      + ", ".join(f"{ndtr(np.sqrt(k) - 5.0):.2e}" for k in (0.0, 2.0, 3.0)))
print("B9: step 3b config d (SO40) minus the preview's upper end: baseline 0.36 - 0.22 = %+.2f; goal 0.66 - 0.40 = %+.2f" % (0.36 - 0.22, 0.66 - 0.40))
