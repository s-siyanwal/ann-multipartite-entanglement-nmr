"""ANN-enhanced detection of multipartite entanglement in a three-qubit NMR processor.

Reusable code behind Gulati*, Siyanwal*, Arvind & Dorai,
Quantum Inf. Process. 24, 74 (2025), doi:10.1007/s11128-025-04696-8.
(* equal contribution)

Modules map to the paper:

- :mod:`.states`     Sec. II A, Table I   canonical-form state generators
- :mod:`.measures`   Sec. II B, Table II  correlation-tensor ranks, 3-tangle
- :mod:`.checks`     Sec. II B            density-matrix validity / rank checks
- :mod:`.features`   Sec. III B-C, Eq.3-5 feature vectors, 128 -> 18 reduction
- :mod:`.dataset`    Sec. III B, III D    training set, experimental set, splits
- :mod:`.selection`  Sec. IV A, Table III ANOVA F-value feature ranking
- :mod:`.models`     Sec. III A, IV, V    ANN, SVM and KNN classifiers
- :mod:`.experiments` Sec. IV B-C, V A    Tables IV-VII protocol (re-implementation)
"""

from . import checks, dataset, features, measures, selection, states

__all__ = ["checks", "dataset", "features", "measures", "selection", "states"]
__version__ = "1.0.0"
