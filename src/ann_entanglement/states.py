"""Random three-qubit pure states in the five-term canonical form (paper Sec. II A, Eq. 1, Table I).

    |psi> = l0|000> + l1 e^{i phi}|100> + l2|101> + l3|110> + l4|111>

Basis index (qubit A is the most significant bit): 000->0, 100->4, 101->5, 110->6, 111->7.

Each generator samples the non-zero coefficients from U(0, 1] (zero excluded, as in the
paper) and phi from U(0, 2*pi], normalises, and returns the 8x8 density matrix.
These are cleaned-up versions of ``FS_ASD`` ... ``GHZ_ASD`` from the original
Colab notebook ``ENT-SIM-DATA-ANALYSIS.ipynb`` (June 2023).
"""

from __future__ import annotations

import numpy as np

# Class codes used throughout the code base and in the paper's tables.
CLASS_NAMES = ("SEP", "BS1", "BS2", "BS3", "W", "GHZ")
# Which qubit is separable from the other two, in the paper's convention.
CLASS_PARTITION = {
    "SEP": "A-B-C",
    "BS1": "A-BC",
    "BS2": "B-AC",
    "BS3": "C-AB",
    "W": "ABC (W)",
    "GHZ": "ABC (GHZ)",
}
GME_CLASSES = frozenset({"W", "GHZ"})

# Non-zero canonical-form amplitudes for each class (Table I).
_SUPPORT = {
    "BS1": (4, 5, 6, 7),  # l0 = 0, l4 != 0, l2 l3 != l1 l4 e^{i phi}  -> A-BC
    "BS2": (0, 4, 5),  # l4 = 0, l2 != 0, l3 = 0                    -> B-AC
    "BS3": (0, 4, 6),  # l4 = 0, l2 = 0, l3 != 0                    -> C-AB
    "W": (0, 4, 5, 6),  # l4 = 0, l2 l3 != 0                         -> W
    "GHZ": (0, 4, 5, 6, 7),  # l0 l4 != 0                                 -> GHZ
}
# Three A-B-C sub-families of Table I, chosen with equal probability.
_SEP_SUPPORTS = ((0, 4), (4, 6), (4, 5))


def _unit_interval(rng: np.random.Generator, size=None):
    """Sample from U(0, 1], i.e. zero excluded (paper Sec. II A)."""
    return 1.0 - rng.random(size)


def _state_from_support(support, rng: np.random.Generator) -> np.ndarray:
    psi = np.zeros(8, dtype=complex)
    for idx in support:
        psi[idx] = _unit_interval(rng)
    phi = 2 * np.pi * _unit_interval(rng)
    psi[4] *= np.exp(1j * phi)  # the |100> amplitude carries the phase
    psi /= np.linalg.norm(psi)
    return psi


def random_canonical_ket(cls: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random normalised canonical-form ket (length-8 complex vector) of SLOCC class ``cls``."""
    rng = np.random.default_rng() if rng is None else rng
    if cls == "SEP":
        support = _SEP_SUPPORTS[rng.integers(len(_SEP_SUPPORTS))]
    elif cls in _SUPPORT:
        support = _SUPPORT[cls]
    else:
        raise ValueError(f"unknown class {cls!r}; choose from {CLASS_NAMES}")
    return _state_from_support(support, rng)


def random_canonical_state(cls: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random 8x8 density matrix |psi><psi| of SLOCC class ``cls`` (Table I)."""
    psi = random_canonical_ket(cls, rng)
    return np.outer(psi, psi.conj())


def ket_to_dm(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).ravel()
    return np.outer(psi, psi.conj())
