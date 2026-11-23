"""Feature vectors (paper Sec. III B-C, Eqs. 3-5, Table III).

- :func:`flatten_rho`      the 128-dimensional vector [Re(rho_00..rho_77), Im(rho_00..rho_77)]
- :data:`CANONICAL_18`     the 18 elements that survive the manual (canonical-form) reduction:
                           14 real + 4 imaginary entries of the upper triangle on the support
                           {000, 100, 101, 110, 111}, with the last diagonal element dropped
                           (it is fixed by unit trace).
"""

from __future__ import annotations

import numpy as np

FULL_128 = [f"Re_rho_{a}{b}" for a in range(8) for b in range(8)] + [
    f"Im_rho_{a}{b}" for a in range(8) for b in range(8)
]

# Order as listed in Table III of the paper (feature numbers 1-18).
CANONICAL_18 = [
    "Re_rho_00",
    "Re_rho_04",
    "Re_rho_05",
    "Re_rho_06",
    "Re_rho_07",
    "Re_rho_44",
    "Re_rho_45",
    "Re_rho_46",
    "Re_rho_47",
    "Re_rho_55",
    "Re_rho_56",
    "Re_rho_57",
    "Re_rho_66",
    "Re_rho_67",
    "Im_rho_04",
    "Im_rho_45",
    "Im_rho_46",
    "Im_rho_47",
]


def flatten_rho(rho: np.ndarray) -> np.ndarray:
    """128-vector [Re(rho) row-major, Im(rho) row-major] (Eqs. 3-4)."""
    rho = np.asarray(rho, dtype=complex).reshape(8, 8)
    return np.concatenate([rho.real.ravel(), rho.imag.ravel()])


def unflatten_rho(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    return (vec[:64] + 1j * vec[64:128]).reshape(8, 8)


def canonical_18(rho: np.ndarray) -> np.ndarray:
    """The 18 canonical-form features of Table III, in Table III order."""
    rho = np.asarray(rho, dtype=complex).reshape(8, 8)
    out = []
    for name in CANONICAL_18:
        part, ij = name.split("_rho_")
        a, b = int(ij[0]), int(ij[1])
        out.append(rho[a, b].real if part == "Re" else rho[a, b].imag)
    return np.array(out)
