"""Density-matrix sanity checks and rank-based separability tests (paper Sec. II B).

Cleaned-up from ``PSD``, ``check_valid_rho``, ``check_pure_rho``, ``fs_check`` ... ``bs3_check``
in the original notebooks. Partial traces are done in NumPy (no QuTiP dependency).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def partial_trace_keep(rho: np.ndarray, keep, dims=(2, 2, 2)) -> np.ndarray:
    """Reduced density matrix on the subsystems in ``keep`` (0 = qubit A)."""
    keep = sorted(keep)
    n = len(dims)
    rho = np.asarray(rho).reshape(dims + dims)
    trace_out = [i for i in range(n) if i not in keep]
    # trace subsystems from the highest index down so axis numbers stay valid
    for count, i in enumerate(sorted(trace_out, reverse=True)):
        cur_n = n - count
        rho = np.trace(rho, axis1=i, axis2=i + cur_n)
    d = int(np.prod([dims[i] for i in keep]))
    return rho.reshape(d, d)


def is_psd(rho: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.all(np.linalg.eigvalsh(rho) >= -tol))


@dataclass(frozen=True)
class RhoReport:
    shape: tuple
    eigenvalues: np.ndarray
    psd: bool
    trace: complex
    hermitian: bool
    rank: int
    idempotent: bool
    purity: complex  # tr(rho^2)


def check_rho(rho: np.ndarray, tol_psd: float = 1e-10, tol_idem: float = 1e-10) -> RhoReport:
    """Validity and purity report for a density matrix.

    Fixes a bug in the original ``check_rho``, which returned ``tr(rho^2)`` in place of
    ``tr(rho)`` because both were bound to the same name ``tr``.
    """
    rho = np.asarray(rho, dtype=complex)
    sq = rho @ rho
    return RhoReport(
        shape=rho.shape,
        eigenvalues=np.linalg.eigvalsh(rho),
        psd=is_psd(rho, tol_psd),
        trace=np.trace(rho),
        hermitian=bool(np.allclose(rho, rho.conj().T)),
        rank=int(np.linalg.matrix_rank(rho, hermitian=True, tol=1e-8)),
        idempotent=bool(np.all(np.abs(sq - rho) <= tol_idem)),
        purity=np.trace(sq),
    )


def single_qubit_ranks(rho: np.ndarray, tol: float = 1e-8) -> tuple[int, int, int]:
    """Ranks of the three single-qubit reduced states (A, B, C)."""
    return tuple(
        int(np.linalg.matrix_rank(partial_trace_keep(rho, [i]), hermitian=True, tol=tol)) for i in range(3)
    )


def rank_class(rho: np.ndarray, tol: float = 1e-8) -> str:
    """Pure-state classification from single-qubit reduced ranks.

    Returns 'SEP', 'BS1' (A-BC), 'BS2' (B-AC), 'BS3' (C-AB) or 'GME' (W or GHZ).
    """
    ra, rb, rc = single_qubit_ranks(rho, tol)
    if (ra, rb, rc) == (1, 1, 1):
        return "SEP"
    if (ra, rb, rc) == (1, 2, 2):
        return "BS1"
    if (ra, rb, rc) == (2, 1, 2):
        return "BS2"
    if (ra, rb, rc) == (2, 2, 1):
        return "BS3"
    return "GME"
