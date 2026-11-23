"""Entanglement measures used as non-ML baselines (paper Sec. II B, Sec. V B, Tables II and VIII).

- correlation-tensor ranks R(T123), R(T231), R(T312), following Ref. [49] of the paper
- 3-tangle in the canonical-form shortcut, tau = 4 <000|rho|000><111|rho|111>
- Coffman-Kundu-Wootters 3-tangle for general pure states (check value)
- negativity and von Neumann entropies (used in the original exploratory notebooks)
"""

from __future__ import annotations

import numpy as np

from .checks import partial_trace_keep

PAULI = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)
SIGMA_Y = PAULI[1]


def correlation_tensor(rho: np.ndarray) -> np.ndarray:
    """Three-body correlation tensor t_ijk = tr(rho sigma_i (x) sigma_j (x) sigma_k), shape (3, 3, 3)."""
    t = np.empty((3, 3, 3))
    for i, si in enumerate(PAULI):
        for j, sj in enumerate(PAULI):
            for k, sk in enumerate(PAULI):
                t[i, j, k] = np.real(np.trace(rho @ np.kron(np.kron(si, sj), sk)))
    return t


def correlation_tensor_ranks(rho: np.ndarray, tol: float = 1e-2) -> tuple[int, int, int]:
    """(R(T123), R(T231), R(T312)): ranks of the 3x9 unfoldings of t_ijk with qubit A, B, C as rows.

    ``tol`` is an absolute singular-value cut-off; noisy experimental matrices need a
    finite tolerance (the default suits states reconstructed to ~1e-2 accuracy).
    """
    t = correlation_tensor(rho)
    unfoldings = (
        t.reshape(3, 9),
        np.transpose(t, (1, 2, 0)).reshape(3, 9),
        np.transpose(t, (2, 0, 1)).reshape(3, 9),
    )
    return tuple(int(np.sum(np.linalg.svd(m, compute_uv=False) > tol)) for m in unfoldings)


def ranks_to_class(ranks: tuple[int, int, int]) -> str:
    """Table II of the paper: correlation-tensor ranks -> entanglement class."""
    table = {
        (3, 3, 3): "GME",
        (2, 2, 2): "GME",
        (1, 3, 3): "BS1",
        (3, 1, 3): "BS2",
        (3, 3, 1): "BS3",
        (1, 1, 1): "SEP",
    }
    return table.get(tuple(ranks), "UNDETERMINED")


def three_tangle_canonical(rho: np.ndarray) -> float:
    """tau_123 = 4 |l0|^2 |l4|^2 = 4 <000|rho|000> <111|rho|111> for canonical-form states (paper Sec. II B)."""
    rho = np.asarray(rho)
    return float(np.real(4 * rho[0, 0] * rho[7, 7]))


def three_tangle_ckw(rho: np.ndarray) -> float:
    """Coffman-Kundu-Wootters 3-tangle tau = tau_A(BC) - tau_AB - tau_AC for a pure 3-qubit state."""
    rho = np.asarray(rho, dtype=complex)
    rho_a = partial_trace_keep(rho, [0])
    tau_a_bc = 4 * np.real(np.linalg.det(rho_a))
    yy = np.kron(SIGMA_Y, SIGMA_Y)

    def concurrence_sq(r2):
        r_tilde = yy @ r2.conj() @ yy
        ev = np.sqrt(np.clip(np.sort(np.real(np.linalg.eigvals(r2 @ r_tilde)))[::-1], 0, None))
        return max(ev[0] - ev[1] - ev[2] - ev[3], 0.0) ** 2

    tau_ab = concurrence_sq(partial_trace_keep(rho, [0, 1]))
    tau_ac = concurrence_sq(partial_trace_keep(rho, [0, 2]))
    return float(tau_a_bc - tau_ab - tau_ac)


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    ev = np.clip(np.real(np.linalg.eigvalsh(rho)), 0, None)
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log(ev)) / np.log(base))


def single_qubit_entropies(rho: np.ndarray) -> tuple[float, float, float]:
    return tuple(von_neumann_entropy(partial_trace_keep(rho, [i])) for i in range(3))


def negativity(rho: np.ndarray, qubit: int = 0) -> float:
    """Negativity of the bipartition (qubit | rest) via partial transpose on ``qubit``."""
    r = np.asarray(rho, dtype=complex).reshape(2, 2, 2, 2, 2, 2)
    axes = list(range(6))
    axes[qubit], axes[qubit + 3] = axes[qubit + 3], axes[qubit]
    pt = np.transpose(r, axes).reshape(8, 8)
    ev = np.real(np.linalg.eigvalsh(pt))
    return float(np.sum(np.abs(ev[ev < 0])))
