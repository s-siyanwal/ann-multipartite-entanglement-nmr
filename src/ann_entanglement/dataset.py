"""Training and experimental datasets (paper Sec. III B and III D).

- :func:`generate_training_set`  balanced simulated set from :mod:`.states`
- :func:`load_experimental`      the 30 NMR states (``data/raw/experimental_states_3q.csv``)
- :func:`validation_test_splits` the '18 validation : 12 test' splits; 3 validation + 2 test
                                 states per class, C(5,3)^6 = 10^6 possible, 100 used in the paper
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CANONICAL_18, FULL_128, canonical_18, flatten_rho
from .states import CLASS_NAMES, GME_CLASSES, random_canonical_state

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def generate_training_set(n_per_class: int, seed: int | None = 0, full: bool = False) -> pd.DataFrame:
    """Balanced simulated dataset: ``n_per_class`` random canonical-form states per SLOCC class.

    Columns: the 18 canonical features (or all 128 if ``full``), plus ``class``,
    ``class_code`` (0-5, order of :data:`CLASS_NAMES`) and ``gme`` (1 for W/GHZ).
    """
    rng = np.random.default_rng(seed)
    cols = FULL_128 if full else CANONICAL_18
    rows, labels = [], []
    for code, cls in enumerate(CLASS_NAMES):
        for _ in range(n_per_class):
            rho = random_canonical_state(cls, rng)
            rows.append(flatten_rho(rho) if full else canonical_18(rho))
            labels.append((cls, code))
    df = pd.DataFrame(np.array(rows), columns=cols)
    df["class"] = [c for c, _ in labels]
    df["class_code"] = [k for _, k in labels]
    df["gme"] = df["class"].isin(GME_CLASSES).astype(int)
    return df


def load_experimental(path: str | Path | None = None, paper_only: bool = True) -> pd.DataFrame:
    """Experimentally reconstructed three-qubit NMR density matrices.

    ``paper_only=True`` keeps the 30 states used in the paper (5 per class). The CSV also
    holds 4 lower-fidelity repeats that the original analysis excluded (``used_in_paper`` = False).
    """
    df = pd.read_csv(path or DATA_DIR / "experimental_states_3q.csv")
    if paper_only:
        df = df[df["used_in_paper"]].reset_index(drop=True)
    df = df.copy()
    df["gme"] = df["class"].isin(GME_CLASSES).astype(int)
    return df


def validation_test_splits(exp: pd.DataFrame, n_splits: int = 100, n_val_per_class: int = 3, seed: int = 0):
    """Yield ``(val_idx, test_idx)`` pairs: per class, ``n_val_per_class`` states to validation, rest to test.

    Draws ``n_splits`` distinct combinations from the C(5,3)^6 possibilities.
    """
    rng = np.random.default_rng(seed)
    per_class = [np.flatnonzero(exp["class"].to_numpy() == c) for c in CLASS_NAMES]
    choices = [list(combinations(ix, n_val_per_class)) for ix in per_class]
    total = int(np.prod([len(c) for c in choices]))
    n_splits = min(n_splits, total)
    seen = set()
    while len(seen) < n_splits:
        pick = tuple(int(rng.integers(len(c))) for c in choices)
        if pick in seen:
            continue
        seen.add(pick)
        val = np.sort(np.concatenate([np.array(choices[k][p]) for k, p in enumerate(pick)]))
        test = np.setdiff1d(np.arange(len(exp)), val)
        yield val, test


def row_to_rho(row) -> np.ndarray:
    """Rebuild the 8x8 density matrix from a row with ``Re_rho_ab`` / ``Im_rho_ab`` columns."""
    re = np.array([[row[f"Re_rho_{a}{b}"] for b in range(8)] for a in range(8)], dtype=float)
    im = np.array([[row.get(f"Im_rho_{a}{b}", 0.0) for b in range(8)] for a in range(8)], dtype=float)
    return re + 1j * im


def experimental_features(exp: pd.DataFrame) -> pd.DataFrame:
    """18 canonical features for each experimental state (same columns as the training set)."""
    return pd.DataFrame([canonical_18(row_to_rho(r)) for _, r in exp.iterrows()], columns=CANONICAL_18)
