"""Physics checks: generated states land in the class Table I says they should."""

import numpy as np
import pytest

from ann_entanglement import checks, features, measures, states


@pytest.mark.parametrize("cls", states.CLASS_NAMES)
def test_generated_states_are_valid_pure_states(cls):
    rng = np.random.default_rng(1)
    for _ in range(20):
        rho = states.random_canonical_state(cls, rng)
        rep = checks.check_rho(rho)
        assert rep.psd and rep.hermitian and rep.idempotent
        assert np.isclose(rep.trace, 1) and np.isclose(rep.purity, 1)


@pytest.mark.parametrize(
    "cls,expected", [("SEP", "SEP"), ("BS1", "BS1"), ("BS2", "BS2"), ("BS3", "BS3"), ("W", "GME"), ("GHZ", "GME")]
)
def test_rank_classification_matches_table_I(cls, expected):
    rng = np.random.default_rng(2)
    for _ in range(50):
        assert checks.rank_class(states.random_canonical_state(cls, rng)) == expected


@pytest.mark.parametrize(
    "cls,expected", [("SEP", "SEP"), ("BS1", "BS1"), ("BS2", "BS2"), ("BS3", "BS3"), ("W", "GME"), ("GHZ", "GME")]
)
def test_correlation_tensor_ranks_match_table_II(cls, expected):
    rng = np.random.default_rng(3)
    for _ in range(50):
        r = measures.correlation_tensor_ranks(states.random_canonical_state(cls, rng), tol=1e-9)
        assert measures.ranks_to_class(r) == expected, r


def test_three_tangle_separates_w_from_ghz():
    rng = np.random.default_rng(4)
    for _ in range(50):
        w = states.random_canonical_state("W", rng)
        g = states.random_canonical_state("GHZ", rng)
        assert measures.three_tangle_canonical(w) == pytest.approx(0, abs=1e-12)
        assert measures.three_tangle_canonical(g) > 0
        # shortcut agrees with the general CKW 3-tangle
        assert measures.three_tangle_canonical(g) == pytest.approx(measures.three_tangle_ckw(g), abs=1e-8)


def test_textbook_states():
    ghz = np.zeros(8, complex); ghz[[0, 7]] = 1 / np.sqrt(2)
    w = np.zeros(8, complex); w[[1, 2, 4]] = 1 / np.sqrt(3)
    assert measures.three_tangle_ckw(states.ket_to_dm(ghz)) == pytest.approx(1)
    assert measures.three_tangle_ckw(states.ket_to_dm(w)) == pytest.approx(0, abs=1e-12)
    assert measures.correlation_tensor_ranks(states.ket_to_dm(ghz)) == (2, 2, 2)
    assert measures.correlation_tensor_ranks(states.ket_to_dm(w)) == (3, 3, 3)


def test_canonical_18_captures_all_free_parameters():
    rng = np.random.default_rng(5)
    rho = states.random_canonical_state("GHZ", rng)
    v = features.flatten_rho(rho)
    assert np.allclose(features.unflatten_rho(v), rho)
    assert len(features.CANONICAL_18) == 18
    f18 = features.canonical_18(rho)
    # every non-zero upper-triangle entry except rho_77 appears in the 18 features
    nz = {(a, b) for a in range(8) for b in range(a, 8) if abs(rho[a, b]) > 1e-12} - {(7, 7)}
    names = {(int(n[-2]), int(n[-1])) for n in features.CANONICAL_18}
    assert nz <= names
    assert np.isclose(np.linalg.norm(f18) > 0, True)
