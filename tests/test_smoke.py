"""Smoke test: import the package, generate a tiny dataset, run one ANN forward pass."""

import numpy as np

from ann_entanglement import dataset, selection
from ann_entanglement.features import CANONICAL_18


def test_generate_rank_and_forward_pass():
    df = dataset.generate_training_set(n_per_class=20, seed=0)
    assert len(df) == 120 and set(df["class_code"]) == set(range(6))
    rank = selection.anova_ranking(df, target="class_code")
    assert set(rank["feature"]) == set(CANONICAL_18)

    from ann_entanglement import models

    cols = selection.top_n(rank, 6)
    x = df[cols].to_numpy("float32")
    slocc = models.build_slocc_ann(len(cols))
    p = slocc.predict(x[:4], verbose=0)
    assert p.shape == (4, 6) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    gme = models.build_gme_ann(4)
    assert gme.layers[0].units == 2  # (4 + 0) / 2
    q = gme.predict(df[selection.top_n(selection.anova_ranking(df, "gme"), 4)].to_numpy("float32")[:4], verbose=0)
    assert q.shape == (4, 1) and np.all((q >= 0) & (q <= 1))


def test_experimental_data_and_splits():
    exp = dataset.load_experimental()
    assert len(exp) == 30 and (exp["class"].value_counts() == 5).all()
    splits = list(dataset.validation_test_splits(exp, n_splits=100))
    assert len(splits) == 100
    v, t = splits[0]
    assert len(v) == 18 and len(t) == 12
    assert len({tuple(v) for v, _ in splits}) == 100


def test_overall_accuracy_eq_11_12():
    from ann_entanglement.experiments import overall_accuracy

    mu, sd = overall_accuracy([1.0, 1.0], [0.5, 0.5])
    assert np.isclose(mu, 0.6 * 1.0 + 0.4 * 0.5) and np.isclose(sd, 0)
    av, at = np.array([0.8, 0.9, 1.0]), np.array([0.5, 0.75, 1.0])
    mu, sd = overall_accuracy(av, at)
    expect = 0.36 * av.var() + 0.16 * at.var() + 0.48 * np.cov(av, at, bias=True)[0, 1]
    assert np.isclose(sd**2, expect)
