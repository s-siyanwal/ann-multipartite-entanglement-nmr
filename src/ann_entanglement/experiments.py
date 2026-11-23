"""End-to-end protocol behind Tables IV-VII (paper Sec. IV B-C, V A). RE-IMPLEMENTATION.

The final 2023 script that produced Tables IV-VII was not among the recovered files; this
module re-implements the protocol exactly as described in the paper:

1. simulate 2x10^3 states per SLOCC class (12x10^3 total), keep the 18 canonical features;
2. rank the features by ANOVA F-value (Table III), separately for the SLOCC and GME targets;
3. for N = 1..18 train on the top-N features; for each of 100 random 18:12 validation:test
   splits of the 30 experimental states, train with early stopping on the validation set and
   score on the held-out test set -> mu(A_v), mu(L_v), mu(A_t), mu(L_t), and the projected
   overall accuracy A_T = 0.6 mu(A_v) + 0.4 mu(A_t) with the error of Eqs. (11)-(12);
4. SVM / KNN benchmarks (A1, A2) trained on the simulated set and scored on all 30 states.

Numbers will not match the paper digit-for-digit (different random states, TF version).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import dataset, models, selection

SLOCC_K = 6


def _onehot(y, k=SLOCC_K):
    return np.eye(k, dtype="float32")[np.asarray(y, int)]


def prepare(n_per_class: int = 2000, seed: int = 0):
    train = dataset.generate_training_set(n_per_class, seed=seed)
    exp = dataset.load_experimental()
    exp_x = pd.concat([exp[["state_id", "class", "class_code", "gme"]], dataset.experimental_features(exp)], axis=1)
    return train, exp_x


def overall_accuracy(acc_v, acc_t, n_val: int = 18, n_total: int = 30):
    """Eqs. (11)-(12): mu(A) = a mu(A_v) + b mu(A_t), sigma(A)^2 = a^2 s_v^2 + b^2 s_t^2 + 2ab cov(A_v, A_t)."""
    a, b = n_val / n_total, 1 - n_val / n_total
    acc_v, acc_t = np.asarray(acc_v, float), np.asarray(acc_t, float)
    cov = np.cov(acc_v, acc_t, bias=True)[0, 1] if len(acc_v) > 1 else 0.0
    var = a * a * acc_v.var() + b * b * acc_t.var() + 2 * a * b * cov
    return a * acc_v.mean() + b * acc_t.mean(), float(np.sqrt(max(var, 0.0)))


def ann_sweep(task: str, train: pd.DataFrame, exp_x: pd.DataFrame, n_values=range(1, 19), n_splits: int = 100,
              seed: int = 0, verbose: int = 0) -> pd.DataFrame:
    """One row per N with the columns of Table IV (task='slocc') or Table V (task='gme')."""
    import tensorflow as tf  # noqa: PLC0415

    target = "class_code" if task == "slocc" else "gme"
    ranking = selection.anova_ranking(train, target=target)
    rows = []
    for n in n_values:
        cols = selection.top_n(ranking, n)
        x = train[cols].to_numpy("float32")
        xe = exp_x[cols].to_numpy("float32")
        if task == "slocc":
            y, ye = _onehot(train[target]), _onehot(exp_x[target])
        else:
            y, ye = train[target].to_numpy("float32"), exp_x[target].to_numpy("float32")
        acc_v, loss_v, acc_t, loss_t, acc_all = [], [], [], [], []
        for i, (vi, ti) in enumerate(dataset.validation_test_splits(exp_x, n_splits=n_splits, seed=seed)):
            tf.keras.utils.set_random_seed(seed + i)
            m = models.build_slocc_ann(n) if task == "slocc" else models.build_gme_ann(n)
            m.fit(x, y, validation_data=(xe[vi], ye[vi]), callbacks=[models.early_stopping()],
                  verbose=verbose, **models.TRAIN_KW[task])
            lv, av = m.evaluate(xe[vi], ye[vi], verbose=0)
            lt, at = m.evaluate(xe[ti], ye[ti], verbose=0)
            _, aa = m.evaluate(xe, ye, verbose=0)
            acc_v.append(av); loss_v.append(lv); acc_t.append(at); loss_t.append(lt); acc_all.append(aa)
        net = f"{n}x6x6" if task == "slocc" else f"{n}x{models.gme_hidden_units(n)}x1"
        a_t, s_t = overall_accuracy(acc_v, acc_t)
        rows.append({"N": n, "network": net, "features": ", ".join(cols),
                     "mu_Av": np.mean(acc_v), "sd_Av": np.std(acc_v), "mu_Lv": np.mean(loss_v), "sd_Lv": np.std(loss_v),
                     "mu_At": np.mean(acc_t), "sd_At": np.std(acc_t), "mu_Lt": np.mean(loss_t), "sd_Lt": np.std(loss_t),
                     "A_T": a_t, "sd_AT": s_t, "acc_all30": np.mean(acc_all)})
    return pd.DataFrame(rows)


def baselines(task: str, train: pd.DataFrame, exp_x: pd.DataFrame, n_values=range(1, 19)) -> pd.DataFrame:
    """A1 (linear SVM, C=0.1) and A2 (KNN) accuracy on all 30 experimental states for each N (Fig. 4)."""
    target = "class_code" if task == "slocc" else "gme"
    ranking = selection.anova_ranking(train, target=target)
    rows = []
    for n in n_values:
        cols = selection.top_n(ranking, n)
        x, y = train[cols].to_numpy(), train[target].to_numpy()
        xe, ye = exp_x[cols].to_numpy(), exp_x[target].to_numpy()
        a1 = models.svm_classifier().fit(x, y).score(xe, ye)
        a2 = models.knn_classifier(task).fit(x, y).score(xe, ye)
        rows.append({"N": n, "A1_svm": a1, "A2_knn": a2})
    return pd.DataFrame(rows)
