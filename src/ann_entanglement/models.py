"""Classifiers (paper Sec. III A, IV B-C, V A) with the hyper-parameters stated in the paper.

- :func:`build_slocc_ann`  N x 6 x 6, hidden 'linear', output softmax, categorical cross-entropy
- :func:`build_gme_ann`    N x (N + N mod 2)/2 x 1, output sigmoid, binary cross-entropy (GME if p0 >= 0.5)
- :func:`early_stopping`   min_delta = 0.01, patience = 20, best weights restored
- :data:`TRAIN_KW`         100 epochs; batch 1000 (SLOCC) / 500 (GME); shuffled every epoch
- :func:`svm_classifier`   linear SVC, C = 0.1                                     (Sec. V A)
- :func:`knn_classifier`   GME: k=18, uniform, p=1; SLOCC: k=20, distance, p=2     (Sec. V A)

Adam, initial learning rate 1e-3. The Dec-2022 exploratory notebooks used 'relu' hidden
layers; the paper states 'linear', which is the default here (``hidden_activation`` switches it).
TensorFlow is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

TRAIN_KW = {
    "slocc": {"epochs": 100, "batch_size": 1000, "shuffle": True},
    "gme": {"epochs": 100, "batch_size": 500, "shuffle": True},
}


def _keras():
    from tensorflow import keras  # noqa: PLC0415  (lazy, heavy import)

    return keras


def gme_hidden_units(n_features: int) -> int:
    """(N + N mod 2) / 2 hidden neurons for the binary GME model (Fig. 1a, Table V)."""
    return (n_features + n_features % 2) // 2


def build_slocc_ann(n_features: int, hidden: int = 6, hidden_activation: str = "linear", learning_rate: float = 1e-3):
    """Labels are one-hot vectors H(rho) over the six classes (Sec. III B)."""
    keras = _keras()
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(hidden, activation=hidden_activation),
            keras.layers.Dense(6, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_gme_ann(n_features: int, hidden_activation: str = "linear", learning_rate: float = 1e-3):
    keras = _keras()
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(gme_hidden_units(n_features), activation=hidden_activation),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def early_stopping(monitor: str = "val_loss"):
    keras = _keras()
    return keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0.01, patience=20, restore_best_weights=True)


def svm_classifier(C: float = 0.1, **kw):
    return SVC(kernel="linear", C=C, **kw)


def knn_classifier(task: str = "slocc", **kw):
    params = {
        "gme": {"n_neighbors": 18, "weights": "uniform", "p": 1},
        "slocc": {"n_neighbors": 20, "weights": "distance", "p": 2},
    }[task]
    params.update(kw)
    return KNeighborsClassifier(**params)
