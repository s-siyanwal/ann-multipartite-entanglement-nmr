"""Univariate feature ranking by ANOVA F-value (paper Sec. IV A, Table III)."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import f_classif

from .features import CANONICAL_18


def anova_ranking(df: pd.DataFrame, target: str = "class_code", features=CANONICAL_18) -> pd.DataFrame:
    """ANOVA F-value of each feature against ``target`` ('class_code' for SLOCC, 'gme' for GME).

    Returns a table sorted by descending F-value; the N-th ANN model of the paper uses the
    top-N features of this ranking.
    """
    f, p = f_classif(df[list(features)].to_numpy(), df[target].to_numpy())
    out = pd.DataFrame({"feature": list(features), "F": f, "p": p})
    return out.sort_values("F", ascending=False, ignore_index=True)


def top_n(ranking: pd.DataFrame, n: int) -> list[str]:
    return ranking["feature"].head(n).tolist()
