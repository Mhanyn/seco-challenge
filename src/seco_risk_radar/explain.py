"""
Explainability layer -> the "explanations + suggested focus areas" output.

Two levels:

* GLOBAL: permutation importance on the held-out set. Model-agnostic, and a
  good sanity check that the red-herring feature ranks near zero.

* LOCAL (per project): a model-agnostic ABLATION / occlusion attribution. For
  each feature we measure how much P(High)+0.5*P(Medium) drops when that
  feature is reset to a neutral reference value (median for numerics, mode for
  categoricals). The features whose removal lowers the risk the most are the
  drivers of *this* project's score -> the inspector's focus areas.

Why ablation rather than SHAP as the default: the deployed model is a sklearn
Pipeline with one-hot encoding, so ablation lets us attribute in the ORIGINAL,
human-readable feature space ("Renovation", "Load-bearing masonry") instead of
encoded columns. It is fully model-agnostic and has no extra dependency. SHAP
is wired in behind an optional flag for game-theoretically consistent values.
"""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    split_features_target,
)


def _risk_score(pipe: Pipeline, X: pd.DataFrame):
    classes = list(pipe.named_steps["clf"].classes_)
    proba = pipe.predict_proba(X)
    idx = {c: i for i, c in enumerate(classes)}
    high = proba[:, idx["High"]] if "High" in idx else 0.0
    med = proba[:, idx["Medium"]] if "Medium" in idx else 0.0
    return high + 0.5 * med


def global_importance(pipe: Pipeline, df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Permutation importance over the full feature set (higher = more used)."""
    X, y = split_features_target(df)
    result = permutation_importance(
        pipe, X, y, n_repeats=8, random_state=seed, scoring="accuracy"
    )
    return (
        pd.DataFrame({"feature": FEATURES, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _reference_row(reference: pd.DataFrame) -> dict:
    """Neutral baseline: median for numerics, mode for categoricals."""
    ref = {}
    for c in NUMERIC_FEATURES:
        ref[c] = reference[c].median()
    for c in CATEGORICAL_FEATURES:
        ref[c] = reference[c].mode().iloc[0]
    return ref


def local_factors(
    pipe: Pipeline,
    project_row: pd.Series,
    reference: pd.DataFrame,
    top_k: int = 5,
) -> list[dict]:
    """Ablation attribution for a single project.

    Returns a list of {feature, value, contribution} sorted by how much each
    feature raises this project's risk score above the neutral baseline.
    """
    ref = _reference_row(reference)
    base_dict = project_row[FEATURES].to_dict()
    base_X = pd.DataFrame([base_dict])
    base_score = float(_risk_score(pipe, base_X)[0])

    contributions = []
    for feat in FEATURES:
        ablated_dict = dict(base_dict)
        ablated_dict[feat] = ref[feat]
        ablated_score = float(_risk_score(pipe, pd.DataFrame([ablated_dict]))[0])
        # Positive contribution = this feature's actual value pushes risk UP.
        contributions.append(
            {
                "feature": feat,
                "value": project_row[feat],
                "contribution": round(base_score - ablated_score, 4),
            }
        )

    contributions.sort(key=lambda d: d["contribution"], reverse=True)
    # Only return features that actually increase risk (positive contribution).
    drivers = [c for c in contributions if c["contribution"] > 0][:top_k]
    return drivers


# ---- Optional SHAP path (used only if shap is installed) --------------------
def shap_local(pipe: Pipeline, project_row: pd.Series):  # pragma: no cover
    """Optional: SHAP values for the deployed model. Requires `pip install shap`."""
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "SHAP not installed. `pip install shap` or use local_factors()."
        ) from exc
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X = pd.DataFrame([project_row[FEATURES].to_dict()])
    Xt = pre.transform(X)
    explainer = shap.TreeExplainer(clf)
    return explainer.shap_values(Xt)
