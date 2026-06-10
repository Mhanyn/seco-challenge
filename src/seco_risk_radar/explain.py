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
def _map_to_original_features(feature_names) -> list[str]:
    """Map each transformed column (e.g. 'cat__works_type_Renovation') back to
    its source feature ('works_type'), so SHAP values can be aggregated into the
    original, human-readable feature space."""
    mapped = []
    for name in feature_names:
        stripped = name.split("__", 1)[1] if "__" in name else name
        original = stripped
        if stripped not in NUMERIC_FEATURES:
            for feat in CATEGORICAL_FEATURES:
                if stripped == feat or stripped.startswith(feat + "_"):
                    original = feat
                    break
        mapped.append(original)
    return mapped


def shap_local(
    pipe: Pipeline,
    project_row: pd.Series,
    reference: pd.DataFrame,
    top_k: int = 5,
    class_name: str = "High",
) -> list[dict]:  # pragma: no cover - runs only when `shap` is installed
    """Optional SHAP attribution for a single project, in the original feature space.

    Auto-selects the correct explainer for whichever model is deployed
    (LinearExplainer for logistic regression, TreeExplainer for gradient
    boosting, a model-agnostic explainer otherwise), computes SHAP values for
    the chosen class, then aggregates the one-hot columns back to their source
    feature so the output matches :func:`local_factors`.

    Requires ``pip install shap``. Returns a list of
    ``{feature, value, contribution}`` sorted by contribution (risk-raising first).
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP not installed. Run `pip install shap`, or use local_factors() "
            "(the default explainer, which needs no extra dependency)."
        ) from exc

    import numpy as np

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    classes = list(clf.classes_)
    cls_idx = classes.index(class_name) if class_name in classes else len(classes) - 1

    # Background sample for the explainer (capped for speed).
    bg = reference[FEATURES]
    if len(bg) > 100:
        bg = bg.sample(100, random_state=42)
    bg_t = np.asarray(pre.transform(bg))
    x_t = np.asarray(pre.transform(pd.DataFrame([project_row[FEATURES].to_dict()])))

    name = type(clf).__name__.lower()
    is_linear = any(k in name for k in ("logistic", "linear", "ridge", "sgd"))
    is_tree = any(k in name for k in ("gradientboosting", "forest", "tree", "xgb", "lgbm", "catboost"))

    if is_linear:
        values = shap.LinearExplainer(clf, bg_t).shap_values(x_t)
    elif is_tree:
        values = shap.TreeExplainer(clf).shap_values(x_t)
    else:
        values = shap.Explainer(clf.predict_proba, bg_t)(x_t).values

    # Normalise the various SHAP output shapes to a 1-D vector for our row/class.
    if isinstance(values, list):                  # list of (n_samples, n_feat) per class
        row = np.asarray(values[min(cls_idx, len(values) - 1)])[0]
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:                         # (n_samples, n_feat, n_classes)
            row = arr[0, :, min(cls_idx, arr.shape[2] - 1)]
        elif arr.ndim == 2:                       # (n_samples, n_feat)
            row = arr[0]
        else:
            row = arr.ravel()

    # Aggregate transformed columns back to original features.
    originals = _map_to_original_features(list(pre.get_feature_names_out()))
    agg: dict[str, float] = {}
    for orig, val in zip(originals, row):
        agg[orig] = agg.get(orig, 0.0) + float(val)

    factors = [
        {"feature": f, "value": project_row[f], "contribution": round(v, 4)}
        for f, v in agg.items()
    ]
    factors.sort(key=lambda d: d["contribution"], reverse=True)
    return [c for c in factors if c["contribution"] > 0][:top_k]
