"""
Model layer.

We train two models and keep the comparison honest:

* Baseline: multinomial Logistic Regression. Cheap, transparent, sets a floor.
* Main:     HistGradientBoostingClassifier. Strong tabular model, captures the
            non-linear size effect and the masonry x renovation interaction
            that the baseline cannot.

Both are wrapped in a Pipeline with the shared preprocessor, so the persisted
artifact scores raw project rows directly (no train/serve skew).

Outputs
-------
For triage we need a single sortable number, not just a class. We define:

    risk_score = P(High) + 0.5 * P(Medium)

a continuous 0..1 priority that ranks the portfolio for inspector allocation.
We report accuracy, macro-F1 and the High-class recall (catching high-risk
projects early is the KPI that matters most for SECO).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import MODELS_DIR, Settings, ensure_dirs
from .features import build_preprocessor, split_features_target

CLASSES = ["Low", "Medium", "High"]
MODEL_PATH = MODELS_DIR / "risk_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: dict = field(default_factory=dict)


def _make_pipeline(kind: str, seed: int) -> Pipeline:
    pre = build_preprocessor()
    if kind == "logreg":
        # sklearn >=1.7 handles multiclass as multinomial by default.
        clf = LogisticRegression(max_iter=2000, C=1.0)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=3,
            max_iter=300,
            min_samples_leaf=25,
            l2_regularization=2.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
    else:  # pragma: no cover
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def _evaluate(pipe: Pipeline, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "high_recall": round(
            recall_score(y_test, pred, labels=["High"], average="macro"), 4
        ),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }


def train(df: pd.DataFrame, settings: Settings | None = None) -> TrainResult:
    """Train baseline + main model, pick the main model, persist it + metrics."""
    settings = settings or Settings()
    ensure_dirs()
    X, y = split_features_target(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.seed, stratify=y
    )

    metrics = {}
    pipelines = {}
    for kind in ("logreg", "hgb"):
        pipe = _make_pipeline(kind, settings.seed)
        pipe.fit(X_tr, y_tr)
        metrics[kind] = _evaluate(pipe, X_te, y_te)
        pipelines[kind] = pipe
        print(f"[train] {kind:6s} acc={metrics[kind]['accuracy']} "
              f"macroF1={metrics[kind]['macro_f1']} "
              f"highRecall={metrics[kind]['high_recall']}")

    # Deploy whichever model wins on held-out macro-F1 (honest, data-driven).
    best_kind = max(metrics, key=lambda k: metrics[k]["macro_f1"])
    best = pipelines[best_kind]
    joblib.dump(best, MODEL_PATH)
    summary = {
        "deployed_model": best_kind,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "classes": CLASSES,
        "metrics": metrics,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[train] deployed '{best_kind}' (best macro-F1) -> {MODEL_PATH}")
    return TrainResult(pipeline=best, metrics=summary)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def _proba_frame(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    proba = pipe.predict_proba(X)
    classes = list(pipe.named_steps["clf"].classes_)
    out = pd.DataFrame(proba, columns=[f"prob_{c.lower()}" for c in classes])
    # Guarantee all three columns exist even if a class is absent.
    for c in CLASSES:
        col = f"prob_{c.lower()}"
        if col not in out:
            out[col] = 0.0
    return out[[f"prob_{c.lower()}" for c in CLASSES]]


def score_portfolio(pipe: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Return per-project predictions: band, continuous risk_score, class probs."""
    X, _ = split_features_target(df)
    probs = _proba_frame(pipe, X)
    risk_score = probs["prob_high"] + 0.5 * probs["prob_medium"]
    pred_band = pipe.predict(X)
    return pd.DataFrame(
        {
            "project_id": df["project_id"].to_numpy(),
            "pred_band": pred_band,
            "risk_score": risk_score.round(4).to_numpy(),
            "prob_low": probs["prob_low"].round(4).to_numpy(),
            "prob_medium": probs["prob_medium"].round(4).to_numpy(),
            "prob_high": probs["prob_high"].round(4).to_numpy(),
        }
    )
