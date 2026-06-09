"""
Feature engineering / preprocessing.

A single source of truth for which columns are features (and their types) so
training and inference can never drift apart. The preprocessing is wrapped in a
ColumnTransformer that lives *inside* the model Pipeline, which means:

* no train/serve skew (the exact same transform is persisted with the model),
* raw project rows can be scored directly without manual preprocessing,
* explanations can be computed in the original, human-readable feature space.

Note `permit_processing_days` is intentionally included as a feature even
though the synthetic label ignores it. A good model should learn to give it low
importance; checking that is part of evaluating the AI component.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "risk_severity"

CATEGORICAL_FEATURES = [
    "canton",
    "region",
    "building_type",
    "works_type",
    "structural_system",
    "foundation_type",
    "season_started",
]

NUMERIC_FEATURES = [
    "gross_floor_area_m2",
    "num_floors",
    "existing_structure_age_yrs",
    "contractor_experience",
    "site_complexity",
    "estimated_cost_eur",
    "permit_processing_days",   # red herring, kept on purpose
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns that exist in the data but must never be fed to the model
# (they would leak the answer).
LEAKAGE_COLUMNS = ["latent_risk_score", "risk_severity", "project_id"]


def split_features_target(df: pd.DataFrame):
    """Return (X, y) with only model-visible features in X."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy() if TARGET in df.columns else None
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals, scale numerics. Robust to unseen cantons."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
