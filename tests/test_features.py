"""Tests for feature handling and the model train/score path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seco_risk_radar.config import Settings
from seco_risk_radar.features import FEATURES, LEAKAGE_COLUMNS, split_features_target
from seco_risk_radar.model import score_portfolio, train
from seco_risk_radar.synthesize import generate_portfolio


def test_no_leakage_in_features():
    """The target / latent score must never be in the feature matrix."""
    df = generate_portfolio(Settings(n_projects=100, seed=2))
    X, y = split_features_target(df)
    for leak in LEAKAGE_COLUMNS:
        assert leak not in X.columns
    assert set(X.columns) == set(FEATURES)


def test_train_beats_majority_baseline():
    """Deployed model accuracy should beat always-predict-majority (~0.55)."""
    df = generate_portfolio(Settings(n_projects=1200, seed=4))
    result = train(df, Settings(n_projects=1200, seed=4))
    acc = result.metrics["metrics"][result.metrics["deployed_model"]]["accuracy"]
    assert acc > 0.55


def test_score_portfolio_shape_and_range():
    df = generate_portfolio(Settings(n_projects=300, seed=6))
    result = train(df, Settings(n_projects=300, seed=6))
    preds = score_portfolio(result.pipeline, df)
    assert len(preds) == len(df)
    assert preds["risk_score"].between(0, 1).all()
    probs = preds[["prob_low", "prob_medium", "prob_high"]].sum(axis=1)
    # Probabilities are stored rounded to 4 dp, so allow small rounding error.
    assert (abs(probs - 1.0) < 2e-3).all()
