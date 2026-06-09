"""Tests for the synthetic portfolio generator."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seco_risk_radar.config import Settings
from seco_risk_radar.synthesize import generate_portfolio


def test_schema_and_size():
    df = generate_portfolio(Settings(n_projects=200, seed=1))
    assert len(df) == 200
    for col in ["project_id", "canton", "building_type", "risk_severity"]:
        assert col in df.columns
    assert df["project_id"].is_unique


def test_reproducible():
    a = generate_portfolio(Settings(n_projects=150, seed=7))
    b = generate_portfolio(Settings(n_projects=150, seed=7))
    assert a.equals(b)


def test_all_three_bands_present():
    df = generate_portfolio(Settings(n_projects=500, seed=3))
    assert set(df["risk_severity"].unique()) == {"Low", "Medium", "High"}


def test_red_herring_is_independent_of_label():
    """permit_processing_days must not correlate with the latent risk score."""
    df = generate_portfolio(Settings(n_projects=2000, seed=5))
    corr = np.corrcoef(df["permit_processing_days"], df["latent_risk_score"])[0, 1]
    assert abs(corr) < 0.1, f"red herring leaked into label (corr={corr:.3f})"


def test_masonry_renovation_riskier():
    """The built-in interaction should make masonry renovations riskier on average."""
    df = generate_portfolio(Settings(n_projects=2000, seed=9))
    mas_reno = df[(df.structural_system == "Load-bearing masonry")
                  & (df.works_type == "Renovation")]["latent_risk_score"].mean()
    overall = df["latent_risk_score"].mean()
    assert mas_reno > overall
