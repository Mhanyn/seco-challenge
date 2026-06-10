"""
Synthetic portfolio generation.

This is the "hybrid" core of the project:

* BASE FEATURES are sampled from the real STATEC building-stock prior
  (canton + building_type distribution), so the portfolio's geography and
  building mix reflect genuine Luxembourg construction activity.
* ADDITIONAL FEATURES (structural system, works type, area, contractor
  experience, site complexity, ...) are plausible SECO-relevant attributes a
  technical-control body would actually record on a file.
* The TARGET (risk_severity) is SYNTHETIC. SECO's real historical inspection
  outcomes are confidential and were not provided, so we generate the label
  from a documented latent function. Raimondo explicitly invited this:
  "scope a challenge version with synthetic data ... each with features and a
  label: high/medium/low risk."

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import taxonomy as tx
from .config import Settings
from .ingest import build_building_stock_prior

# ---- Documented latent-risk weights (domain priors, NOT learned truth) ------
# Higher value => pushes a project toward higher technical risk.
_WORKS_RISK = {"New build": 0.0, "Extension": 0.35, "Renovation": 0.75}
_STRUCT_RISK = {
    "Reinforced concrete frame": 0.0,
    "Steel frame": 0.05,
    "Timber frame": 0.20,
    "Mixed / hybrid": 0.35,
    "Load-bearing masonry": 0.45,   # older / heritage stock, more unknowns
}
_FOUND_RISK = {"Shallow / strip": 0.0, "Raft": 0.10, "Piled / deep": 0.30}
_SEASON_RISK = {"Summer": 0.0, "Spring": 0.05, "Autumn": 0.10, "Winter": 0.25}


def _sample_base(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Sample canton + building_type from the real STATEC prior."""
    prior = build_building_stock_prior()
    idx = rng.choice(len(prior), size=n, p=prior["weight"].to_numpy())
    base = prior.iloc[idx][["canton", "region", "building_type"]].reset_index(drop=True)
    return base


def _engineer(rng: np.random.Generator, base: pd.DataFrame) -> pd.DataFrame:
    """Attach realistic project attributes to the sampled base rows."""
    n = len(base)
    df = base.copy()
    df["project_id"] = [f"LU-{2026:04d}-{i:05d}" for i in range(1, n + 1)]

    df["works_type"] = rng.choice(tx.WORKS_TYPES, size=n, p=[0.55, 0.20, 0.25])
    df["structural_system"] = rng.choice(
        tx.STRUCTURAL_SYSTEMS, size=n, p=[0.45, 0.10, 0.10, 0.15, 0.20]
    )
    df["foundation_type"] = rng.choice(tx.FOUNDATION_TYPES, size=n, p=[0.55, 0.25, 0.20])
    df["season_started"] = rng.choice(tx.SEASONS, size=n)

    # Floor area depends on building type (log-normal, type-specific scale).
    type_scale = {
        "Residential - single dwelling": 180,
        "Residential - multi dwelling": 1200,
        "Residential - community": 2200,
        "Non-residential": 1600,
    }
    scale = df["building_type"].map(type_scale).to_numpy()
    df["gross_floor_area_m2"] = np.round(
        scale * rng.lognormal(mean=0.0, sigma=0.55, size=n)
    ).astype(int).clip(40, 60000)

    df["num_floors"] = (
        1 + rng.poisson(lam=df["building_type"].map(
            {"Residential - single dwelling": 1.2,
             "Residential - multi dwelling": 3.5,
             "Residential - community": 2.5,
             "Non-residential": 2.0}).to_numpy())
    ).clip(1, 25)

    # Existing structure age only meaningful for renovation/extension.
    age = rng.integers(0, 90, size=n)
    df["existing_structure_age_yrs"] = np.where(
        df["works_type"].eq("New build"), 0, age
    )

    df["contractor_experience"] = rng.beta(a=5, b=2, size=n).round(3)  # skewed high
    df["site_complexity"] = rng.beta(a=2, b=3, size=n).round(3)        # skewed low/mid

    # Cost proxy ~ area * type unit cost * noise (EUR).
    unit_cost = df["building_type"].map(
        {"Residential - single dwelling": 2200,
         "Residential - multi dwelling": 2000,
         "Residential - community": 2400,
         "Non-residential": 1800}).to_numpy()
    df["estimated_cost_eur"] = np.round(
        df["gross_floor_area_m2"] * unit_cost * rng.normal(1.0, 0.12, size=n)
    ).astype(int).clip(20000, None)

    # RED HERRING: administrative lead time, deliberately unrelated to risk.
    df["permit_processing_days"] = rng.integers(20, 220, size=n)
    return df


def _latent_risk(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Compute the latent risk score (higher = riskier). Documented weights."""
    n = len(df)
    z = np.zeros(n)

    z += df["works_type"].map(_WORKS_RISK).to_numpy()
    z += df["structural_system"].map(_STRUCT_RISK).to_numpy()
    z += df["foundation_type"].map(_FOUND_RISK).to_numpy()
    z += df["season_started"].map(_SEASON_RISK).to_numpy()

    # Lower contractor experience -> higher risk (strong driver).
    z += (1.0 - df["contractor_experience"].to_numpy()) * 0.9
    # Higher site complexity -> higher risk.
    z += df["site_complexity"].to_numpy() * 0.8
    # Older existing structure (renovation) -> higher risk.
    z += (df["existing_structure_age_yrs"].to_numpy() / 90.0) * 0.5

    # NON-LINEAR size effect: very small (informal) and very large (mega) are
    # riskier than mid-size. U-shape on log-area.
    la = np.log10(df["gross_floor_area_m2"].to_numpy())
    z += 0.40 * (la - la.mean()) ** 2

    # INTERACTION: masonry renovations are disproportionately risky.
    interaction = (
        df["structural_system"].eq("Load-bearing masonry")
        & df["works_type"].eq("Renovation")
    ).to_numpy().astype(float)
    z += 1.0 * interaction

    # Genuine noise so the label is probabilistic, not deterministic.
    z += rng.normal(0.0, 0.38, size=n)
    return z


def generate_portfolio(settings: Settings | None = None) -> pd.DataFrame:
    """Generate the full synthetic project portfolio with a 3-class risk label."""
    settings = settings or Settings()
    rng = np.random.default_rng(settings.seed)

    base = _sample_base(rng, settings.n_projects)
    df = _engineer(rng, base)

    z = _latent_risk(df, rng)
    # Map latent score to tertile bands so classes are reasonably balanced.
    lo, hi = np.quantile(z, [0.55, 0.82])
    band = np.where(z >= hi, "High", np.where(z >= lo, "Medium", "Low"))
    df["risk_severity"] = band

    # Also expose the (hidden) latent probability for transparency / debugging.
    df["latent_risk_score"] = (1 / (1 + np.exp(-(z - z.mean()) / z.std()))).round(4)
    return df


if __name__ == "__main__":
    out = generate_portfolio()
    print(out["risk_severity"].value_counts(normalize=True).round(3))
    print(out.head())
