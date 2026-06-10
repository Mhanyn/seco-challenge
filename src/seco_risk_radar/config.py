from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = two levels up from this file (src/seco_risk_radar/config.py).
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "risk_radar.db"

# Global seed so every run (sampling, train/test split, model) is reproducible.
SEED = 42

# data.public.lu udata API + the real dataset / resource IDs for
# "Autorisations de bâtir" (STATEC, CC0). Verified from the dataset page.
DATA_PUBLIC_LU_API = "https://data.public.lu/api/1"
STATEC_DATASET_ID = "6523270543d5730744b8943f"

# Resource permalinks (download endpoints) for the slices we care about.
# Pattern: https://data.public.lu/en/datasets/r/{resource_id}
STATEC_RESOURCES: dict[str, str] = {
    "buildings_by_type": "9263bdcd-3465-48c1-a7c7-759ca34c3ade",
    "buildings_by_type_canton": "c7195403-1f49-4012-bace-5fb73f847fb2",
    "dwellings_by_type_canton": "32b40575-e5c2-47ac-8ef6-0132e51d80eb",
}


def resource_url(resource_id: str) -> str:
    """Build the stable download URL for a data.public.lu resource."""
    return f"https://data.public.lu/en/datasets/r/{resource_id}"


@dataclass
class Settings:
    """Runtime knobs that a reviewer might want to change without editing code."""

    n_projects: int = 1500             # size of the synthetic portfolio
    test_size: float = 0.25
    seed: int = SEED
    # LLM briefing provider: "auto" (use a key if present), or "off".
    llm_mode: str = field(default_factory=lambda: os.getenv("RISK_RADAR_LLM", "auto"))


def ensure_dirs() -> None:
    """Create the data/model directories if they do not yet exist."""
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
