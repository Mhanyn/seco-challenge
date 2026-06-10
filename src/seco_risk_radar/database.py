from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from .config import DB_PATH, RAW_DIR, ensure_dirs

_SCHEMA = """
CREATE TABLE IF NOT EXISTS provenance (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    project_id                TEXT PRIMARY KEY,
    canton                    TEXT,
    region                    TEXT,
    building_type             TEXT,
    works_type                TEXT,
    structural_system         TEXT,
    foundation_type           TEXT,
    season_started            TEXT,
    gross_floor_area_m2       INTEGER,
    num_floors                INTEGER,
    existing_structure_age_yrs INTEGER,
    contractor_experience     REAL,
    site_complexity           REAL,
    estimated_cost_eur        INTEGER,
    permit_processing_days    INTEGER,
    risk_severity             TEXT,
    latent_risk_score         REAL
);

CREATE TABLE IF NOT EXISTS predictions (
    project_id      TEXT PRIMARY KEY,
    pred_band       TEXT,
    risk_score      REAL,
    prob_low        REAL,
    prob_medium     REAL,
    prob_high       REAL,
    top_factors     TEXT,   -- JSON list of {feature, direction, contribution}
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_score ON predictions(risk_score DESC);
"""


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def load_projects(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Replace the projects table with the given portfolio."""
    df.to_sql("projects", conn, if_exists="replace", index=False)
    conn.commit()


def load_provenance(conn: sqlite3.Connection) -> None:
    """Copy the ingestion provenance file into the DB as an audit trail."""
    prov_file = RAW_DIR / "provenance.json"
    if not prov_file.exists():
        return
    prov = json.loads(prov_file.read_text())
    rows = [(k, json.dumps(v) if not isinstance(v, str) else v) for k, v in prov.items()]
    conn.executemany("INSERT OR REPLACE INTO provenance(key, value) VALUES (?, ?)", rows)
    conn.commit()


def save_predictions(conn: sqlite3.Connection, preds: pd.DataFrame) -> None:
    preds.to_sql("predictions", conn, if_exists="replace", index=False)
    conn.commit()


def read_projects(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM projects", conn)


def read_scored_portfolio(conn: sqlite3.Connection) -> pd.DataFrame:
    """Join projects + predictions for the dashboard, sorted by risk."""
    query = """
        SELECT p.*, pr.pred_band, pr.risk_score, pr.prob_low,
               pr.prob_medium, pr.prob_high, pr.top_factors
        FROM projects p
        LEFT JOIN predictions pr ON p.project_id = pr.project_id
        ORDER BY pr.risk_score DESC
    """
    return pd.read_sql(query, conn)
