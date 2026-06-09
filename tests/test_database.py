"""Tests for the SQLite storage layer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seco_risk_radar import database as db
from seco_risk_radar.config import Settings
from seco_risk_radar.synthesize import generate_portfolio


def test_db_roundtrip(tmp_path):
    dbfile = tmp_path / "test.db"
    df = generate_portfolio(Settings(n_projects=120, seed=11))

    conn = db.connect(dbfile)
    db.init_db(conn)
    db.load_projects(conn, df)
    out = db.read_projects(conn)
    conn.close()

    assert len(out) == 120
    assert set(out["project_id"]) == set(df["project_id"])
