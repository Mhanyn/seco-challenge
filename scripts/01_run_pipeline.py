#!/usr/bin/env python3
"""
Step 1 - Data pipeline.

ingest (real STATEC provenance + building-stock prior)
  -> synthesize (sample a realistic project portfolio + synthetic risk label)
  -> store (SQLite: provenance + projects tables)

Run: python scripts/01_run_pipeline.py [--n 600]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seco_risk_radar import database as db          # noqa: E402
from seco_risk_radar.config import Settings         # noqa: E402
from seco_risk_radar.ingest import ingest           # noqa: E402
from seco_risk_radar.synthesize import generate_portfolio  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the project portfolio.")
    parser.add_argument("--n", type=int, default=1500, help="number of projects")
    args = parser.parse_args()

    settings = Settings(n_projects=args.n)

    # 1. Ingest (records provenance, builds the real-data sampling prior).
    ingest()

    # 2. Synthesize the portfolio from the prior + synthetic label.
    portfolio = generate_portfolio(settings)
    dist = portfolio["risk_severity"].value_counts(normalize=True).round(3).to_dict()
    print(f"[pipeline] generated {len(portfolio)} projects | class balance: {dist}")

    # 3. Store.
    conn = db.connect()
    db.init_db(conn)
    db.load_provenance(conn)
    db.load_projects(conn, portfolio)
    conn.close()
    print("[pipeline] stored projects + provenance in SQLite.")


if __name__ == "__main__":
    main()
