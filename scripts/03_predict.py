#!/usr/bin/env python3
"""
Step 3 - Score the portfolio and compute per-project focus areas.

Loads the deployed model, scores every project, computes the top local risk
drivers (focus areas) for each, and writes the predictions table.

Run: python scripts/03_predict.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seco_risk_radar import database as db          # noqa: E402
from seco_risk_radar.explain import local_factors   # noqa: E402
from seco_risk_radar.model import load_model, score_portfolio  # noqa: E402


def main() -> None:
    conn = db.connect()
    projects = db.read_projects(conn)
    if projects.empty:
        raise SystemExit("No projects found. Run scripts/01_run_pipeline.py first.")

    pipe = load_model()
    preds = score_portfolio(pipe, projects)

    # Per-project focus areas (local attribution). Reference = full portfolio.
    factor_json = []
    for _, row in projects.iterrows():
        drivers = local_factors(pipe, row, reference=projects, top_k=5)
        factor_json.append(json.dumps(drivers, default=str))
    preds["top_factors"] = factor_json

    db.save_predictions(conn, preds)
    conn.close()

    top = preds.sort_values("risk_score", ascending=False).head(5)
    print("[predict] top 5 highest-risk projects:")
    print(top[["project_id", "pred_band", "risk_score"]].to_string(index=False))
    print(f"[predict] wrote {len(preds)} predictions to SQLite.")


if __name__ == "__main__":
    main()
