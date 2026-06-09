#!/usr/bin/env python3
"""
Step 2 - Train the risk-scoring model.

Reads the projects table, trains the baseline + gradient-boosted models,
prints/persists metrics, and saves the deployed pipeline.

Run: python scripts/02_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seco_risk_radar import database as db          # noqa: E402
from seco_risk_radar.explain import global_importance  # noqa: E402
from seco_risk_radar.model import train             # noqa: E402


def main() -> None:
    conn = db.connect()
    projects = db.read_projects(conn)
    conn.close()
    if projects.empty:
        raise SystemExit("No projects found. Run scripts/01_run_pipeline.py first.")

    result = train(projects)

    print("\n[train] global feature importance (top 8):")
    imp = global_importance(result.pipeline, projects)
    print(imp.head(8).to_string(index=False))

    # Sanity check: the red-herring feature should rank low.
    rank = imp.reset_index().query("feature == 'permit_processing_days'")
    if not rank.empty:
        print(f"\n[train] red-herring 'permit_processing_days' ranked "
              f"#{int(rank['index'].iloc[0]) + 1} of {len(imp)} (expected: low).")


if __name__ == "__main__":
    main()
