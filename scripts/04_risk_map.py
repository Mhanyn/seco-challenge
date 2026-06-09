"""Build a Luxembourg canton risk map from the scored portfolio.

Usage:
    python scripts/04_risk_map.py                # auto: fetch boundaries, choropleth
    python scripts/04_risk_map.py --no-fetch     # force offline bubble map
    python scripts/04_risk_map.py --out my.png   # custom output path

Reads predictions from SQLite (run scripts 01-03 first), aggregates predicted
risk per canton, and writes a PNG. Uses real canton boundaries from
data.public.lu when reachable, otherwise a centroid bubble map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seco_risk_radar import database as db, geo  # noqa: E402
from seco_risk_radar.config import DB_PATH, DATA_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the canton risk map.")
    parser.add_argument("--no-fetch", action="store_true", help="skip boundary download (bubble map)")
    parser.add_argument("--force-fetch", action="store_true", help="re-download boundaries, ignore cache")
    parser.add_argument("--out", default=str(DATA_DIR / "maps" / "risk_by_canton.png"))
    args = parser.parse_args()

    conn = db.connect(DB_PATH)
    scored = db.read_scored_portfolio(conn)
    if scored.empty:
        raise SystemExit("No scored projects found. Run scripts 01-03 first.")

    by_canton = geo.aggregate_risk_by_canton(scored)

    geojson = None
    if not args.no_fetch:
        geojson = geo.fetch_canton_geojson(force=args.force_fetch)
        if geojson is None:
            print("[map] boundaries unavailable (offline?) -> bubble-map fallback")

    fig, mode = geo.build_risk_map(by_canton, geojson=geojson)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")

    print(f"[map] mode={mode}")
    print("[map] risk ranking by canton:")
    for _, row in by_canton.iterrows():
        print(
            f"       {row['canton']:<18} mean={row['mean_risk_score']:.3f} "
            f"high={int(row['n_high'])}/{int(row['n_projects'])}"
        )
    print(f"[map] wrote {out_path}")


if __name__ == "__main__":
    main()
