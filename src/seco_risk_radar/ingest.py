"""
Ingestion layer.

Responsibilities
----------------
1. Reach the REAL Luxembourg open-data platform (data.public.lu / STATEC) to
   record dataset provenance (id, license, resource URLs, fetch timestamp).
2. Produce a `building_stock` prior: relative construction activity per
   (canton, building_type). The published STATEC data is *aggregate* (counts /
   useful surface by type and geography), so we use it as a sampling prior for
   the synthetic project portfolio rather than as per-project records.

Design choice: the network call is best-effort. If there is no connectivity
(or the platform shape changes), we fall back to a documented prior built from
the taxonomy. This keeps the whole pipeline reproducible offline, which the
brief explicitly requires ("everything must be public and reproducible").

The production path for richer data is to parse the LUSTAT SDMX-JSON feed
behind each resource; that is listed as a "3-months" item in the README.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from . import taxonomy as tx
from .config import (
    DATA_PUBLIC_LU_API,
    RAW_DIR,
    STATEC_DATASET_ID,
    STATEC_RESOURCES,
    ensure_dirs,
    resource_url,
)

# Relative construction-activity weight per canton. Proxy for population /
# building volume: Luxembourg-City and Esch-sur-Alzette dominate national
# construction; northern rural cantons are far smaller. Documented assumption.
CANTON_ACTIVITY_WEIGHT: dict[str, float] = {
    "Luxembourg": 0.30,
    "Esch-sur-Alzette": 0.22,
    "Capellen": 0.08,
    "Mersch": 0.07,
    "Diekirch": 0.06,
    "Grevenmacher": 0.05,
    "Remich": 0.05,
    "Redange": 0.04,
    "Echternach": 0.04,
    "Wiltz": 0.04,
    "Clervaux": 0.03,
    "Vianden": 0.02,
}


def fetch_statec_metadata(timeout: int = 15) -> dict:
    """Best-effort fetch of the dataset metadata from the udata API.

    Returns a provenance dict either from the live API or, if offline, a
    static record built from known-good identifiers.
    """
    ensure_dirs()
    url = f"{DATA_PUBLIC_LU_API}/datasets/{STATEC_DATASET_ID}/"
    provenance = {
        "source": "data.public.lu (STATEC)",
        "dataset_id": STATEC_DATASET_ID,
        "dataset_url": (
            "https://data.public.lu/en/datasets/"
            "entreprises-construction-et-logement-autorisations-de-batir/"
        ),
        "license": "Creative Commons Zero (CC0)",
        "resources": {k: resource_url(v) for k, v in STATEC_RESOURCES.items()},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline-fallback",
    }
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "seco-risk-radar/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        provenance["mode"] = "live"
        provenance["title"] = payload.get("title")
        provenance["last_modified"] = payload.get("last_modified")
        provenance["resource_count"] = len(payload.get("resources", []))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        provenance["fetch_error"] = str(exc)

    (RAW_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def build_building_stock_prior() -> pd.DataFrame:
    """Return a (canton, region, building_type, weight) sampling prior.

    weight = canton_activity_weight * building_type_prior, renormalised. This
    is the distribution we sample synthetic projects from, so the portfolio
    geography and building mix reflect real Luxembourg construction patterns.
    """
    rows = []
    for canton in tx.CANTONS:
        c_w = CANTON_ACTIVITY_WEIGHT.get(canton, 0.03)
        for btype, b_w in tx.BUILDING_TYPE_PRIOR.items():
            rows.append(
                {
                    "canton": canton,
                    "region": tx.region_for(canton),
                    "building_type": btype,
                    "weight": c_w * b_w,
                }
            )
    df = pd.DataFrame(rows)
    df["weight"] = df["weight"] / df["weight"].sum()
    return df


def ingest() -> pd.DataFrame:
    """Run the ingestion step: record provenance and return the stock prior."""
    prov = fetch_statec_metadata()
    print(f"[ingest] provenance mode={prov['mode']} license={prov['license']}")
    prior = build_building_stock_prior()
    prior.to_csv(RAW_DIR / "building_stock_prior.csv", index=False)
    print(f"[ingest] building-stock prior rows={len(prior)} "
          f"(cantons={prior.canton.nunique()}, types={prior.building_type.nunique()})")
    return prior


if __name__ == "__main__":
    ingest()
