"""
Domain taxonomy for the Luxembourg construction sector.

Everything here is grounded in REAL, public reference data:

* The 12 Luxembourg cantons are the official administrative subdivisions.
* The building-type categories mirror the breakdown used by STATEC in the
  "Autorisations de bâtir" open dataset (residential single-dwelling,
  residential multi-dwelling, residential for communities, non-residential).

The region grouping is a *simplified* cartographic grouping of cantons. STATEC
publishes its own regional aggregates; we use a coarse, documented mapping so
the geography stays consistent across the pipeline. This is an illustrative
modelling choice, not an official STATEC region definition.
"""

from __future__ import annotations

# The 12 official cantons of Luxembourg.
CANTONS: list[str] = [
    "Capellen",
    "Clervaux",
    "Diekirch",
    "Echternach",
    "Esch-sur-Alzette",
    "Grevenmacher",
    "Luxembourg",
    "Mersch",
    "Redange",
    "Remich",
    "Vianden",
    "Wiltz",
]

# Coarse, documented canton -> region grouping (illustrative).
CANTON_TO_REGION: dict[str, str] = {
    "Luxembourg": "Centre",
    "Mersch": "Centre",
    "Capellen": "Ouest",
    "Redange": "Ouest",
    "Esch-sur-Alzette": "Sud",
    "Clervaux": "Nord",
    "Wiltz": "Nord",
    "Vianden": "Nord",
    "Diekirch": "Nord",
    "Echternach": "Est",
    "Grevenmacher": "Est",
    "Remich": "Est",
}

# Building types, aligned with the STATEC "type de bâtiment" breakdown.
BUILDING_TYPES: list[str] = [
    "Residential - single dwelling",      # maison d'habitation à un logement
    "Residential - multi dwelling",       # maison d'habitation à plusieurs logements
    "Residential - community",            # bâtiment résidentiel pour collectivité
    "Non-residential",                    # bâtiment non résidentiel (commerce, industrie, public)
]

# Approximate share of authorised buildings by type (orders of magnitude
# consistent with STATEC: single-dwelling dominates by count). Used only as a
# sampling prior when we have to fall back to fully synthetic generation.
BUILDING_TYPE_PRIOR: dict[str, float] = {
    "Residential - single dwelling": 0.55,
    "Residential - multi dwelling": 0.22,
    "Residential - community": 0.05,
    "Non-residential": 0.18,
}

# Engineered structural systems (not in the open data; realistic SECO-relevant
# categories that a technical-control body would record on a project).
STRUCTURAL_SYSTEMS: list[str] = [
    "Reinforced concrete frame",
    "Load-bearing masonry",
    "Steel frame",
    "Timber frame",
    "Mixed / hybrid",
]

WORKS_TYPES: list[str] = ["New build", "Extension", "Renovation"]

FOUNDATION_TYPES: list[str] = ["Shallow / strip", "Raft", "Piled / deep"]

SEASONS: list[str] = ["Winter", "Spring", "Summer", "Autumn"]

RISK_BANDS: list[str] = ["Low", "Medium", "High"]


def region_for(canton: str) -> str:
    """Return the region for a canton, defaulting to 'Centre' if unknown."""
    return CANTON_TO_REGION.get(canton, "Centre")
