"""Geospatial risk view: aggregate predicted risk by Luxembourg canton and
render a map.

Design notes (defensible choices):

* **Real geometry, fetched at runtime.** Canton boundaries come from the
  official Luxembourg open-data portal ("Cantons in Luxembourg 2024",
  CC-BY 4.0). We do not vendor the 450 KB file into the repo; instead we
  download it once and cache it under ``data/geo/``. This mirrors the rest of
  the pipeline, which grounds itself in live STATEC data but degrades
  gracefully offline.
* **Graceful offline fallback.** If the boundaries cannot be fetched (no
  network, portal down), we fall back to a *bubble map* drawn at hard-coded
  canton centroids. The product still answers "where is the risk?" — just with
  circles instead of filled polygons. Nothing hard-fails.
* **One renderer, no heavy GIS stack.** We parse the GeoJSON with the standard
  library and draw with matplotlib only (no geopandas/plotly). The same
  ``build_risk_map`` function powers both the CLI script and the Streamlit tab,
  so there is a single code path to reason about.

Attribution for the boundary data (required by CC-BY 4.0):
    Cantons in Luxembourg 2024 — SIG-GR / GIS-GR 2024, data.public.lu,
    licensed CC-BY 4.0. Sources: GeoBasis-DE/BKG, IGN France, NGI-Belgium,
    ACT Luxembourg, Statbel, et al.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

import pandas as pd

from .config import DATA_DIR
from .taxonomy import CANTONS

# Official canton boundaries (CC-BY 4.0). Permalink is stable; it redirects to
# the timestamped resource on the download CDN.
CANTON_GEOJSON_URL = (
    "https://data.public.lu/en/datasets/r/e1b71a44-beb5-4dde-94f6-3b6920903a1d"
)
GEOJSON_ATTRIBUTION = (
    "Canton boundaries: Cantons in Luxembourg 2024, SIG-GR / GIS-GR, "
    "data.public.lu (CC-BY 4.0)."
)

GEO_DIR = DATA_DIR / "geo"
CANTON_GEOJSON_CACHE = GEO_DIR / "cantons_lux_2024.geojson"

# Approximate canton centroids (WGS84 lat, lon). Used to (a) place labels on the
# choropleth and (b) draw the offline bubble-map fallback. These are
# eyeballed centres, accurate enough for placement, not for analysis.
CANTON_CENTROIDS: dict[str, tuple[float, float]] = {
    "Capellen": (49.65, 5.99),
    "Clervaux": (50.05, 6.03),
    "Diekirch": (49.87, 6.16),
    "Echternach": (49.81, 6.42),
    "Esch-sur-Alzette": (49.50, 5.98),
    "Grevenmacher": (49.68, 6.44),
    "Luxembourg": (49.61, 6.13),
    "Mersch": (49.75, 6.10),
    "Redange": (49.76, 5.89),
    "Remich": (49.55, 6.36),
    "Vianden": (49.93, 6.21),
    "Wiltz": (49.97, 5.93),
}


# --------------------------------------------------------------------------- #
# Name handling
# --------------------------------------------------------------------------- #
def _normalize(name: str) -> str:
    """Casefold + strip accents + collapse separators, for robust matching of
    GeoJSON property values against our canton names (e.g. 'Esch-sur-Alzette'
    vs 'Esch-Sur-Alzette' vs 'Esch sur Alzette')."""
    if name is None:
        return ""
    txt = unicodedata.normalize("NFKD", str(name))
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    txt = txt.lower()
    for ch in (" ", "_", "-", "."):
        txt = txt.replace(ch, "")
    return txt.strip()


_CANTON_LOOKUP = {_normalize(c): c for c in CANTONS}

# Property keys commonly used to hold a canton/region name in LU/EU GeoJSON.
_NAME_KEY_CANDIDATES = (
    "CANTON", "canton", "Canton",
    "NAME", "name", "Name",
    "NAME_LATN", "NAME_2", "NUTS_NAME",
    "LIBELLE", "libelle", "LIBGEO",
    "nom", "NOM", "TITRE",
)


def _match_canton(properties: dict) -> Optional[str]:
    """Return the canonical canton name for a GeoJSON feature, or None if the
    feature is not one of Luxembourg's 12 cantons (the official file is
    Greater-Region harmonised and may include foreign units)."""
    # Try the known key names first.
    for key in _NAME_KEY_CANDIDATES:
        if key in properties:
            hit = _CANTON_LOOKUP.get(_normalize(properties[key]))
            if hit:
                return hit
    # Fall back to scanning every string property.
    for value in properties.values():
        if isinstance(value, str):
            hit = _CANTON_LOOKUP.get(_normalize(value))
            if hit:
                return hit
    return None


# --------------------------------------------------------------------------- #
# Boundary fetch / cache
# --------------------------------------------------------------------------- #
def fetch_canton_geojson(
    cache_path: Path = CANTON_GEOJSON_CACHE,
    *,
    force: bool = False,
    timeout: int = 30,
) -> Optional[dict]:
    """Load canton boundaries, using a local cache when possible.

    Returns the parsed GeoJSON ``dict`` or ``None`` if it could not be obtained
    (e.g. offline on first run). Never raises on network failure.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass  # corrupt cache -> try to refetch

    try:
        req = Request(CANTON_GEOJSON_URL, headers={"User-Agent": "seco-risk-radar/0.1"})
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted gov URL)
            raw = resp.read().decode("utf-8")
        geojson = json.loads(raw)
    except Exception:  # noqa: BLE001 — any failure -> graceful fallback
        return None

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(raw, encoding="utf-8")
    except OSError:
        pass  # caching is best-effort
    return geojson


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate_risk_by_canton(scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse a scored portfolio to one row per canton.

    Expects columns ``canton``, ``risk_score`` and ``pred_band``. Returns
    columns: canton, n_projects, n_high, high_share, mean_risk_score,
    plus the centroid lat/lon for plotting.
    """
    if "canton" not in scored or "risk_score" not in scored:
        raise KeyError("scored frame must contain 'canton' and 'risk_score'")

    g = scored.groupby("canton")
    out = pd.DataFrame(
        {
            "n_projects": g.size(),
            "mean_risk_score": g["risk_score"].mean(),
        }
    )
    if "pred_band" in scored:
        out["n_high"] = g["pred_band"].apply(lambda s: (s == "High").sum())
    else:
        out["n_high"] = 0
    out["high_share"] = out["n_high"] / out["n_projects"].clip(lower=1)
    out = out.reset_index()

    out["lat"] = out["canton"].map(lambda c: CANTON_CENTROIDS.get(c, (None, None))[0])
    out["lon"] = out["canton"].map(lambda c: CANTON_CENTROIDS.get(c, (None, None))[1])
    return out.sort_values("mean_risk_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Geometry helpers (pure-python GeoJSON polygon walk)
# --------------------------------------------------------------------------- #
def _iter_rings(geometry: dict):
    """Yield exterior rings (lists of [lon, lat]) from a Polygon/MultiPolygon."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        if coords:
            yield coords[0]  # exterior ring only (good enough for a choropleth)
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                yield poly[0]


def _color_for(value: float, vmin: float, vmax: float, cmap) -> tuple:
    if vmax <= vmin:
        norm = 0.5
    else:
        norm = (value - vmin) / (vmax - vmin)
    return cmap(0.15 + 0.8 * norm)  # avoid the very pale low end


# --------------------------------------------------------------------------- #
# Renderer (matplotlib only)
# --------------------------------------------------------------------------- #
def build_risk_map(
    by_canton: pd.DataFrame,
    geojson: Optional[dict] = None,
    *,
    title: str = "SECO Risk Radar — predicted technical risk by canton",
    ax=None,
):
    """Render the risk map and return a matplotlib Figure.

    If ``geojson`` is provided we draw a filled choropleth; otherwise we draw a
    bubble map at canton centroids. ``by_canton`` is the output of
    :func:`aggregate_risk_by_canton`.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    risk_by_name = dict(zip(by_canton["canton"], by_canton["mean_risk_score"]))
    vmin = float(by_canton["mean_risk_score"].min())
    vmax = float(by_canton["mean_risk_score"].max())
    cmap = plt.get_cmap("YlOrRd")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 8))
    else:
        fig = ax.figure

    mode = "bubble"
    drawn = set()
    if geojson and isinstance(geojson, dict):
        for feat in geojson.get("features", []):
            canton = _match_canton(feat.get("properties", {}) or {})
            if not canton or canton not in risk_by_name:
                continue
            value = risk_by_name[canton]
            color = _color_for(value, vmin, vmax, cmap)
            for ring in _iter_rings(feat.get("geometry", {}) or {}):
                xs = [pt[0] for pt in ring]
                ys = [pt[1] for pt in ring]
                ax.fill(xs, ys, facecolor=color, edgecolor="white", linewidth=0.8, zorder=2)
            drawn.add(canton)
        if drawn:
            mode = "choropleth"
            # Label each drawn canton at its centroid.
            for canton in drawn:
                lat, lon = CANTON_CENTROIDS.get(canton, (None, None))
                if lat is None:
                    continue
                ax.annotate(
                    canton,
                    (lon, lat),
                    fontsize=7,
                    ha="center",
                    va="center",
                    color="#1a1a1a",
                    zorder=3,
                )

    if mode == "bubble":
        # Fallback: circles sized by project count, coloured by mean risk.
        for _, row in by_canton.iterrows():
            if pd.isna(row["lat"]):
                continue
            color = _color_for(row["mean_risk_score"], vmin, vmax, cmap)
            size = 120 + 12 * float(row["n_projects"]) ** 0.5 * 8
            ax.scatter(
                row["lon"], row["lat"], s=size, color=color,
                edgecolor="#333333", linewidth=0.8, zorder=2,
            )
            ax.annotate(
                f"{row['canton']}\n{row['mean_risk_score']:.2f}",
                (row["lon"], row["lat"]),
                fontsize=7, ha="center", va="center", zorder=3,
            )
        ax.set_aspect(1.4)  # rough lat/lon aspect for Luxembourg's latitude

    if mode == "choropleth":
        ax.set_aspect("equal")

    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.margins(0.05)

    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean predicted risk score (0–1)")

    sub = "filled cantons" if mode == "choropleth" else "centroid bubbles (offline fallback)"
    fig.text(
        0.5, 0.015,
        f"{sub} · darker = higher risk · {GEOJSON_ATTRIBUTION}",
        ha="center", fontsize=6.5, color="#555555", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, mode
