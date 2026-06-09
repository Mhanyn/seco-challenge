#!/usr/bin/env python3
"""
SECO Risk Radar - inspector triage dashboard.

Run: streamlit run app/app.py

The UI is intentionally focused on ONE job: help a SECO inspection planner
decide where to send inspectors first. It shows portfolio-level KPIs, a risk-
ranked table with filters, and a per-project drill-down with the drivers
(focus areas) and a natural-language briefing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seco_risk_radar import database as db          # noqa: E402
from seco_risk_radar import geo                      # noqa: E402
from seco_risk_radar.briefing import generate_briefing  # noqa: E402

BAND_COLOR = {"High": "#c0392b", "Medium": "#e67e22", "Low": "#27ae60"}

st.set_page_config(page_title="SECO Risk Radar", page_icon="🏗️", layout="wide")


@st.cache_data
def load_portfolio() -> pd.DataFrame:
    conn = db.connect()
    try:
        df = db.read_scored_portfolio(conn)
    finally:
        conn.close()
    return df


def kpi_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects in portfolio", len(df))
    c2.metric("High-risk projects", int((df["pred_band"] == "High").sum()))
    c3.metric("Avg. priority score", f"{df['risk_score'].mean():.2f}")
    flagged = df[df["pred_band"] == "High"]["risk_score"].sum()
    c4.metric("High-risk priority mass", f"{flagged:.1f}")


@st.cache_data(show_spinner=False)
def load_canton_geojson(allow_fetch: bool):
    """Cached boundary loader. Returns parsed GeoJSON or None (offline)."""
    if not allow_fetch:
        # Only use a boundary file if it's already cached locally.
        from seco_risk_radar.geo import CANTON_GEOJSON_CACHE
        import json as _json
        if CANTON_GEOJSON_CACHE.exists():
            try:
                return _json.loads(CANTON_GEOJSON_CACHE.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None
        return None
    return geo.fetch_canton_geojson()


def render_risk_map(view: pd.DataFrame) -> None:
    st.subheader("Where is the risk concentrated?")
    st.caption(
        "Cantons coloured by the mean predicted priority score of their "
        "projects — a portfolio view of where to weight inspection capacity."
    )
    if view.empty:
        st.info("No projects match the current filters.")
        return

    allow_fetch = st.checkbox(
        "Use official canton boundaries (downloads ~450 KB once, then cached)",
        value=True,
        help="Off = draw a centroid bubble map without any download.",
    )
    geojson = load_canton_geojson(allow_fetch)
    if allow_fetch and geojson is None:
        st.info("Boundaries unavailable (offline?). Showing a centroid bubble map instead.")

    by_canton = geo.aggregate_risk_by_canton(view)
    fig, mode = geo.build_risk_map(by_canton, geojson=geojson)

    mcol, tcol = st.columns([3, 2])
    with mcol:
        st.pyplot(fig, use_container_width=True)
    with tcol:
        st.markdown("**Canton risk ranking**")
        tbl = by_canton[["canton", "n_projects", "n_high", "mean_risk_score"]].rename(
            columns={
                "canton": "Canton", "n_projects": "Projects",
                "n_high": "High", "mean_risk_score": "Mean score",
            }
        )
        st.dataframe(
            tbl,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Mean score": st.column_config.ProgressColumn(
                    "Mean score", min_value=0.0,
                    max_value=float(by_canton["mean_risk_score"].max()),
                    format="%.2f",
                )
            },
            height=460,
        )
    st.caption(geo.GEOJSON_ATTRIBUTION if mode == "choropleth" else "Bubble map (no boundary file).")


def main() -> None:
    st.title("🏗️ SECO Risk Radar")
    st.caption(
        "Predictive technical-risk scoring for construction projects. "
        "Ranks a portfolio so scarce inspector time goes where defect risk is highest."
    )

    df = load_portfolio()
    if df.empty or df["risk_score"].isna().all():
        st.warning(
            "No scored projects found. Run the pipeline first:\n\n"
            "```\npython scripts/01_run_pipeline.py\n"
            "python scripts/02_train.py\npython scripts/03_predict.py\n```"
        )
        return

    # ---- Sidebar filters --------------------------------------------------
    st.sidebar.header("Filters")
    bands = st.sidebar.multiselect(
        "Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
    )
    cantons = st.sidebar.multiselect(
        "Canton", sorted(df["canton"].dropna().unique())
    )
    btypes = st.sidebar.multiselect(
        "Building type", sorted(df["building_type"].dropna().unique())
    )

    view = df[df["pred_band"].isin(bands)] if bands else df
    if cantons:
        view = view[view["canton"].isin(cantons)]
    if btypes:
        view = view[view["building_type"].isin(btypes)]

    tab_portfolio, tab_map = st.tabs(["📋 Portfolio", "🗺️ Risk map"])

    with tab_map:
        render_risk_map(view)

    with tab_portfolio:
        kpi_row(view)
        st.divider()

        left, right = st.columns([3, 2])

        # ---- Risk-ranked table ------------------------------------------------
        with left:
            st.subheader("Risk-ranked portfolio")
            table = view[
                ["project_id", "pred_band", "risk_score", "canton",
                 "building_type", "works_type", "structural_system"]
            ].rename(
                columns={
                    "project_id": "Project", "pred_band": "Band",
                    "risk_score": "Priority", "canton": "Canton",
                    "building_type": "Type", "works_type": "Works",
                    "structural_system": "Structure",
                }
            )
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Priority": st.column_config.ProgressColumn(
                        "Priority", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
                height=520,
            )

        # ---- Per-project drill-down ------------------------------------------
        with right:
            st.subheader("Project briefing")
            if view.empty:
                st.info("No projects match the current filters.")
            else:
                pid = st.selectbox("Select a project", view["project_id"].tolist())
                proj = view[view["project_id"] == pid].iloc[0]

                band = proj["pred_band"]
                color = BAND_COLOR.get(band, "#555")
                st.markdown(
                    f"### {pid} &nbsp; "
                    f"<span style='color:{color};font-weight:700'>{band} risk</span>",
                    unsafe_allow_html=True,
                )
                st.progress(float(proj["risk_score"]), text=f"Priority {proj['risk_score']:.2f}")

                meta = (
                    f"{proj['building_type']} · {proj['works_type']} · {proj['structural_system']} · "
                    f"{proj['canton']} ({proj['region']}) · {int(proj['gross_floor_area_m2']):,} m²"
                )
                st.caption(meta)

                factors = json.loads(proj["top_factors"]) if proj["top_factors"] else []
                if factors:
                    st.markdown("**Focus areas (what drives this score):**")
                    fdf = pd.DataFrame(factors)
                    fdf["contribution"] = fdf["contribution"].astype(float)
                    st.bar_chart(fdf.set_index("feature")["contribution"])

                if st.button("Generate inspector briefing"):
                    with st.spinner("Writing briefing…"):
                        text, source = generate_briefing(
                            proj.to_dict(), band, float(proj["risk_score"]), factors
                        )
                    st.markdown(text)
                    st.caption(f"Briefing source: {source}")

    st.divider()
    st.caption(
        "Data: STATEC *Autorisations de bâtir* (data.public.lu, CC0) for the "
        "building mix; risk labels are synthetic (documented in the README). "
        "Scores validate the pipeline & UX, not real-world predictive validity."
    )


if __name__ == "__main__":
    main()
