"""SECO ERA Expert System: inspection-planning decision support.

A professional dashboard for SECO technical-control planners. It does four
things, one per tab:

  1. Portfolio triage: rank incoming projects by predicted technical risk.
  2. Score a building: rate a single project from its attributes (live).
  3. Geographic risk: see where risk concentrates across the cantons.
  4. About the model: the factors considered, plus a short model card.

Design goals: readable, business oriented, and honest about what the model is
and is not. The drivers of every score are given their own section, following
explainability best practice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seco_risk_radar import database as db                 # noqa: E402
from seco_risk_radar import model as rrmodel               # noqa: E402
from seco_risk_radar import explain                        # noqa: E402
from seco_risk_radar import geo                            # noqa: E402
from seco_risk_radar import taxonomy as tax                # noqa: E402
from seco_risk_radar.briefing import generate_briefing     # noqa: E402
from seco_risk_radar.config import DB_PATH                 # noqa: E402

st.set_page_config(
    page_title="ERA Expert System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Brand + semantic colours
# --------------------------------------------------------------------------- #
SECO_NAVY = "#0E3A6E"
SECO_BLUE = "#2C6FB5"
BAND_COLOR = {"Low": "#2E8B57", "Medium": "#E08A1E", "High": "#C0392B"}

FRIENDLY = {
    "works_type": "Type of works",
    "structural_system": "Structural system",
    "existing_structure_age_yrs": "Age of existing structure",
    "contractor_experience": "Contractor track record",
    "site_complexity": "Site complexity",
    "building_type": "Building type",
    "foundation_type": "Foundation type",
    "gross_floor_area_m2": "Floor area",
    "num_floors": "Number of floors",
    "estimated_cost_eur": "Estimated cost",
    "season_started": "Construction season",
    "canton": "Location (canton)",
    "region": "Region",
    "permit_processing_days": "Permit processing time",
}

# Factors the model uses, grouped for the "About" tab.
FACTOR_GROUPS = [
    ("Building and use", [
        ("Building type", "Use and occupancy. Drives the consequence of a failure."),
        ("Type of works", "New build, extension or renovation. Renovations carry more unknowns."),
    ]),
    ("Structure and site", [
        ("Structural system", "Some systems, such as load-bearing masonry, are more defect sensitive."),
        ("Foundation type", "Suitability against the ground conditions."),
        ("Site complexity", "Geotechnical, sequencing and coordination difficulty."),
        ("Age of existing structure", "For renovations: degradation and hidden condition."),
    ]),
    ("Scale", [
        ("Floor area", "Larger or very small projects carry different risk profiles."),
        ("Number of floors", "Height adds structural and execution complexity."),
        ("Estimated cost", "A proxy for ambition and schedule or budget pressure."),
    ]),
    ("Delivery", [
        ("Contractor track record", "Experience is a strong driver of execution quality."),
        ("Construction season", "Weather and curing conditions during the works."),
    ]),
    ("Location", [
        ("Canton and region", "Captures geographic patterns in the building stock."),
    ]),
]


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 3.5rem; max-width: 1400px; }}
          header[data-testid="stHeader"] {{ background: transparent; }}
          .seco-bar {{
              border-left: 6px solid {SECO_NAVY};
              padding: 0.4rem 0 0.4rem 0.9rem; margin: 0.6rem 0 0.4rem 0;
          }}
          .seco-mark {{
              font-weight: 800; letter-spacing: 0.06em; color: {SECO_NAVY};
              font-size: 1.05rem;
          }}
          .seco-title {{ font-weight: 800; color: {SECO_NAVY}; margin: 0.1rem 0 0.4rem 0; }}
          .era-hero {{
              background: linear-gradient(90deg, #0E3A6E 0%, #2C6FB5 100%);
              color: #ffffff; padding: 1.15rem 1.35rem; border-radius: 10px;
              margin: 0.3rem 0 0.7rem 0;
          }}
          .era-hero h2 {{ color: #ffffff !important; margin: 0 0 0.55rem 0; font-size: 2.15rem; font-weight: 800; line-height: 1.15; }}
          .era-hero p {{ margin: 0; font-size: 0.96rem; line-height: 1.55; color: #eaf1fb; }}
          .pill {{
              display:inline-block; padding: 0.18rem 0.7rem; border-radius: 999px;
              color: white; font-weight: 700; font-size: 0.85rem;
          }}
          .stTabs [data-baseweb="tab-list"] {{ gap: 1.5rem; }}
          .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
          h2, h3 {{ color: {SECO_NAVY}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def pill(band: str) -> str:
    return f'<span class="pill" style="background:{BAND_COLOR.get(band, SECO_NAVY)}">{band} risk</span>'


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_portfolio() -> pd.DataFrame:
    conn = db.connect(DB_PATH)
    try:
        return db.read_scored_portfolio(conn)
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def load_model():
    return rrmodel.load_model()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    p = Path(rrmodel.MODEL_PATH).parent / "metrics.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


@st.cache_data(show_spinner=False)
def load_canton_geojson():
    return geo.fetch_canton_geojson()


# --------------------------------------------------------------------------- #
# Score presentation
# --------------------------------------------------------------------------- #
PRIORITY_EXPLAINER = (
    "The risk index shows how strongly a project is predicted to be at technical risk, on a scale of "
    "0 to 100. It combines the model's predicted probability that the project is High risk (counted in "
    "full) with its probability of being Medium risk (counted at half weight). An index of 99 means the "
    "model predicts the project is almost certainly in the High-risk band. It reflects predicted risk, "
    "not a confirmed probability that the building will fail."
)


def present_priority(score: float, band: str, portfolio: pd.DataFrame) -> None:
    """Business-friendly score block: 0 to 100 index, band, and portfolio percentile."""
    idx = int(round(float(score) * 100))
    st.markdown(pill(band), unsafe_allow_html=True)
    st.metric("Risk index", f"{idx} / 100")
    st.progress(min(max(float(score), 0.0), 1.0))
    if portfolio is not None and not portfolio.empty:
        pct = float((portfolio["risk_score"] < score).mean()) * 100
        st.caption(f"Predicted to be more at risk than {pct:.0f}% of projects currently in scope.")
    with st.expander("What does this index mean?"):
        st.write(PRIORITY_EXPLAINER)


# --------------------------------------------------------------------------- #
# Value formatting + explanation rendering
# --------------------------------------------------------------------------- #
def fmt_value(feature: str, value) -> str:
    try:
        if feature == "existing_structure_age_yrs":
            return f"{int(round(float(value)))} yrs"
        if feature == "gross_floor_area_m2":
            return f"{int(round(float(value))):,} m2"
        if feature == "estimated_cost_eur":
            return f"EUR {int(round(float(value))):,}"
        if feature == "num_floors":
            return f"{int(round(float(value)))}"
        if feature == "permit_processing_days":
            return f"{int(round(float(value)))} days"
        if feature == "contractor_experience":
            v = float(value)
            return "extensive" if v > 0.8 else "strong" if v > 0.6 else "average" if v > 0.4 else "limited"
        if feature == "site_complexity":
            v = float(value)
            return "very high" if v > 0.8 else "high" if v > 0.6 else "moderate" if v > 0.4 else "low"
    except (TypeError, ValueError):
        pass
    return str(value)


def drivers_figure(factors: list[dict]):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    pos = [f for f in factors if f.get("contribution", 0) > 0][:5]
    if not pos:
        return None
    labels = [FRIENDLY.get(f["feature"], f["feature"]) for f in pos][::-1]
    vals = [float(f["contribution"]) for f in pos][::-1]

    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(labels) + 0.7))
    ax.barh(labels, vals, color=SECO_BLUE, edgecolor=SECO_NAVY, linewidth=0.6)
    ax.set_xlabel("Contribution to the risk score", fontsize=9)
    ax.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    fig.tight_layout()
    return fig


def render_drivers(factors: list[dict]) -> None:
    """The 'What's driving this score' section, used wherever a score appears."""
    st.markdown("##### What's driving this score")
    pos = [f for f in factors if f.get("contribution", 0) > 0][:5]
    if not pos:
        st.caption("No single factor stands out. The score reflects a broad, balanced profile.")
        return
    lead = ", ".join(
        f"**{FRIENDLY.get(f['feature'], f['feature']).lower()}** ({fmt_value(f['feature'], f['value'])})"
        for f in pos[:3]
    )
    st.markdown(
        f"The main reasons for this score are {lead}. "
        "Bars show how much each factor lifts the score above a neutral baseline."
    )
    fig = drivers_figure(factors)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    st.caption(
        "Attributions are computed by ablation against a neutral reference project, "
        "so they reflect this project, not just global averages."
    )


# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
def header() -> None:
    st.markdown(
        '<div class="seco-bar"><span class="seco-mark">SECO</span> '
        '<span style="color:#6b778c">| technical control</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="era-hero">'
        "<h2>Welcome to ERA, your Early Risk Assessment Expert System</h2>"
        "<p>Built for SECO's technical-control teams, ERA rates a project's technical risk before work "
        "begins and turns a stream of incoming projects into a clear, ranked plan: which buildings need "
        "expert risk mitigation, and why.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def kpi_strip(df: pd.DataFrame) -> None:
    n = len(df)
    n_high = int((df["pred_band"] == "High").sum())
    n_med = int((df["pred_band"] == "Medium").sum())
    avg = float(df["risk_score"].mean()) if n else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects in scope", f"{n:,}", help="Projects matching the current filters.")
    c2.metric("High-risk projects", f"{n_high:,}", help="Projects predicted to be most at risk.")
    c2.caption(f"{(n_high / n * 100):.0f}% of projects in scope" if n else "")
    c3.metric("Medium-risk projects", f"{n_med:,}",
              help="Projects with moderate predicted risk.")
    c3.caption(f"{(n_med / n * 100):.0f}% of projects in scope" if n else "")
    c4.metric("Portfolio average risk index", f"{int(round(avg * 100))} / 100",
              help="Average risk index across the projects currently in scope (0 to 100).")


# --------------------------------------------------------------------------- #
# Tab 1: Portfolio triage
# --------------------------------------------------------------------------- #
def portfolio_tab(view: pd.DataFrame, pipe, portfolio: pd.DataFrame) -> None:
    st.subheader("Risk-ranked portfolio")
    st.caption("Sorted by predicted risk, highest first.")

    table = view[
        ["project_id", "pred_band", "canton",
         "building_type", "works_type", "structural_system"]
    ].rename(columns={
        "project_id": "Project", "pred_band": "Risk band",
        "canton": "Canton", "building_type": "Building type",
        "works_type": "Works", "structural_system": "Structure",
    })
    st.dataframe(table, use_container_width=True, hide_index=True, height=380)

    st.divider()
    st.subheader("Project detail")
    if view.empty:
        st.info("No projects match the current filters.")
        return

    pid = st.selectbox("Select a project to review", view["project_id"].tolist())
    proj = view[view["project_id"] == pid].iloc[0]
    band = proj["pred_band"]

    with st.container(border=True):
        top = st.columns([2, 1])
        with top[0]:
            st.markdown(f"### {pid}")
            st.markdown(
                f"{proj['building_type']} · {proj['works_type']} · {proj['structural_system']}  \n"
                f"{proj['canton']} ({proj['region']}) · {int(proj['gross_floor_area_m2']):,} m2 · "
                f"{int(proj['num_floors'])} floors"
            )
        with top[1]:
            present_priority(float(proj["risk_score"]), band, portfolio)

        st.divider()
        cols = st.columns(2)
        with cols[0]:
            factors = json.loads(proj["top_factors"]) if proj["top_factors"] else []
            render_drivers(factors)
        with cols[1]:
            st.markdown("##### Recommended inspection focus")
            st.caption("A short, plain-language briefing the planner can hand to the assigned inspector.")
            if st.button("Generate briefing", key="brief_portfolio"):
                with st.spinner("Preparing briefing..."):
                    try:
                        factors = json.loads(proj["top_factors"]) if proj["top_factors"] else []
                        text, _ = generate_briefing(
                            proj.to_dict(), band, float(proj["risk_score"]), factors
                        )
                        st.markdown(text)
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Could not generate a briefing: {exc}")
            else:
                st.info("Click to draft an inspector briefing for this project.")


# --------------------------------------------------------------------------- #
# Tab 2: Score a building (live)
# --------------------------------------------------------------------------- #
EXPERIENCE_SCALE = {
    "Very limited": 0.15, "Limited": 0.35, "Average": 0.55, "Strong": 0.75, "Extensive": 0.95,
}
COMPLEXITY_SCALE = {
    "Simple": 0.15, "Moderate": 0.4, "Complex": 0.65, "Very complex": 0.9,
}


def score_tab(pipe, portfolio: pd.DataFrame) -> None:
    st.subheader("Score a building")
    st.caption(
        "Enter a project's attributes to get an instant risk rating and the factors behind it. "
        "Useful for a new permit application before it enters the portfolio."
    )

    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            building_type = st.selectbox("Building type", tax.BUILDING_TYPES)
            works_type = st.selectbox("Type of works", tax.WORKS_TYPES)
            canton = st.selectbox("Canton", tax.CANTONS, index=tax.CANTONS.index("Luxembourg"))
        with c2:
            structural_system = st.selectbox("Structural system", tax.STRUCTURAL_SYSTEMS)
            foundation_type = st.selectbox("Foundation type", tax.FOUNDATION_TYPES)
            season = st.selectbox("Construction season", tax.SEASONS)
        with c3:
            gfa = st.number_input("Gross floor area (m2)", min_value=20, max_value=50000, value=400, step=10)
            floors = st.number_input("Number of floors", min_value=1, max_value=60, value=3, step=1)
            cost = st.number_input("Estimated cost (EUR)", min_value=10000, max_value=100_000_000,
                                   value=800_000, step=50_000)

        c4, c5, c6 = st.columns(3)
        with c4:
            is_existing = works_type in ("Renovation", "Extension")
            age = st.number_input(
                "Age of existing structure (yrs)", min_value=0, max_value=200,
                value=40 if is_existing else 0, step=1,
                help="For renovations and extensions. New builds are 0.",
                disabled=not is_existing,
            )
        with c5:
            exp_label = st.select_slider("Contractor track record",
                                         options=list(EXPERIENCE_SCALE), value="Average")
        with c6:
            cx_label = st.select_slider("Site complexity",
                                        options=list(COMPLEXITY_SCALE), value="Moderate")

        with st.expander("Advanced (administrative)"):
            permit_days = st.number_input(
                "Permit processing time (days)", min_value=10, max_value=400, value=120, step=5,
                help="Included for completeness. The model treats this as non-predictive, "
                     "it carries no real technical-risk signal.",
            )

        submitted = st.form_submit_button("Rate this building", type="primary")

    if not submitted:
        return

    row = {
        "project_id": "NEW-PROJECT",
        "canton": canton,
        "region": tax.CANTON_TO_REGION.get(canton, "Centre"),
        "building_type": building_type,
        "works_type": works_type,
        "structural_system": structural_system,
        "foundation_type": foundation_type,
        "season_started": season,
        "gross_floor_area_m2": float(gfa),
        "num_floors": int(floors),
        "existing_structure_age_yrs": float(age if is_existing else 0),
        "contractor_experience": EXPERIENCE_SCALE[exp_label],
        "site_complexity": COMPLEXITY_SCALE[cx_label],
        "estimated_cost_eur": float(cost),
        "permit_processing_days": float(permit_days),
    }
    df1 = pd.DataFrame([row])
    pred = rrmodel.score_portfolio(pipe, df1).iloc[0]
    band = pred["pred_band"]
    score = float(pred["risk_score"])

    st.divider()
    with st.container(border=True):
        head = st.columns([2, 1])
        with head[0]:
            st.markdown("### Rating")
            st.markdown(f"{building_type} · {works_type} · {structural_system} · {canton}")
            pl, pm, ph = float(pred["prob_low"]), float(pred["prob_medium"]), float(pred["prob_high"])
            pcols = st.columns(3)
            pcols[0].metric("P(Low)", f"{pl:.0%}")
            pcols[1].metric("P(Medium)", f"{pm:.0%}")
            pcols[2].metric("P(High)", f"{ph:.0%}")
        with head[1]:
            present_priority(score, band, portfolio)

        st.divider()
        try:
            factors = explain.local_factors(pipe, df1.iloc[0], reference=portfolio)
        except Exception:  # noqa: BLE001
            factors = []
        render_drivers(factors)


# --------------------------------------------------------------------------- #
# Tab 3: Geographic risk
# --------------------------------------------------------------------------- #
def map_tab(view: pd.DataFrame) -> None:
    st.subheader("Where is the risk concentrated?")
    st.caption(
        "Cantons coloured by the mean predicted risk of their projects, showing how predicted risk "
        "is distributed across the territory."
    )
    if view.empty:
        st.info("No projects match the current filters.")
        return

    use_real = st.checkbox(
        "Use official canton boundaries (downloads once, then cached)", value=True,
        help="Off uses a centroid bubble map with no download.",
    )
    geojson = load_canton_geojson() if use_real else None
    if use_real and geojson is None:
        st.info("Boundaries unavailable (offline?). Showing a centroid bubble map instead.")

    by_canton = geo.aggregate_risk_by_canton(view)
    fig, mode = geo.build_risk_map(by_canton, geojson=geojson)

    mcol, tcol = st.columns([3, 2])
    with mcol:
        st.pyplot(fig, use_container_width=True)
    with tcol:
        st.markdown("**Canton risk ranking**")
        tbl = by_canton[["canton", "n_projects", "n_high", "mean_risk_score"]].copy()
        tbl["mean_risk_score"] = (tbl["mean_risk_score"] * 100).round().astype(int)
        tbl = tbl.rename(columns={"canton": "Canton", "n_projects": "Projects",
                                  "n_high": "High", "mean_risk_score": "Mean index"})
        st.dataframe(
            tbl, use_container_width=True, hide_index=True, height=420,
            column_config={
                "Mean index": st.column_config.ProgressColumn(
                    "Mean index", min_value=0,
                    max_value=int((by_canton["mean_risk_score"].max() * 100).round()), format="%d",
                )
            },
        )
    st.caption(geo.GEOJSON_ATTRIBUTION if mode == "choropleth" else "Bubble map (no boundary file).")


# --------------------------------------------------------------------------- #
# Tab 4: About the model
# --------------------------------------------------------------------------- #
def about_tab() -> None:
    st.subheader("Factors considered in the risk assessment")
    st.markdown("The model rates each project as Low, Medium or High risk using the factors below:")
    for group, items in FACTOR_GROUPS:
        st.markdown(f"**{group}**")
        st.markdown("\n".join(f"- **{name}.** {why}" for name, why in items))

    st.info(
        "Permit processing time is recorded for completeness but is not used to judge technical risk. "
        "How long the administrative paperwork takes says nothing about how a building is designed or built."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()
    header()

    try:
        df = load_portfolio()
    except Exception:  # noqa: BLE001
        df = pd.DataFrame()
    if df.empty:
        st.error("No scored portfolio found. Run the pipeline first: "
                 "scripts/01_run_pipeline.py, 02_train.py, 03_predict.py.")
        st.stop()

    try:
        pipe = load_model()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the trained model (run scripts/02_train.py). Details: {exc}")
        st.stop()

    st.sidebar.header("Filters")
    st.sidebar.caption("Narrow the portfolio. Filters apply to the triage and map views.")
    bands = st.sidebar.multiselect("Risk band", tax.RISK_BANDS, default=tax.RISK_BANDS)
    cantons = st.sidebar.multiselect("Canton", sorted(df["canton"].unique()))
    btypes = st.sidebar.multiselect("Building type", sorted(df["building_type"].unique()))

    view = df[df["pred_band"].isin(bands)] if bands else df
    if cantons:
        view = view[view["canton"].isin(cantons)]
    if btypes:
        view = view[view["building_type"].isin(btypes)]

    kpi_strip(view)
    st.write("")

    t1, t2, t3, t4 = st.tabs(
        ["📋 Portfolio triage", "🏗️ Score a building", "🗺️ Geographic risk", "ℹ️ About the model"]
    )
    with t1:
        portfolio_tab(view, pipe, df)
    with t2:
        score_tab(pipe, df)
    with t3:
        map_tab(view)
    with t4:
        about_tab()

    st.divider()
    st.caption(
        "Data: STATEC Autorisations de batir (data.public.lu, CC0). The risk label is synthetic and "
        "documented. Scores reflect predicted risk, they are not a verdict on any project."
    )


if __name__ == "__main__":
    main()
