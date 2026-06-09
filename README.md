# SECO Risk Radar 🏗️

**Predictive technical-risk scoring for construction projects.**
A mini "Building Intelligence" product that turns public Luxembourg construction
data into a ranked work-list, so a SECO inspection planner can send scarce
expert inspectors to the projects most likely to have defects — *first*.

> Score → explanation → focus areas → inspector briefing, for every project in a
> portfolio.

---

## TL;DR

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_run_pipeline.py   # ingest (STATEC) → synthesize portfolio → SQLite
python scripts/02_train.py          # train + benchmark models, persist the winner
python scripts/03_predict.py        # score portfolio + compute per-project focus areas
python scripts/04_risk_map.py       # render the canton risk map (PNG)
streamlit run app/app.py            # open the inspector triage dashboard
```

No API keys required. Everything runs offline and is reproducible (seeded).

---

## 1. What problem am I solving, and for whom?

**User: a SECO inspection planner / risk officer.**

SECO inspects buildings and infrastructure and assesses technical risk. But there
are always more projects than inspector-hours. Today that prioritisation is
**expert-driven and qualitative**: it depends on who reads the file, similar
projects can be treated differently, and a genuinely high-risk project can slip
down the queue and only get attention once a defect has already cost money.

**SECO Risk Radar** gives the planner a consistent, explainable first pass:

- every project gets a **risk band** (Low / Medium / High) and a continuous
  **priority score** (0–1) to sort the portfolio;
- every score comes with its **drivers / focus areas** (the *why*), and an
  optional natural-language **inspector briefing**;
- a **canton risk map** shows *where* risk concentrates, so capacity can be
  weighted by geography as well as by individual project.

It does **not** replace the expert — it routes expert attention. The KPI it moves
is exactly the one in the role brief: *number of high-risk issues caught early,
prioritisation consistency, and expert-time allocation.*

## 2. Why is this relevant to SECO?

- It is a direct embodiment of the AI & Data Engineer job ad: *construction risk,
  technical inspection, predictive intelligence.*
- It matches SECO's business model: SECO sells **independent risk assessment** to
  developers and insurers. A predictive layer makes that advice more systematic
  and scalable **without removing the human** — the model triages, the inspector
  decides.
- It is built around SECO's core asset — **accumulated technical data** — and
  shows the shape of a product that would sit on top of real inspection history.

## 3. Which data sources did I use, and why?

**Real, public, CC0:** STATEC *Autorisations de bâtir* (building permits) on
[data.public.lu](https://data.public.lu/en/datasets/entreprises-construction-et-logement-autorisations-de-batir/).
This is genuine Luxembourg construction activity by **building type × canton ×
period**, under a Creative Commons Zero licence — fully open and reproducible,
which the brief requires.

**The honest catch — and why it shapes the design:** this open dataset is
**aggregate statistics** (counts and useful surface per type/geography), *not*
per-project records, and it carries **no defect / risk outcome** (that data is
SECO's confidential inspection history, which I don't have).

So I used a **hybrid** approach (the one Raimondo explicitly suggested):

1. **Base features sampled from the real distribution** — `ingest.py` records
   provenance from the live data.public.lu API and builds a `(canton,
   building_type)` sampling prior so the synthetic portfolio's **geography and
   building mix are realistic** (Luxembourg-City and Esch dominate, single-dwelling
   houses dominate by count, etc.).
2. **A documented synthetic risk label** — `synthesize.py` attaches a `risk_severity`
   target generated from a transparent latent function encoding domain
   assumptions (renovations and load-bearing masonry are riskier than new
   reinforced-concrete builds, winter starts are riskier, low contractor
   experience is riskier, very small and very large projects are riskier…).

This is deliberately **not** a trick:

- the latent function mixes several drivers, with **genuine Gaussian noise**, a
  **non-linear (U-shaped) size effect**, and a **masonry × renovation interaction**;
- I injected a **red-herring feature** (`permit_processing_days`) that does *not*
  affect risk, so a good model must learn to ignore it.

> A second dataset, Kaggle's *Infrastructure Structural Defects*, inspired the
> feature schema (`Infrastructure_Type`, `Severity_Level`, …). I adapted that
> structure to project level rather than using it directly, to keep one coherent
> Luxembourg-grounded story.

## 4. Technical decisions and trade-offs

| Decision | Why | Trade-off I accepted |
|---|---|---|
| **SQLite** for storage | Zero-infra, file-based, fully reproducible, trivially inspectable. The schema is layered (`provenance → projects → predictions`) to show data lineage. | Not concurrent / not a warehouse. Maps cleanly onto Postgres + dbt later. |
| **Streamlit** for the UI | Fastest path to a *usable, data-rich* triage dashboard for one engineer on a short budget. The value is in the data + model + explanations. | React is SECO's stack and the right production choice for a multi-user, branded, role-aware app. Streamlit is the MVP trade. |
| **scikit-learn**, two models | I benchmark a **logistic-regression baseline** against a **gradient-boosted** model and *deploy whichever wins on held-out macro-F1* — honest and data-driven, not "fancy model by default". | On this (mostly additive) synthetic label the **regularised linear model wins**, so it ships. I keep HGB because on SECO's richer real data with stronger interactions I'd expect it to pull ahead. |
| **Preprocessing inside the Pipeline** | One-hot + scaling live in the persisted artifact → no train/serve skew; raw rows score directly. | Slightly less control than a hand-rolled feature store. |
| **Ablation/occlusion** for local explanations | Attributes risk in the **original, human-readable feature space** ("Renovation", "Load-bearing masonry") — an inspector's vocabulary — with no extra dependency. | SHAP gives game-theoretically "purer" values; it's wired in behind an optional flag (`explain.shap_local`). |
| **LLM briefing is optional** | The LLM only *translates* the structured drivers into prose. If no API key is set, a deterministic template produces the same content. | Briefing prose is plainer without a key — but the product is fully functional and reproducible offline. The LLM is never on the critical path. |

## 5. Production tomorrow vs. throw away

**Ship tomorrow (it's already the right shape):**
- the layered pipeline + provenance audit trail;
- the `Pipeline`-wrapped model with automatic baseline benchmarking and
  best-model selection;
- explainability in the inspector's own vocabulary;
- the triage dashboard pattern (rank → filter → drill-down → briefing).

**Throw away / replace:**
- the **synthetic risk label** — replace it with SECO's real historical inspection
  outcomes. The pipeline is built so this is a *label-source swap + retrain*, not a
  rewrite.
- the offline sampling prior — replace with the parsed **LUSTAT SDMX** feed and,
  ideally, project-level permit records;
- SQLite → managed Postgres; Streamlit → a React front-end on a scoring API.

## 6. If I had 3 more months

- **Real labels + calibration.** Train on SECO inspection history; calibrate
  probabilities (reliability curves) and tune the High-risk recall threshold,
  because in triage **missing a high-risk project costs more than a false alarm**.
- **Documents, not just tabular.** Add an OCR + LLM/RAG layer over inspection
  reports and permit PDFs to extract structured risk signals (the Kaggle
  `dacl10k` concrete-defect images are a natural computer-vision extension).
- **Monitoring.** Data-drift and performance dashboards, plus an inspector
  feedback loop (was the flag useful?) to retrain on.
- **Productionisation.** FastAPI scoring service + React UI + role-based access,
  SHAP explanations served alongside scores, and an experiment-tracking setup.

---

## Architecture

```
data.public.lu (STATEC, CC0)            ┌─────────────────────────────┐
        │  ingest.py (provenance+prior) │  Streamlit triage dashboard │
        ▼                               │  rank · filter · drill-down │
  synthesize.py  ──►  SQLite  ──►  model.py  ──►  explain.py  ──►  briefing.py
  (hybrid data)      projects/    (LR vs HGB,    (focus areas)   (LLM or template)
                     predictions   best wins)
```

## Results (held-out test set, n_test = 375, seed = 42)

| Model | Accuracy | Macro-F1 | High-risk recall |
|---|---|---|---|
| Majority baseline | 0.55 | — | — |
| Logistic Regression **(deployed)** | **0.731** | **0.669** | 0.603 |
| Gradient Boosting | 0.680 | 0.609 | 0.485 |

Sanity check: the red-herring feature `permit_processing_days` ranks **13th of 14**
by permutation importance — the model correctly learned to ignore it.

**On evaluation honesty:** these metrics validate the **pipeline, the product, and
the UX end-to-end**. They are *not* a claim of real-world predictive validity,
because the label is synthetic. Real validity requires SECO's labelled inspection
data — at which point the same code retrains and the same dashboard works.

## Repository layout

```
src/seco_risk_radar/
  config.py       paths, seed, real STATEC dataset/resource IDs
  taxonomy.py     real LU cantons/regions + STATEC building types
  ingest.py       fetch provenance + build real-data sampling prior (offline fallback)
  synthesize.py   hybrid portfolio: real prior + documented synthetic label
  database.py     SQLite schema + load/read helpers
  features.py     feature definitions + preprocessing (no leakage)
  model.py        train/benchmark/deploy + scoring
  explain.py      global importance + local ablation attributions (+ optional SHAP)
  briefing.py     LLM inspector briefing with template fallback
  geo.py          per-canton risk aggregation + map (real boundaries, offline fallback)
scripts/          01_run_pipeline · 02_train · 03_predict · 04_risk_map
app/app.py        Streamlit dashboard (Portfolio + Risk map tabs)
tests/            synthesize / features / database
```

## Tests

```bash
pip install pytest && pytest -q
```

## Data licence

STATEC *Autorisations de bâtir* is published under **Creative Commons Zero (CC0)**.
All synthetic data in this repo is generated locally and reproducibly.

The map view uses canton boundaries from *Cantons in Luxembourg 2024* (SIG-GR /
GIS-GR, [data.public.lu](https://data.public.lu/en/datasets/cantons-in-luxembourg-2024/)),
licensed **CC-BY 4.0**. They are downloaded on first use and cached under
`data/geo/`; if the portal is unreachable the map falls back to a centroid
bubble map, so nothing hard-fails offline.
