# SECO's Early Risk Assessment Expert System (ERA)

A consistent, explainable, data-driven technical-risk assessment, produced at intake.

SECO's mission is to guarantee the safety and quality of construction. Today the technical risk of a project is assessed qualitatively, expert by expert, so judgements are inconsistent and high risk is often spotted late. ERA gives SECO's technical-control teams a consistent risk rating for every project at intake, a band (Low, Medium, High) plus a 0 to 100 risk index, with the reasons behind it shown in the expert's own vocabulary.

It is built to be trusted. Every rating is explainable and auditable, it uses only information available before work starts, and it is designed so that real SECO inspection data later replaces the demonstration label by retraining, not rewriting.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_run_pipeline.py   # ingest real STATEC priors, synthesize portfolio, write SQLite
python scripts/02_train.py          # benchmark two models, deploy the better one
python scripts/03_predict.py        # score every project, store predictions and drivers
python scripts/04_risk_map.py       # render the canton risk map
streamlit run app/app.py            # open the dashboard
```

Everything runs offline and is reproducible from a fixed seed. No API keys are required. An optional language-model briefing and an optional SHAP explainer activate only if their dependencies are present, and the product degrades cleanly to deterministic fallbacks when they are not.

The sections below answer the questions in the brief, then set out the results, the roadmap, the adoption approach, and how the value created should be tracked.

## 1. What problem is being solved, and for whom?

The user is SECO's technical-control team: the experts who assess projects and the people who route incoming work to them.

The problem is risk analysis, not scheduling. Today technical risk is judged manually and qualitatively, one project at a time. Three things follow:

- Assessments vary between experts, so two comparable projects can receive different judgements.
- There is no early, systematic signal, so a genuinely high-risk project is often recognised late, once a defect is already expensive to fix.
- The reasoning lives in individual heads, so it is hard to standardise, audit, or pass on.

What is missing is a consistent, data-driven risk assessment, available at intake before work starts, that encodes domain expertise, stays fully explainable and auditable, and gives experts a shared, objective baseline they can trust and build on. ERA is exactly that baseline: for every project it produces a risk band, a 0 to 100 risk index, and the drivers behind the score.

A consequence, not the use case: once every project carries a consistent risk rating, SECO can run its portfolio by risk and put its most experienced experts on the riskiest projects, with lighter-touch review on low-risk work. That operating-model gain follows from good risk analysis; it is a benefit of the use case, not the use case itself.

## 2. Why is this relevant to SECO?

SECO's business is independent technical control. It guarantees the safety and quality of construction, measured against regulatory standards, through expert review that intervenes upstream at design and downstream during execution and handover. That assurance underpins long-term liability for owners, developers, and insurers, so the product SECO actually sells is trustworthy judgement about where construction risk lies.

That mission turns into four standing challenges, and ERA is a direct lever on each:

- Allocate expertise: focus scarce expert capacity where risk is highest.
- Detect earlier: surface issues at design or early build, when remediation is cheap.
- Standardise judgement: keep assessments consistent and defensible across experts.
- Scale the knowledge: turn individual experience into an institutional capability.

The value compounds. Earlier and more precise detection lowers remediation and rework cost, consistency lowers reputational and liability exposure, and SECO's inspection history becomes a reusable asset rather than tacit knowledge held by individuals.

## 3. Which data sources were used, and why?

The real, public, openly licensed source is STATEC Autorisations de batir (building permits) on data.public.lu, published under Creative Commons Zero. This is genuine Luxembourg construction activity broken down by building type, canton, and period, which the brief's reproducibility requirement calls for.

The honest constraint, which shapes the whole design, is that this open dataset is aggregate statistics, not per-project records, and it carries no risk or defect outcome. The data that would carry a real risk label is SECO's confidential inspection history, which is not available for a take-home. The response is the hybrid approach the brief suggests, and the only honest one given the data:

1. Base features are sampled from the real distribution. The ingest step records provenance from the live data.public.lu API and builds a sampling prior over building type and canton, so the synthetic portfolio's geography and building mix match reality.
2. A documented risk label is attached from a transparent latent function that encodes construction-domain assumptions: renovations and load-bearing masonry are riskier than new reinforced-concrete builds, winter starts are riskier than summer, low contractor experience and high site complexity raise risk, and both very small and very large projects are riskier than mid-sized ones.

The label is deliberately not rigged to look easy. It mixes several drivers, adds genuine noise so the outcome is probabilistic, includes a non-linear size effect and a masonry-by-renovation interaction, and includes a deliberate red-herring feature, permit processing time, that carries no risk signal. A good model has to learn to ignore it. That it does (see Results) is part of the evidence the pipeline behaves correctly.

## 4. Technical decisions and trade-offs

| Decision | Reasoning | Trade-off accepted |
|---|---|---|
| SQLite for storage | Zero infrastructure, file-based, reproducible, easy to inspect. The schema is layered (provenance, then projects, then predictions) to make data lineage explicit. | Not concurrent and not a warehouse. Maps cleanly onto Postgres later without changing application logic. |
| Streamlit for the interface | Fastest path to a usable, data-rich assessment interface for one engineer on a short budget. The value is in the data, the model, and the explanations. | React is the right choice for a multi-user, role-aware production application. Streamlit is the deliberate prototype trade. |
| scikit-learn, two models benchmarked | A regularised logistic-regression baseline is benchmarked against a gradient-boosted model, and whichever wins on held-out macro F1 is deployed. Honest and data-driven rather than reaching for the fanciest model. | On this largely additive label the linear model wins and ships; the gradient-boosted model is retained and would be picked up automatically on SECO's richer real data. |
| Preprocessing inside the model pipeline | Encoding and scaling live inside the persisted artifact, which removes train-serve skew and lets raw project rows be scored directly. | Slightly less control than a separate feature store, which only matters at larger scale. |
| Explanations in the original feature space | Drivers are attributed in the inspector's vocabulary by ablation, with no extra dependency, so a rating can always be justified. | SHAP is wired in as an optional path that auto-selects the right explainer for the deployed model and has been verified to agree with ablation on the top drivers. |
| Language-model briefing is optional | The model only turns the structured drivers into prose; with no API key a deterministic template produces the same content. | Briefing prose is plainer without a key. The product is fully functional and reproducible offline. |

Built for trust, the three principles that run through the build:

- No leakage: only intake-time information is used, so the rating is ready when the decision is made.
- Honest data: a real building mix, a documented label, and a planted red-herring control the model must learn to ignore.
- Replaceable label: real inspection outcomes drop in as a retrain, not a rewrite.

## 5. What would ship to production tomorrow, and what would be replaced?

What is already the right shape and would ship:

- the layered pipeline with a provenance audit trail,
- the pipeline-wrapped model with automatic benchmarking and best-model selection,
- explanations in the inspector's vocabulary, with two methods that cross-validate each other,
- the assessment workflow of score, explain, and review.

What would be replaced:

- the demonstration label, replaced by SECO's real historical inspection outcomes. The system is built so this is a label-source swap and a retrain, not a rewrite. This is the single most important property of the design.
- the offline sampling prior, replaced by the parsed LUSTAT SDMX feed and, ideally, project-level records,
- SQLite, replaced by managed Postgres, and Streamlit, replaced by a React front end over a scoring API.

## 6. With three more months

The immediate horizon is to make the risk assessment real and trustworthy on SECO's own data: train on the inspection history, calibrate the probabilities, and set the high-risk recall threshold deliberately, since a missed high-risk project costs far more than a false alarm. The fuller path, including richer signals and adoption, is set out in the roadmap below.

## Results (held-out test set, n_train = 1125, n_test = 375, seed = 42)

| Model | Accuracy | Macro F1 | High-risk recall |
|---|---|---|---|
| Majority-class baseline | 0.55 | n/a | n/a |
| Logistic regression (deployed) | 0.73 | 0.67 | 0.60 |
| Gradient boosting | 0.68 | 0.61 | 0.49 |

The model separates risk well, recovers sensible drivers, and ignores the planted noise:

- The red-herring feature, permit processing time, ranks 13th of 14 by importance.
- Two independent explanation methods, ablation and SHAP, agree on the same top drivers.
- The top drivers match technical-control experience: type of works, age of the existing structure, structural system, site complexity, and contractor track record.

Honest read: these numbers validate the pipeline, the product, and the workflow, not real-world predictive accuracy, because the label is synthetic. Real validity requires SECO's labelled data, at which point the same code retrains and the same application works unchanged.

## Roadmap: ERA expansion and adoption

The build and the adoption journey advance together, so trust is earned before the tool influences decisions.

| Horizon | ERA expansion | Adoption and change management |
|---|---|---|
| Validate (0 to 3 months) | Train on real inspection history, calibrate, and set the high-risk recall threshold. | Run in shadow mode, brief the teams, and baseline today's assessment for comparison. |
| Enrich (3 to 6 months) | Add document and defect-image reading, drift monitoring, and an inspector feedback loop. | Move to assisted mode, train experts to read the drivers rather than just the score, and log every override. |
| Productionise (6 to 9 months) | Managed database, React front end, scoring API, role-based access. | Embed in the workflow, run the portfolio by risk, and assign the riskiest projects to the most experienced experts. |
| Sustain (continuous) | Model versioning, full audit trail, automation bias managed. | Track override rate, time to assess, high-risk caught early, and expert hours reallocated to high-risk work. |

## Adoption and change management

The model is the easy part. Whether ERA delivers value depends on whether experts adopt it, so the rollout treats trust, not accuracy, as the hard problem.

- Explanations are the adoption mechanism. A score with a reason the expert recognises gets used; a black box gets ignored. The drivers are shown in the expert's vocabulary precisely so the reasoning can be audited on each project.
- The tool supports, it does not decide. ERA ranks and explains; the expert decides and stays accountable. This is both the correct safety posture and what makes adoption possible, because it does not threaten professional authority or deskill the role.
- Overrides are signal, not failure. Every disagreement between ERA and an expert is logged and becomes training data, which improves the model and keeps the expert in control.
- Automation bias is the main risk to manage. The rating is never an automatic pass or fail; it surfaces uncertainty, keeps a human accountable, and keeps a full audit trail so any rating can be reconstructed and defended.

The operating-model consequence is the payoff: once risk is assessed consistently, SECO can manage the portfolio by risk and put its most experienced experts where risk is highest, with lighter-touch review on low-risk work.

## Tracking value creation

Value should be tracked from day one, so the business case is proven rather than assumed. A balanced set of indicators across four dimensions:

- Cost avoidance: defects caught earlier and more precisely, leading to less remediation, rework, and late-stage intervention.
- Productivity gains: faster and more consistent risk assessments, and expert hours redirected to high-value work.
- Adoption and activation: share of assessments using ERA, active users, and the override rate trending down as trust grows.
- Client and quality outcomes: higher client satisfaction, fewer escaped defects, and stronger, more defensible assurance.

## Architecture

```
data.public.lu (STATEC, CC0)
        |
        v
  ingest.py        record provenance, build a real building-mix sampling prior
        |
        v
  synthesize.py    hybrid portfolio: real prior plus a documented label
        |
        v
  SQLite           provenance -> projects -> predictions (explicit lineage)
        |
        v
  model.py         benchmark logistic regression vs gradient boosting, deploy the winner
        |
        v
  explain.py       global importance and per-project drivers (ablation, optional verified SHAP)
        |
        v
  briefing.py      assessment briefing (language model or deterministic template)
        |
        v
  app/app.py       dashboard: triage, single-building scoring, geographic risk, factors
  geo.py           per-canton risk aggregation and map (real boundaries, offline fallback)
```

## Repository layout

```
src/seco_risk_radar/
  config.py       paths, seed, real STATEC dataset and resource identifiers
  taxonomy.py     real Luxembourg cantons and regions, STATEC building types
  ingest.py       fetch provenance and build the real-data sampling prior (offline fallback)
  synthesize.py   hybrid portfolio: real prior plus a documented label
  database.py     SQLite schema and load/read helpers
  features.py     feature definitions and leakage-safe preprocessing
  model.py        train, benchmark, deploy, and score
  explain.py      global importance and local ablation attributions, plus optional SHAP
  briefing.py     assessment briefing with a deterministic template fallback
  geo.py          per-canton risk aggregation and map (real boundaries, offline fallback)
scripts/          01_run_pipeline, 02_train, 03_predict, 04_risk_map
app/app.py        dashboard (triage, single-building scoring, geographic risk, factors)
tests/            synthesize, features, database
```

## Tests

```bash
pip install pytest && pytest -q
```

## Data licence

STATEC Autorisations de batir is published under Creative Commons Zero (CC0). All synthetic data in this repository is generated locally and reproducibly.

The map uses canton boundaries from Cantons in Luxembourg 2024 (SIG-GR / GIS-GR, data.public.lu), licensed CC-BY 4.0. They are downloaded on first use and cached locally. If the portal is unreachable, the map falls back to a centroid bubble map, so nothing fails when offline.
