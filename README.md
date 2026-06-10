# ERA Expert System

Early Risk Assessment Expert System, built for SECO technical control.

ERA turns public Luxembourg construction data into a decision-support layer for
inspection planning. For every project in a portfolio it produces a predicted
technical-risk band (Low, Medium, High), a continuous risk index from 0 to 100,
and the specific factors driving that score. The point is to let SECO manage an
inspection portfolio by predicted risk instead of by intuition or order of
arrival, and to allocate scarce expert capacity, including the seniority of the
assigned inspector, to where risk is highest.

The model triages. The expert decides. ERA exists to make that triage
consistent, explainable, and defensible.

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

Everything runs offline and is reproducible from a fixed seed. No API keys are
required. An optional language-model briefing and an optional SHAP explainer
activate only if their dependencies are present, and the product degrades
cleanly to deterministic fallbacks when they are not.

The remaining sections answer the questions in the brief, then go beyond them to
the part that actually decides whether a tool like this succeeds: the operating
model, change management, and adoption.

## 1. What problem is being solved, and for whom?

The user is a SECO inspection planner or technical-control coordinator.

The structural problem is a supply-and-demand mismatch. There are always more
projects to inspect than there are inspector-hours, and the scarcest resource of
all is senior inspector time. Today the decision of where that time goes is
largely qualitative. It depends on who reads the file, similar projects can be
handled differently by different people, and a genuinely high-risk project can
sit in the queue until a defect has already become expensive.

The correct way to frame this is as a portfolio allocation problem. On one side
there is a portfolio of incoming projects with heterogeneous and unobserved
risk. On the other side there is a finite, heterogeneous supply of inspection
capacity that ranges from junior to highly experienced. The allocation that
maximises caught defects per inspector-hour is straightforward in principle:
rank the portfolio by predicted risk, direct the most experienced inspectors and
the most intensive control to the riskiest projects, and route low-risk work to
lighter-touch or desk review. The reason this is not done systematically today
is that the ranking does not exist in a consistent, defensible form. ERA
produces exactly that ranking, plus the rationale that makes the resulting
assignment auditable.

Concretely, for each project ERA outputs:

- a risk band (Low, Medium, High) and a 0 to 100 risk index for sorting the
  portfolio,
- the factors driving the score, in the inspector's own vocabulary
  (for example "Renovation", "Load-bearing masonry", "low contractor track
  record"),
- an optional plain-language briefing for the assigned inspector,
- a geographic view showing how predicted risk is distributed across the
  cantons.

The metric ERA is built to move is the one in the role brief: high-risk issues
caught early, consistency of prioritisation across planners, and efficient
allocation of expert time.

## 2. Why is this relevant to SECO?

SECO's business is independent technical control. It sells assurance about
construction risk to developers, owners, and insurers, and that assurance is
tied to liability over the long defect-liability period. The product SECO
actually sells is judgement about where things go wrong.

A predictive layer is valuable to that business for four reasons that compound:

1. Allocation. The marginal value of a senior inspector is far higher on a
   high-risk project than on a routine one. Any tool that reliably sorts the
   portfolio by risk lets SECO put its most expensive expertise where it pays
   off, which is a direct efficiency gain on the firm's scarcest input.
2. Early detection. A defect identified at design or early construction is far
   cheaper to remedy than one found after handover, and far cheaper than one
   that surfaces during the liability period. Moving detection earlier is money.
3. Consistency and defensibility. A documented, reproducible risk rating reduces
   the variance in how similar projects are treated, and gives SECO a clear,
   auditable basis for its decisions if a rating is ever challenged by a client,
   an insurer, or in a dispute.
4. Data as an asset. SECO's accumulated inspection history is a moat. ERA shows
   the shape of a product that sits on top of that history and turns it into a
   repeatable, scalable capability rather than knowledge locked in individual
   inspectors' heads.

None of this removes the human. It makes the human's judgement systematic.

## 3. Which data sources were used, and why?

The real, public, openly licensed source is STATEC Autorisations de batir
(building permits) on data.public.lu, published under Creative Commons Zero.
This is genuine Luxembourg construction activity broken down by building type,
canton, and period. It is fully open and reproducible, which the brief requires.

The honest constraint, which shapes the entire design, is that this open dataset
is aggregate statistics, not per-project records, and it carries no defect or
risk outcome. The data that would carry a real risk label is SECO's confidential
inspection history, which is not available for a take-home.

The response is a hybrid approach, which is the one suggested in the brief, and
which I consider the only honest option given the data:

1. Base features are sampled from the real distribution. The ingest step records
   provenance from the live data.public.lu API and builds a sampling prior over
   building type and canton, so that the synthetic portfolio's geography and
   building mix match reality. Luxembourg City and Esch dominate by volume,
   single-dwelling housing dominates by count, and so on.
2. A documented synthetic risk label is attached. The label is generated from a
   transparent latent function that encodes construction-domain assumptions:
   renovations and load-bearing masonry are riskier than new reinforced-concrete
   builds, winter starts are riskier than summer, low contractor experience and
   high site complexity raise risk, and both very small and very large projects
   are riskier than mid-sized ones.

This is deliberately not a model that is rigged to look good. The latent function
mixes several drivers, adds genuine Gaussian noise so the label is probabilistic
rather than deterministic, includes a non-linear (U-shaped) size effect, and
includes an interaction term where the combination of load-bearing masonry and
renovation is disproportionately risky. It also includes a deliberate red-herring
feature, permit processing time, that has no effect on risk, so that a good model
has to learn to ignore it. The fact that it does (see Results) is part of the
evidence that the pipeline behaves correctly.

## 4. Technical decisions and trade-offs

| Decision | Reasoning | Trade-off accepted |
|---|---|---|
| SQLite for storage | Zero infrastructure, file-based, reproducible, trivially inspectable. The schema is layered (provenance, then projects, then predictions) to make data lineage explicit. | Not concurrent and not a warehouse. Maps cleanly onto Postgres later without changing the application logic. |
| Streamlit for the interface | Fastest path to a usable, data-rich decision-support interface for one engineer on a short budget. The value is in the data, the model, and the explanations, not in bespoke front-end work. | React is SECO's production stack and the right choice for a multi-user, role-aware, branded application. Streamlit is the deliberate prototype trade. |
| scikit-learn, two models benchmarked | A regularised logistic-regression baseline is benchmarked against a gradient-boosted model, and whichever wins on held-out macro F1 is deployed automatically. This is honest and data-driven rather than reaching for the fanciest model by default. | On this largely additive synthetic label the linear model wins and ships. The gradient-boosted model is retained because on SECO's richer real data, with stronger interactions, it would be expected to pull ahead, and the harness will pick it up automatically when it does. |
| Preprocessing inside the model pipeline | One-hot encoding and scaling live inside the persisted artifact, which removes any train-serve skew and lets raw project rows be scored directly. | Slightly less control than a separate feature store, which is the right structure only at a larger scale. |
| Ablation for local explanations | Attributes risk in the original, human-readable feature space, which is the inspector's vocabulary, with no extra dependency. | SHAP produces game-theoretically purer values. It is wired in as an optional path that auto-selects the correct explainer for the deployed model (linear or tree), and it has been verified to agree with the ablation method on the same top drivers. |
| Language-model briefing is optional | The model only translates the structured drivers into prose. With no API key, a deterministic template produces the same content. The language model is never on the critical path. | Briefing prose is plainer without a key. The product is fully functional and reproducible offline. |

## 5. What would ship to production tomorrow, and what would be replaced?

What is already the right shape and would ship:

- the layered pipeline with a provenance audit trail,
- the pipeline-wrapped model with automatic benchmarking and best-model
  selection,
- explanations in the inspector's vocabulary, with two methods that
  cross-validate each other,
- the decision-support pattern of rank, filter, drill down, brief, and the
  risk-based allocation workflow it supports.

What would be replaced:

- the synthetic risk label, replaced by SECO's real historical inspection
  outcomes. The system is built so that this is a label-source swap and a
  retrain, not a rewrite. This is the single most important property of the
  design.
- the offline sampling prior, replaced by the parsed LUSTAT SDMX feed and,
  ideally, project-level permit records,
- SQLite, replaced by managed Postgres, and Streamlit, replaced by a React
  front-end over a scoring API.

## 6. With three more months

- Real labels and calibration. Train on SECO inspection history, calibrate the
  probabilities, and tune the High-risk recall threshold, because in this
  setting a missed high-risk project costs far more than a false alarm, so the
  operating point should be chosen deliberately rather than left at the default.
- Documents, not just tabular data. Add an extraction layer over inspection
  reports and permit PDFs, and a computer-vision component over defect imagery,
  to pull structured risk signals out of unstructured records.
- Monitoring and a feedback loop. Track data drift and model performance, and
  capture inspector overrides and outcomes as new training signal.
- An allocation layer. Turn the risk ranking into an actual assignment under
  capacity constraints, matching inspector seniority to project risk
  automatically, which is the subject of the next section.

## Operating model, change management, and adoption

The model is the easy part. Whether a tool like this delivers value depends
almost entirely on the operating model it enables and on whether the people who
do the work adopt it. This section is the part that matters most for the next
steps.

### The operating-model shift: managing the portfolio by risk

Today the inspection portfolio is handled largely first-come and by individual
judgement. ERA enables a different operating model: a risk-based portfolio.
Incoming projects are ranked by their risk index, and the response is tiered to
the band. High-risk projects receive intensive technical control and are
assigned to the most experienced inspectors. Medium-risk projects receive
standard review. Low-risk projects receive lighter-touch or desk-based review.

The central lever is the assignment of the riskiest portfolio to the most
experienced inspector. This is not a stylistic preference, it follows from the
economics. A senior inspector's comparative advantage is detecting subtle,
high-consequence defects that a junior inspector would miss. Spending that
capacity on low-risk, routine work is the single largest avoidable waste of the
firm's scarcest resource. By producing a defensible ranking and the reasons
behind it, ERA lets a planner concentrate senior expertise where its marginal
value is highest, and lets junior inspectors safely handle the long tail of
low-risk work, with the rationale documented in case the assignment is ever
questioned.

### Adoption is a trust problem, not a technology problem

Inspectors are experienced professionals who are accountable for their
judgement. They will not, and should not, defer to a black box. The design
choices that look technical are in fact the adoption mechanism:

- Explanations in the inspector's own vocabulary exist so that a person can
  audit the reasoning and decide whether to trust it on a given project. A score
  without a reason gets ignored. A score with a reason the inspector recognises
  gets used.
- The factors the model considers are documented in plain language in the
  application, so the basis of a rating is never hidden.
- The system is explicitly positioned as routing attention, not making the call.
  The inspector remains the decision-maker and remains accountable. This is both
  the correct safety posture and the thing that makes adoption possible, because
  it does not threaten the inspector's professional authority or deskill the
  role.

### A phased rollout that earns trust before it influences decisions

1. Shadow mode. Run ERA alongside the existing process without letting it change
   any decision. Compare its rankings to what planners and inspectors actually
   did, and to outcomes where available. This builds an evidence base and
   surfaces where the model is weak before anything is at stake.
2. Assisted mode. Let ERA inform triage and assignment, with every override
   logged. Overrides are not failures, they are the most valuable signal in the
   system: they show where the model and the expert disagree, and they become
   training data.
3. Embedded mode. Once trust is established and measured, integrate the ranking
   and assignment into the normal planning workflow, still advisory, still with
   the human accountable.

### Governance and the failure modes to manage

- Auditability. The provenance, projects, and predictions layers, together with
  versioned models, mean any rating can be reconstructed and explained after the
  fact. This matters for a firm whose product is independent, defensible
  judgement.
- Automation bias is the main risk. The danger is not that the model is wrong
  occasionally, it is that people stop thinking and treat the score as truth.
  Mitigations: keep the tool advisory, never let it become an automatic pass or
  fail, surface uncertainty, and keep accountability with the human.
- Adoption metrics. Track the override rate trending down as trust grows, the
  time taken to triage a portfolio, the share of genuinely high-risk projects
  caught early, the volume of senior inspector-hours reallocated from low-risk to
  high-risk work, and the reduction in variance between how different planners
  handle comparable projects. These measure whether the operating model is
  actually changing, which is the real objective.

## Architecture

```
data.public.lu (STATEC, CC0)
        |
        v
  ingest.py        record provenance, build a real building-mix sampling prior
        |
        v
  synthesize.py    hybrid portfolio: real prior plus a documented synthetic label
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
  briefing.py      inspector briefing (language model or deterministic template)
        |
        v
  app/app.py       dashboard: portfolio triage, score a building, geographic risk, factors
  geo.py           per-canton risk aggregation and map (real boundaries, offline fallback)
```

## Results (held-out test set, n_train = 1125, n_test = 375, seed = 42)

| Model | Accuracy | Macro F1 | High-risk recall |
|---|---|---|---|
| Majority-class baseline | 0.55 | n/a | n/a |
| Logistic regression (deployed) | 0.731 | 0.669 | 0.603 |
| Gradient boosting | 0.680 | 0.609 | 0.485 |

Sanity check: the red-herring feature, permit processing time, ranks 13th of 14
by permutation importance. The model correctly learned that it carries no risk
signal.

On evaluation honesty: these numbers validate the pipeline, the product, and the
end-to-end workflow. They are not a claim of real-world predictive validity,
because the label is synthetic. Real validity requires SECO's labelled inspection
data, at which point the same code retrains and the same application works
unchanged.

## Repository layout

```
src/seco_risk_radar/
  config.py       paths, seed, real STATEC dataset and resource identifiers
  taxonomy.py     real Luxembourg cantons and regions, STATEC building types
  ingest.py       fetch provenance and build the real-data sampling prior (offline fallback)
  synthesize.py   hybrid portfolio: real prior plus a documented synthetic label
  database.py     SQLite schema and load/read helpers
  features.py     feature definitions and leakage-safe preprocessing
  model.py        train, benchmark, deploy, and score
  explain.py      global importance and local ablation attributions, plus optional SHAP
  briefing.py     inspector briefing with a deterministic template fallback
  geo.py          per-canton risk aggregation and map (real boundaries, offline fallback)
scripts/          01_run_pipeline, 02_train, 03_predict, 04_risk_map
app/app.py        dashboard (portfolio triage, score a building, geographic risk, factors)
tests/            synthesize, features, database
```

## Tests

```bash
pip install pytest && pytest -q
```

## Data licence

STATEC Autorisations de batir is published under Creative Commons Zero (CC0). All
synthetic data in this repository is generated locally and reproducibly.

The map uses canton boundaries from Cantons in Luxembourg 2024 (SIG-GR / GIS-GR,
data.public.lu), licensed CC-BY 4.0. They are downloaded on first use and cached
locally. If the portal is unreachable, the map falls back to a centroid bubble
map, so nothing fails when offline.
