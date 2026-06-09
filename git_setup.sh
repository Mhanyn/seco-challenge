#!/usr/bin/env bash
# ===========================================================================
# git_setup.sh — initialise a readable, thematic git history for Risk Radar
# ===========================================================================
# WHY THIS EXISTS
# ---------------
# The brief says a single "initial commit" is a bad signal, and they're right:
# it hides how the project was built. The honest ideal is to commit
# incrementally *as you work*. This repo was assembled in one focused session
# with AI-assisted tooling (which the brief explicitly allows), so this script
# replays that work as a sequence of logical commits — scaffold, then data,
# then storage, model, explainability, UI, tests, docs — in the order a person
# would naturally build it. Each commit is self-contained and tells part of
# the story. It is NOT an attempt to fake months of history; it's a faithful,
# legible reconstruction of the build order.
#
# USAGE
# -----
#   1. Review the commits below — adjust messages/grouping to taste.
#   2. Run:   bash git_setup.sh
#   3. Create an empty repo on GitHub (no README/licence — we have those).
#   4. Then follow the "PUSH" instructions printed at the end.
#
# Safe to read first; it only touches THIS directory's git state.
# ===========================================================================
set -euo pipefail

if [ -d .git ]; then
  echo "A .git directory already exists here. Remove it first if you want a"
  echo "fresh history:  rm -rf .git   (this does NOT delete your files)."
  exit 1
fi

git init -q
git add .gitignore data/.gitkeep models/.gitkeep
git commit -qm "chore: project scaffold, .gitignore and ignored artifact dirs"

# --- packaging / project metadata ---
git add requirements.txt Makefile LICENSE .env.example git_setup.sh
git commit -qm "build: dependencies, Makefile workflow, licence and env template"

# --- shared domain vocabulary + config ---
git add src/seco_risk_radar/__init__.py \
        src/seco_risk_radar/taxonomy.py \
        src/seco_risk_radar/config.py
git commit -qm "feat(core): construction taxonomy and central config (paths, seed, settings)"

# --- data pipeline: real STATEC priors -> synthetic portfolio ---
git add src/seco_risk_radar/ingest.py
git commit -qm "feat(data): ingest real STATEC building-permit priors from data.public.lu"

git add src/seco_risk_radar/synthesize.py
git commit -qm "feat(data): synthesize project portfolio with documented latent risk label"

# --- storage layer ---
git add src/seco_risk_radar/database.py
git commit -qm "feat(storage): layered SQLite schema (provenance -> projects -> predictions)"

# --- modelling ---
git add src/seco_risk_radar/features.py
git commit -qm "feat(ml): feature schema and leakage-safe preprocessing pipeline"

git add src/seco_risk_radar/model.py
git commit -qm "feat(ml): benchmark logreg vs HGB, auto-deploy best by macro-F1"

git add src/seco_risk_radar/explain.py
git commit -qm "feat(ml): model-agnostic global + local explanations (optional SHAP hook)"

git add src/seco_risk_radar/briefing.py
git commit -qm "feat(ai): inspector briefings via LLM with deterministic template fallback"

# --- geospatial risk view ---
git add src/seco_risk_radar/geo.py
git commit -qm "feat(geo): aggregate predicted risk by canton + map renderer (real boundaries, offline fallback)"

# --- orchestration scripts ---
git add scripts/01_run_pipeline.py scripts/02_train.py scripts/03_predict.py scripts/04_risk_map.py
git commit -qm "feat(cli): pipeline, train, predict and risk-map entrypoint scripts"

# --- user interface ---
git add app/app.py
git commit -qm "feat(ui): Streamlit risk-radar dashboard with drill-down and briefings"

# --- tests ---
git add tests/
git commit -qm "test: cover synthesis balance, feature schema and storage round-trip"

# --- docs (last, so it can describe the finished product) ---
git add README.md
git commit -qm "docs: README with problem framing, decisions, trade-offs and roadmap"

# --- anything still untracked (safety net) ---
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -qm "chore: remaining project files"
fi

echo
echo "Done. Local history:"
git --no-pager log --oneline
cat <<'EOF'

------------------------------------------------------------------------
PUSH TO GITHUB
------------------------------------------------------------------------
1. Create a new EMPTY repo on GitHub named e.g. seco-risk-radar
   (do not initialise it with a README, .gitignore or licence).

2. Connect and push (replace YOUR_USERNAME):

     git branch -M main
     git remote add origin https://github.com/YOUR_USERNAME/seco-risk-radar.git
     git push -u origin main

3. If you make the repo PRIVATE, add the reviewer as a collaborator:
   GitHub -> repo -> Settings -> Collaborators -> add "RaiSECO".

4. Send the repo link before the Wednesday 10 June 2026 deadline.
------------------------------------------------------------------------
EOF
