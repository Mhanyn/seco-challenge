# ===========================================================================
# SECO Risk Radar - reproducible developer workflow
# ===========================================================================
# Run `make help` to see available targets. The happy path is simply:
#
#     make install
#     make all      # pipeline -> train -> predict
#     make app       # launch the dashboard
#
# PYTHON can be overridden, e.g.  make train PYTHON=python3.12
# N controls how many synthetic projects the pipeline generates.
# ===========================================================================

PYTHON ?= python
N      ?= 1500

.DEFAULT_GOAL := help

.PHONY: help install pipeline train predict map app test all clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

pipeline: ## 1. Ingest real STATEC priors + synthesize the project portfolio into SQLite
	$(PYTHON) scripts/01_run_pipeline.py --n $(N)

train: ## 2. Benchmark models, deploy the best, write metrics + importances
	$(PYTHON) scripts/02_train.py

predict: ## 3. Score every project and persist predictions + drivers
	$(PYTHON) scripts/03_predict.py

map: ## Render the canton risk map (PNG) from the scored portfolio
	$(PYTHON) scripts/04_risk_map.py

app: ## Launch the Streamlit dashboard
	streamlit run app/app.py

test: ## Run the test suite
	$(PYTHON) -m pytest -q

all: pipeline train predict ## Run the full data + model pipeline end to end

clean: ## Remove generated data and model artifacts (keeps .gitkeep)
	find data -mindepth 1 -not -name '.gitkeep' -delete
	find models -mindepth 1 -not -name '.gitkeep' -delete
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
