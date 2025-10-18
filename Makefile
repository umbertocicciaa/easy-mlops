SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
CLI := $(VENV)/bin/make-mlops-easy
IMAGE ?= make-mlops-easy

.PHONY: help venv install install-dev lint format test coverage docs-serve docs-build docs-deploy train predict status observe docker-build docker-run docker-shell clean

help: ## Show available make targets
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z0-9_\-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create the Python virtual environment if missing
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

install: venv ## Install package in editable mode
	$(PIP) install -e .

install-dev: venv ## Install package with development dependencies
	$(PIP) install -e .[dev]

lint: venv ## Run code quality checks
	$(VENV)/bin/flake8 easy_mlops tests

format: venv ## Format code using black
	$(VENV)/bin/black easy_mlops tests

test: venv ## Execute test suite
	$(PYTEST)

coverage: venv ## Run tests with coverage summary (using Python trace)
	@mkdir -p trace_summary
	@$(PYTHON_BIN) -c "import os, sys, trace, pytest; ignoredirs=[os.path.join(os.getcwd(), '.venv'), sys.prefix, sys.exec_prefix]; ignoremods={'pytest','_pytest','numpy','pandas','sklearn'}; tracer=trace.Trace(count=True, trace=False, ignoremods=ignoremods, ignoredirs=ignoredirs); exit_code=tracer.runfunc(pytest.main, ['tests']); results=tracer.results(); results.write_results(summary=True, coverdir='trace_summary'); sys.exit(exit_code)"
	@echo "Coverage report saved to trace_summary/ (see summary above)"

docs-serve: venv ## Serve MkDocs site locally
	$(VENV)/bin/mkdocs serve

docs-build: venv ## Build MkDocs site into the site/ directory
	$(VENV)/bin/mkdocs build

docs-deploy: venv ## Deploy documentation to GitHub Pages (requires permissions)
	$(VENV)/bin/mkdocs gh-deploy --force

train: venv ## Run training pipeline (DATA=path/to.csv TARGET=column CONFIG=path DEPLOY=false ARGS=\"...\")
	@if [ -z "$(DATA)" ]; then echo "Usage: make train DATA=path/to.csv [TARGET=col] [CONFIG=cfg.yaml] [DEPLOY=false] [ARGS=\"...\"]"; exit 1; fi; \
	cmd="$(CLI) train $(DATA)"; \
	if [ -n "$(TARGET)" ]; then cmd="$$cmd --target $(TARGET)"; fi; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --config $(CONFIG)"; fi; \
	if [ "$(DEPLOY)" = "false" ]; then cmd="$$cmd --no-deploy"; fi; \
	if [ -n "$(ARGS)" ]; then cmd="$$cmd $(ARGS)"; fi; \
	echo "$$cmd"; \
	eval "$$cmd"

predict: venv ## Run prediction pipeline (DATA=path/to.csv MODEL_DIR=deployment_dir CONFIG=path OUTPUT=preds.json ARGS=\"...\")
	@if [ -z "$(DATA)" ] || [ -z "$(MODEL_DIR)" ]; then echo "Usage: make predict DATA=path/to.csv MODEL_DIR=models/deployment_xxx [CONFIG=cfg.yaml] [OUTPUT=preds.json] [ARGS=\"...\"]"; exit 1; fi; \
	cmd="$(CLI) predict $(DATA) $(MODEL_DIR)"; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --config $(CONFIG)"; fi; \
	if [ -n "$(OUTPUT)" ]; then cmd="$$cmd --output $(OUTPUT)"; fi; \
	if [ -n "$(ARGS)" ]; then cmd="$$cmd $(ARGS)"; fi; \
	echo "$$cmd"; \
	eval "$$cmd"

status: venv ## Show deployment status (MODEL_DIR=deployment_dir CONFIG=path)
	@if [ -z "$(MODEL_DIR)" ]; then echo "Usage: make status MODEL_DIR=models/deployment_xxx [CONFIG=cfg.yaml]"; exit 1; fi; \
	cmd="$(CLI) status $(MODEL_DIR)"; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --config $(CONFIG)"; fi; \
	echo "$$cmd"; \
	eval "$$cmd"

observe: venv ## Generate observability report (MODEL_DIR=deployment_dir CONFIG=path)
	@if [ -z "$(MODEL_DIR)" ]; then echo "Usage: make observe MODEL_DIR=models/deployment_xxx [CONFIG=cfg.yaml]"; exit 1; fi; \
	cmd="$(CLI) observe $(MODEL_DIR)"; \
	if [ -n "$(CONFIG)" ]; then cmd="$$cmd --config $(CONFIG)"; fi; \
	echo "$$cmd"; \
	eval "$$cmd"

docker-build: ## Build Docker image (IMAGE=make-mlops-easy)
	docker build -t $(IMAGE) .

docker-run: ## Run Docker image (CMD defaults to --help)
	docker run --rm $(DOCKER_FLAGS) $(IMAGE) $(or $(CMD),--help)

docker-shell: ## Start interactive shell inside Docker image
	docker run --rm -it $(DOCKER_FLAGS) $(IMAGE) /bin/bash

clean: ## Remove build artifacts and caches
	# remove build artifacts
	rm -rf build dist *.egg-info .pytest_cache coverage_data trace_summary site
	# remove python compiled files and caches
	# skip the virtualenv directory when removing caches
	find . -path './.venv' -prune -o -type d -name "__pycache__" -exec rm -rf {} + || true
	find . -path './.venv' -prune -o -type f -name "*.py[co]" -exec rm -f {} + || true
