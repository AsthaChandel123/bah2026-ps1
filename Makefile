# urbanheat — developer Makefile (ISRO BAH-2026 PS-1)
# Usage:  make <target>.  See `make help` for the list.
# PYTHON can be overridden:  make demo PYTHON=python3.11

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
CITY   ?= Delhi

.DEFAULT_GOAL := help
.PHONY: help setup demo app test lint format clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package (editable) + dev extras
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "Core installed. For full stack:  $(PIP) install -e '.[all]'"
	@echo "For GEE mode also run:  earthengine authenticate"

demo:  ## Run the end-to-end SYNTHETIC pipeline (no GEE / no network), city=$(CITY)
	$(PYTHON) -m urbanheat.cli run --mode synthetic --city $(CITY) --output-dir outputs

app:  ## Launch the Streamlit dashboard
	$(PYTHON) -m streamlit run app/streamlit_app.py

test:  ## Run the test suite (GEE-marked tests skipped by default)
	$(PYTHON) -m pytest

lint:  ## Lint with ruff (and check black formatting)
	$(PYTHON) -m ruff check urbanheat tests app
	$(PYTHON) -m black --check urbanheat tests app

format:  ## Auto-format with black + ruff --fix
	$(PYTHON) -m ruff check --fix urbanheat tests app
	$(PYTHON) -m black urbanheat tests app

clean:  ## Remove caches, build artifacts and generated outputs (keeps .gitkeep)
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} +
	find outputs -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find data    -type f ! -name '.gitkeep' -delete 2>/dev/null || true
