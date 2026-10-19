.PHONY: help install run lint format test clean
.DEFAULT_GOAL := help

help: ## Show the available make targets
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Install the dependencies (uv sync)
	uv sync

run: ## Run a sample, e.g. make run SAMPLE=triage (without SAMPLE: list them)
	uv run decision-models $(SAMPLE)

lint: ## Check the code with ruff
	uv run ruff check .

format: ## Format the code with ruff
	uv run ruff format .

test: ## Run the tests
	uv run pytest

clean: ## Remove the pytest, ruff and Python caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__
