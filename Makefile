.PHONY: install run lint format test clean

install:
	uv sync

run:
	uv run decision-models

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
