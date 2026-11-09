.DEFAULT_GOAL := help
.NOTPARALLEL:
.PHONY: help setup check test populate ping run experiments offline report verify build review

help:
	@printf '%s\n' 'make populate     Create .env, install dependencies and generate the catalogue' 'make review       Check code and execute all live scenarios' 'make offline      Check code and execute scripted failure scenarios' 'make run          Search using QUERY_TEXT or PRODUCT_QUERY in .env' 'make report       Render the saved measurements as Markdown' 'make verify       Validate results and journal without writing files'

setup:
	@if [ ! -e .env ]; then cp .env.example .env; fi
	uv sync --locked

check:
	uv run --locked ruff format --check catalogue tests
	uv run --locked ruff check catalogue tests
	uv run --locked mypy

test:
	uv run --locked pytest -q

populate: setup
	uv run --locked catalogue-populate

ping:
	uv run --locked catalogue-ping

run: populate
	uv run --locked catalogue-search

experiments:
	uv run --locked catalogue-experiments

report:
	uv run --locked catalogue-report

verify:
	uv run --locked catalogue-verify

build:
	uv build --no-sources

offline:
	$(MAKE) populate check test
	uv run --locked catalogue-offline
	$(MAKE) report verify

review:
	$(MAKE) populate check test
	$(MAKE) experiments report verify
