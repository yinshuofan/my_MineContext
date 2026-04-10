.PHONY: check lint format typecheck test

check: lint format typecheck test

lint:
	uv run ruff check opencontext/ tests/

format:
	uv run ruff format --check opencontext/ tests/

typecheck:
	uv run mypy opencontext/

test:
	uv run pytest -m unit -v --tb=short
