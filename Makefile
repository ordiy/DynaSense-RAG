.PHONY: lint test

lint:
	lint-imports

test:
	lint-imports && .venv/bin/pytest tests/ -x -q
