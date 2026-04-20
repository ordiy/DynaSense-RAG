VENV := .venv

.PHONY: lint test eval

lint:
	$(VENV)/bin/lint-imports
	$(VENV)/bin/python scripts/check_rag_core_imports.py

test:
	$(VENV)/bin/lint-imports && \
	$(VENV)/bin/python scripts/check_rag_core_imports.py && \
	$(VENV)/bin/pytest tests/ -x -q -m 'not slow'

eval:
	$(VENV)/bin/pytest tests/eval/ -v -m slow
