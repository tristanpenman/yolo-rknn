PYTHON ?= python3
PYLINT ?= pylint

.PHONY: lint

lint:
	$(PYLINT) python/**/*.py
