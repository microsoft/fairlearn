# simple makefile to simplify repetitive build env management tasks

PYTHON ?= python
PYTEST ?= pytest

all: clean inplace test-unit

clean:
	git clean -xfd

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-coverage:
	$(PYTEST) test -m "not notebooks" --ignore=test/perf --ignore=test/install --cov=fairlearn --cov-report=xml --cov-report=html

test-unit:
	$(PYTEST) ./test/unit

test-perf:
	$(PYTEST) ./test/perf