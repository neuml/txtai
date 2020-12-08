# Project utility scripts
.PHONY: test

# Setup environment
export SRC_DIR := ./src/python
export TEST_DIR := ./test/python
export PYTHONPATH := ${SRC_DIR}:${TEST_DIR}:${PYTHONPATH}
export PATH := ${TEST_DIR}:${PATH}
export PYTHONWARNINGS := ignore

# Disable tokenizer parallelism for tests
export TOKENIZERS_PARALLELISM := false

# Default python executable if not provided
PYTHON ?= python

# Unit tests
test:
	${PYTHON} -m unittest discover -v -s ${TEST_DIR}

# Run tests while calculating code coverage
coverage:
	coverage run --source=${SRC_DIR} -m unittest discover -v -s ${TEST_DIR}
