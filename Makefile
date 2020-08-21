# Project utility scripts
.PHONY: test

# Setup environment
export SRC_DIR := ./src/python
export TEST_DIR := ./test/python
export PYTHONPATH := ${SRC_DIR}:${TEST_DIR}:${PYTHONPATH}
export PATH := ${TEST_DIR}:${PATH}

# Default python executable if not provided
PYTHON ?= python

# Unit tests
test:
	${PYTHON} -W ignore::DeprecationWarning -m unittest discover -v -s ${TEST_DIR}
