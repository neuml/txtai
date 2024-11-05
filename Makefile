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

# Check for wget
WGET := $(shell wget --version 2> /dev/null)
ifndef WGET
    $(error "Required binary `wget` not found, please install wget OS package")
endif

# Download test data
data:
	mkdir -p /tmp/txtai
	wget -N https://github.com/neuml/txtai/releases/download/v6.2.0/tests.tar.gz -P /tmp
	tar -xvzf /tmp/tests.tar.gz -C /tmp

# Unit tests
test:
	${PYTHON} -m unittest discover -v -s ${TEST_DIR}

# Run tests while calculating code coverage
coverage:
	coverage run -m unittest discover -v -k testagent -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testann -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testapi -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testapp -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testarchive -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testcloud -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testconsole -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testdatabase -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testembeddings -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testgraph -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testmodels -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testoptional -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testpipeline.testaudio -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testpipeline.testdata -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testpipeline.testimage -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testpipeline.testllm -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testpipeline.testtext -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testpipeline.testtrain -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testscoring -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testserialize -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testvectors -s ${TEST_DIR}
	coverage run -m unittest discover -v -k testworkflow -s ${TEST_DIR}
	coverage combine
