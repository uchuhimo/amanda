.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

STORE_TENSOR_SRCS = $(wildcard cc/tensorflow/store_tensor_to_file/kernels/*.cc) $(wildcard cc/tensorflow/store_tensor_to_file/ops/*.cc)
PY_HOOK_SRCS = $(wildcard cc/tensorflow/py_hook/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

PY_CONFIG = $(shell PKG_CONFIG_PATH=${CONDA_PREFIX}/lib/pkgconfig pkg-config --cflags --libs python3)
CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11 ${PY_CONFIG}
LDFLAGS = -shared ${TF_LFLAGS}

STORE_TENSOR_TARGET_LIB = cc/tensorflow/store_tensor_to_file/ops/store_tensor_to_file_ops.so
PY_HOOK_TARGET_LIB = cc/tensorflow/py_hook/py_hook.so

store_tensor_op: $(STORE_TENSOR_TARGET_LIB)
py_hook_op: $(PY_HOOK_TARGET_LIB)

build_cc: store_tensor_op py_hook_op

$(STORE_TENSOR_TARGET_LIB): $(STORE_TENSOR_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}
$(PY_HOOK_TARGET_LIB): $(PY_HOOK_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

proto:
	protoc -I src --python_out=src --mypy_out=src src/amanda/io/*.proto

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	rm -f $(STORE_TENSOR_TARGET_LIB)

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr .dephell_report

lint: ## check style with flake8
	flake8 src/amanda

test: build_cc ## run tests quickly with the default Python
	pytest -n auto

test-all: build_cc ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source src/amanda -m pytest
	coverage report -m
	coverage html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/amanda.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ src/amanda
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	poetry build

install: clean ## install the package to the active Python's site-packages
	poetry install
