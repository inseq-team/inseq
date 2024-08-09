#* Variables
SHELL := /bin/bash
PYTHON := .venv/bin/python

#* Docker variables
IMAGE := inseq
VERSION := latest

.PHONY: help
help:
	@echo "Commands:"
	@echo "uv-download     : downloads and installs the uv package manager"
	@echo "install         : installs required dependencies"
	@echo "install-dev     : installs the dev dependencies for the project"
	@echo "update-deps     : updates the dependencies and writes them to requirements.txt"
	@echo "check-style     : run checks on all files without fixing them."
	@echo "fix-style       : run checks on files and potentially modifies them."
	@echo "check-safety    : run safety checks on all tests."
	@echo "lint            : run linting on all files (check-style + check-safety)"
	@echo "test            : run all tests."
	@echo "test-cpu        : run all tests that do not depend on Torch GPU support."
	@echo "fast-test       : run all quick tests."
	@echo "codecov         : check coverage of all the code."
	@echo "build-docs      : build sphinx documentation."
	@echo "serve-docs      : serve documentation locally."
	@echo "docs            : build and serve generated documentation locally."
	@echo "docker-build    : builds docker image for the package."
#	@echo "docker-remove   : removes built docker image."
	@echo "clean           : cleans all unecessary files."

#* UV
uv-download:
	@echo "Downloading uv package manager..."
	@if [[ $OS == "Windows_NT" ]]; then \
		irm https://astral.sh/uv/install.ps1 | iex; \
	else \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv venv


.PHONY: uv-activate
uv-activate:
	@if [[ "$(OS)" == "Windows_NT" ]]; then \
		./uv/Scripts/activate.ps1 \
	else \
		source .venv/bin/activate; \
	fi

#* Installation

.PHONY: install
install:
	make uv-activate && uv pip install -r requirements.txt && uv pip install -e .

.PHONY: install-dev
install-dev:
	make uv-activate && uv pip install -r requirements-dev.txt && pre-commit install && pre-commit autoupdate
	

.PHONY: install-ci
install-ci:
	make uv-activate && uv pip install -r requirements-dev.txt

.PHONY: update-deps
update-deps:
	uv pip compile pyproject.toml -o requirements.txt
	uv pip compile --all-extras pyproject.toml -o requirements-dev.txt

#* Linting
.PHONY: check-style
check-style:
	$(PYTHON) -m ruff format --check --config pyproject.toml ./
	$(PYTHON) -m ruff check --no-fix --config pyproject.toml ./
#   $(PYTHON) -m pydoclint --config pyproject.toml inseq/
#	$(PYTHON) -m mypy --config-file pyproject.toml ./

.PHONY: fix-style
fix-style:
	$(PYTHON) -m ruff format --config pyproject.toml ./
	$(PYTHON) -m ruff check --config pyproject.toml ./

.PHONY: check-safety
check-safety:
	$(PYTHON) -m safety check --full-report -i 70612 -i 71670 -i 72089

.PHONY: lint
lint: fix-style check-safety

#* Linting
.PHONY: test
test:
	$(PYTHON) -m pytest -n auto -c pyproject.toml -v

.PHONY: test-cpu
test-cpu:
	$(PYTHON) -m pytest -n auto -c pyproject.toml -v -m "not require_cuda_gpu"

.PHONY: fast-test
fast-test:
	$(PYTHON) -m pytest -n auto -c pyproject.toml -v -m "not slow"

# Limits the number of threads to 4 to avoid overloading the CI
.PHONY: fast-test-ci
fast-test-ci:
	$(PYTHON) -m pytest -n 4 -c pyproject.toml -v -m "not slow"

.PHONY: codecov
codecov:
	$(PYTHON) -m pytest -n auto --cov inseq --cov-report html

#* Docs
.PHONY: build-docs
build-docs:
	cd docs && make html SPHINXOPTS="-W -j 4"

.PHONY: serve-docs
serve-docs:
	cd docs/_build/html && python3 -m http.server 8080

.PHONY: docs
docs: build-docs serve-docs

#* Docker
# Example: make docker VERSION=latest
# Example: make docker IMAGE=some_name VERSION=0.1.0
.PHONY: docker-build
docker-build:
	@echo "Building docker $(IMAGE):$(VERSION) ..."
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make clean_docker VERSION=latest
# Example: make clean_docker IMAGE=some_name VERSION=0.1.0
#.PHONY: docker-remove
#docker-remove:
#	@echo "Removing docker $(IMAGE):$(VERSION) ..."
# docker rmi -f $(IMAGE):$(VERSION)

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: clean
clean: pycache-remove build-remove # docker-remove
