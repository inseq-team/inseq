#* Variables
SHELL := /usr/bin/env bash
PYTHON := python3

#* Docker variables
IMAGE := inseq
VERSION := latest

.PHONY: help
help:
	@echo "Commands:"
	@echo "poetry-download : downloads and installs the poetry package manager"
	@echo "poetry-remove   : removes the poetry package manager"
	@echo "install         : installs required dependencies"
	@echo "install-gpu    : installs required dependencies, plus Torch GPU support"
	@echo "install-dev     : installs the dev dependencies for the project"
	@echo "install-dev-gpu : installs the dev dependencies for the project, plus Torch GPU support"
	@echo "update-deps     : updates the dependencies and writes them to requirements.txt"
	@echo "check-style     : run checks on all files without fixing them."
	@echo "fix-style       : run checks on files and potentially modifies them."
	@echo "check-safety    : run safety checks on all tests."
	@echo "lint            : run linting on all files (check-style + check-safety)"
	@echo "test            : run all tests."
	@echo "fast-test       : run all quick tests."
	@echo "codecov         : check coverage of all the code."
	@echo "build-docs      : build sphinx documentation."
	@echo "serve-docs      : serve documentation locally."
	@echo "docs            : build and serve generated documentation locally."
	@echo "docker-build    : builds docker image for the package."
	@echo "docker-remove   : removes built docker image."
	@echo "clean           : cleans all unecessary files."

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | $(PYTHON) - --uninstall

#* Installation

.PHONY: add-torch-gpu
add-torch-gpu:
	poetry run poe upgrade-pip
	poetry run pip uninstall torch -y
	poetry run poe torch-cuda11

.PHONY: install
install:
	poetry install

.PHONY: install-dev
install-dev:
	poetry install --all-extras --with lint,docs --sync
#	-poetry run mypy --install-types --non-interactive ./
	poetry run pre-commit install
	poetry run pre-commit autoupdate

.PHONY: install-gpu
install-gpu: install add-torch-gpu

.PHONY: install-dev-gpu
install-dev-gpu: install-dev add-torch-gpu

.PHONY: install-ci
install-ci:
	poetry install --with lint

.PHONY: update-deps
update-deps:
	poetry lock && poetry export --without-hashes > requirements.txt

#* Linting
.PHONY: check-style
check-style:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./
	poetry run black --diff --check --config pyproject.toml ./
#   poetry run darglint --verbosity 2 inseq tests
	poetry run flake8 --config setup.cfg ./
#	poetry run mypy --config-file pyproject.toml ./

.PHONY: fix-style
fix-style:
	poetry run pyupgrade --exit-zero-even-if-changed --py38-plus **/*.py
	poetry run isort --settings-path pyproject.toml ./
	poetry run black --config pyproject.toml ./

.PHONY: check-safety
check-safety:
	poetry check
	poetry run safety check --full-report -i 51457 -i 51358
	poetry run bandit -ll --recursive inseq tests

.PHONY: lint
lint: check-style check-safety

#* Linting
.PHONY: test
test:
	poetry run pytest -c pyproject.toml -v

.PHONY: fast-test
fast-test:
	poetry run pytest -c pyproject.toml -v -m "not slow"

.PHONY: codecov
codecov:
	poetry run pytest --cov inseq --cov-report html

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
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make clean_docker VERSION=latest
# Example: make clean_docker IMAGE=some_name VERSION=0.1.0
.PHONY: docker-remove
docker-remove:
	@echo Removing docker $(IMAGE):$(VERSION) ...
#	docker rmi -f $(IMAGE):$(VERSION)

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: clean
clean: pycache-remove build-remove docker-remove
