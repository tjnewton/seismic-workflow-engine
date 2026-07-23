# Developer task runner for seismic-workflow-engine.
# Wraps the exact commands CI runs. Assumes the `swe` conda env:
#   conda activate swe   (or prefix targets with: conda run -n swe make <target>)
.PHONY: help env lint type test docs notebooks guard

help:
	@echo "Targets: env lint type test docs notebooks guard"

env:
	conda env update -n swe -f environment.yml
	pip install -e . --no-deps

lint:
	ruff check .
	ruff format --check .

type:
	mypy src/swe

test:
	python -m pytest tests/ --cov=swe --cov-report=term-missing --cov-fail-under=85

docs:
	sphinx-build -b html -W --keep-going docs docs/_build/html

notebooks:
	MPLBACKEND=Agg jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=swe notebooks/*.ipynb

guard:
	python scripts/leak_guard.py --tracked
