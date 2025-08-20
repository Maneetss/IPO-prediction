.PHONY: setup all train eval predict smoke
VENV=.venv
PY=$(VENV)/bin/python

setup:
	python -m venv $(VENV)
	. $(VENV)/bin/activate; pip install -r requirements.txt

all:
	$(PY) prepdata.py
	$(PY) visz.py
	$(PY) outliners.py
	$(PY) scale.py
	$(PY) split-train-test.py
	$(PY) model.py --epochs 60 --optimizer adam --lr 1e-3
	$(PY) evalutions.py

train:
	$(PY) model.py --epochs 60 --optimizer adam --lr 1e-3

eval:
	$(PY) evalutions.py

predict:
	$(PY) predict.py --csv data/Indian_IPO_Market_Data.csv --out data/predictions.csv

smoke:
	$(PY) split-train-test.py
	$(PY) model.py --epochs 2
