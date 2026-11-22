.PHONY: install experiment train serve simulate test docker
install:
	pip install -r requirements.txt && pip install -e .
experiment:
	python -m upi_ids.cli experiment --out results
train:
	python -m upi_ids.cli train --out results/models/hybrid_detector.pkl
serve:
	python -m upi_ids.cli serve --model results/models/hybrid_detector.pkl --port 8000
simulate:
	python -m upi_ids.cli simulate --n 2000
test:
	pytest -q
docker:
	docker compose up --build
