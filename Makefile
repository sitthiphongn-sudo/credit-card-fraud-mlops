.PHONY: lint test validate all serve down

lint:
	ruff check .

test:
	pytest -q

validate:
	python -m fraud.validate data/raw/creditcard.csv

all:
	python pipelines/flow.py

serve:
	docker compose up -d --build

down:
	docker compose down
