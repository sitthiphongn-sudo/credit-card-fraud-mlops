.PHONY: lint test validate all serve down

lint:
	ruff check .

test:
	pytest -q

validate:
	python -m fraud.validate data/raw/creditcard.csv

all:
	docker compose up -d --build --force-recreate --wait

serve:
	docker compose up -d --build --wait

down:
	docker compose down
