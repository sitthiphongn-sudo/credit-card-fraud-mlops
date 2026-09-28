.PHONY: lint test validate all gate rollback serve down

lint:
	ruff check .

test:
	pytest -q

validate:
	python -m fraud.validate data/raw/creditcard.csv

all:
	python pipelines/flow.py

gate:
	python -m fraud.gate check --candidate "$(CANDIDATE)" $(if $(CHAMPION),--champion "$(CHAMPION)",)

rollback:
	python -m fraud.gate rollback "$(VERSION)"

serve:
	docker compose up -d --build

down:
	docker compose down
