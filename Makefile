.PHONY: install train train-fast legalbert api dashboard test lint docker

install:
	pip install -r requirements-dev.txt

train:          ## full linear model on CUAD (~15 min on 2 CPU cores, downloads CUAD on first run)
	python scripts/train.py

train-fast:     ## word n-grams only (~3 min)
	python scripts/train.py --fast

legalbert:      ## fine-tune Legal-BERT (GPU recommended; see notebooks/colab_legalbert.ipynb)
	python scripts/train_legalbert.py

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest -q

lint:
	ruff check src api dashboard scripts tests

docker:
	docker compose up --build
