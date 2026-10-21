.PHONY: install train test lint format serve docker clean

install:
	pip install -e ".[api,dev]"
	pre-commit install

train:
	demandcast train

test:
	pytest --cov=demandcast --cov-report=term-missing

lint:
	ruff check . && ruff format --check . && mypy src

format:
	ruff check --fix . && ruff format .

serve:
	demandcast serve --host 0.0.0.0 --port 8000

docker:
	docker build -t demandcast:latest .
	docker run --rm -p 8000:8000 demandcast:latest

clean:
	rm -rf artifacts .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
