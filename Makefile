setup:
	python -m pip install --upgrade pip wheel poetry==1.6.1

install: setup
	poetry install

format:
	poetry run black .
	poetry run ruff src/ --fix

check-formatting:
	poetry run black src/ --check
	poetry run ruff check src/

check-typing:
	poetry run mypy src/

check: check-formatting check-typing
