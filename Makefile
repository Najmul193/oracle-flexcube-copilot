.PHONY: install lint typecheck test clean ui run

install:
	uv sync --dev

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/ tests/

test:
	pytest

test-cov:
	pytest --cov=src/oracle_flexcube_copilot --cov-report=term-missing

ui:
	streamlit run src/oracle_flexcube_copilot/ui/app.py

run:
	streamlit run src/oracle_flexcube_copilot/ui/app.py

clean:
	rm -rf .venv/
	rm -rf chroma_db/
	rm -rf cache/
	rm -rf logs/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true