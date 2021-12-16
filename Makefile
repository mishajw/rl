PYTHON_FILES := $(wildcard *.py)

.PHONY: run
run: .env $(PYTHON_FILES)
	.env/bin/streamlit run main.py --server.headless true

.PHONY: fmt
fmt: .env
	.env/bin/black --line-length 80 $(PYTHON_FILES)

.env: requirements.txt
	python -m venv .env
	.env/bin/pip install -r requirements.txt
