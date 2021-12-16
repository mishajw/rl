PYTHON_FILES := $(wildcard *.py)

.PHONY: bandit_testbed
bandit_testbed: .env $(PYTHON_FILES)
	.env/bin/streamlit run bandit_testbed.py --server.headless true

.PHONY: fmt
fmt: .env
	.env/bin/black --line-length 80 $(PYTHON_FILES)

.env: requirements.txt
	python -m venv .env
	.env/bin/pip install -r requirements.txt
