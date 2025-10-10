.PHONY: setup run_all analysis clean paper test help

# Default target
help:
	@echo "Dual RL Poker Project Commands:"
	@echo "  setup     - Set up environment and dependencies"
	@echo "  run_all   - Run all experiments"
	@echo "  analysis  - Generate plots and tables"
	@echo "  paper     - Compile LaTeX paper"
	@echo "  test      - Run unit tests"
	@echo "  clean     - Clean generated files"
	@echo "  help      - Show this help message"

setup:
	@echo "Setting up environment..."
	python -m venv venv
	source venv/bin/activate && pip install --upgrade pip
	source venv/bin/activate && pip install -r requirements.txt
	@echo "Note: OpenSpiel (pyspiel) needs to be installed manually from source"
	@echo "See https://github.com/deepmind/openSpiel for installation instructions"

run_all:
	@echo "Running all experiments..."
	source venv/bin/activate && python scripts/run_all_experiments.py

analysis:
	@echo "Generating analysis..."
	source venv/bin/activate && python scripts/plot.py
	source venv/bin/activate && python scripts/make_tables.py

paper:
	@echo "Compiling paper..."
	cd paper && latexmk -pdf main.tex

test:
	@echo "Running tests..."
	source venv/bin/activate && python -m pytest tests/ -v

clean:
	@echo "Cleaning generated files..."
	rm -rf results/*
	rm -rf logs/*
	cd paper && latexmk -c
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Individual experiment targets
kuhn_deep_cfr:
	source venv/bin/activate && python scripts/train.py --game kuhn_poker --method deep_cfr --seeds 0-19

kuhn_sd_cfr:
	source venv/bin/activate && python scripts/train.py --game kuhn_poker --method sd_cfr --seeds 0-19

kuhn_armac:
	source venv/bin/activate && python scripts/train.py --game kuhn_poker --method armac --seeds 0-19

leduc_deep_cfr:
	source venv/bin/activate && python scripts/train.py --game leduc_poker --method deep_cfr --seeds 0-9

leduc_sd_cfr:
	source venv/bin/activate && python scripts/train.py --game leduc_poker --method sd_cfr --seeds 0-9

leduc_armac:
	source venv/bin/activate && python scripts/train.py --game leduc_poker --method armac --seeds 0-9