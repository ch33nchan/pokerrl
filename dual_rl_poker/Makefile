.PHONY: setup submission run_all analysis clean paper test help

# Default target
help:
	@echo "ARMAC: Actor-Critic with Regret Matching - Project Commands:"
	@echo "  setup     - Set up environment and dependencies"
	@echo "  submission- Reproduce the submission sweep"
	@echo "  run_all   - Legacy alias for 'submission'"
	@echo "  analysis  - Generate plots and results analysis"
	@echo "  paper     - Compile LaTeX paper (paper_icml/)"
	@echo "  test      - Run unit tests (if available)"
	@echo "  clean     - Clean generated files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Documentation:"
	@echo "  ARMAC_COMPREHENSIVE_DOCUMENTATION.md - Complete technical documentation"
	@echo "  paper_icml/dual_rl_poker.pdf         - Academic paper"
	@echo "  README.md                            - Project overview"

setup:
	@echo "Setting up environment..."
	python -m venv .venv
	source .venv/bin/activate && pip install --upgrade pip
	source .venv/bin/activate && pip install -r requirements.txt
	@echo "Note: OpenSpiel (pyspiel) needs to be installed manually from source"
	@echo "See https://github.com/deepmind/openSpiel for installation instructions"

submission:
	@echo "Running neural-tabular ARMAC sweep (Kuhn & Leduc; seeds 0-4)..."
	@for game in kuhn_poker leduc_poker; do \
		for seed in 0 1 2 3 4; do \
			echo "==> $$game (seed $$seed)"; \
			python3 run_real_training.py --game $$game --iterations 500 --episodes-per-iteration 128 --seed $$seed --output-dir results/submission_runs --tag submission; \
		done; \
	done
	@echo "Running CFR anchors (Kuhn & Leduc)..."
	@python3 run_real_training.py --game kuhn_poker --iterations 1000 --algorithm cfr --seed 0 --output-dir results/submission_runs --tag submission
	@python3 run_real_training.py --game leduc_poker --iterations 1000 --algorithm cfr --seed 0 --output-dir results/submission_runs --tag submission
	@echo "Aggregating submission summary..."
	@python3 generate_results.py --results-dir results/submission_runs --output results/submission_summary.json
	@echo "Refreshing global experiment summary..."
	@python3 generate_results.py
	@echo "Regenerating plots and tables..."
	@python3 create_plots.py

run_all: submission
	@echo "'run_all' is now an alias for 'submission'."

analysis:
	@echo "Generating analysis and plots..."
	python3 generate_results.py
	python3 create_plots.py

paper:
	@echo "Compiling academic paper..."
	cd paper_icml && pdflatex dual_rl_poker.tex && pdflatex dual_rl_poker.tex

test:
	@echo "Running tests..."
	@if [ -d "tests" ]; then \
		source .venv/bin/activate && python -m pytest tests/ -v; \
	else \
		echo "No tests directory found - test suite not implemented"; \
	fi

clean:
	@echo "Cleaning generated files..."
	rm -rf results/plots/*
	rm -rf results/final/*
	cd paper_icml && rm -f *.aux *.log *.out *.bbl *.blg
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Quick experiment targets
quick_kuhn:
	@echo "Quick Kuhn poker experiment..."
	source .venv/bin/activate && python run_real_training.py --game kuhn_poker --iterations 200 --seed 0 --tag quick

quick_leduc:
	@echo "Quick Leduc poker experiment..."
	source .venv/bin/activate && python run_real_training.py --game leduc_poker --iterations 200 --seed 0 --tag quick

# Results verification
verify_results:
	@echo "Verifying experimental results..."
	source .venv/bin/activate && python -c "
import pandas as pd
import os
if os.path.exists('results/enhanced_manifest.csv'):
    df = pd.read_csv('results/enhanced_manifest.csv')
    print(f'Total experiments: {len(df)}')
    print('Methods:', df['method'].value_counts().to_dict())
    print('Games:', df['game'].value_counts().to_dict())
else:
    print('No results found - run experiments first')
"
