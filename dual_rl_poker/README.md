# Dual RL for Small Poker: Actor-Critic with Regret Matching

## Overview
This project implements and compares three algorithm families on Kuhn and Leduc poker:
- **Deep CFR** (external sampling)
- **SD-CFR** (Self-Play Deep CFR)
- **ARMAC-style Dual RL** (Actor-Critic with Regret Matching)

All methods are evaluated using OpenSpiel's exact NashConv/exploitability metrics and head-to-head EV comparisons.

## Environment Setup
- Python 3.11 on macOS
- Dependencies: OpenSpiel, PyTorch (CPU), NumPy, SciPy, pandas, matplotlib, tyro
- Exact versions pinned in `requirements.txt`

## Repository Structure
```
dual_rl_poker/
├── games/          # Game wrappers and encodings
├── algs/           # Algorithm implementations
├── nets/           # Neural network architectures
├── eval/           # Evaluation utilities
├── logging/        # Diagnostics and manifest tracking
├── scripts/        # Training, evaluation, and plotting scripts
├── configs/        # YAML configuration files
├── results/        # CSV, Parquet results
└── paper/          # LaTeX manuscript
```

## Quick Start
```bash
# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run all experiments
make run_all

# Generate plots and tables
make analysis
```

## Primary Results
- Main metric: OpenSpiel NashConv/exploitability in game units
- Statistical significance: 95% bootstrap confidence intervals
- Secondary metrics: steps-to-threshold, wall-clock time, head-to-head EV

## Reproducibility
- All results auto-generated from single manifest source of truth
- Pinned Python and OpenSpiel versions
- One-command artifact reproduction on macOS CPU