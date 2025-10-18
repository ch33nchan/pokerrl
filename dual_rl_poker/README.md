# Dual RL Poker Benchmarks

Reproducible self‐play experiments for Kuhn and Leduc poker built around the
ARMAC (actor + regret) paradigm. The repository currently ships:

- **Neural-tabular ARMAC** with logistic λ adaptation (no learned scheduler in
  the data path yet).
- **OpenSpiel CFR anchors** for both games to validate evaluation metrics.
- **Scheduler, meta-regret, and Rust environment infrastructure** ready for
  integration once higher-capacity experiments resume.

Fresh submission runs (500 iterations, 128 episodes per iteration, seeds 0–4)
now land under structured folders inside `results/`, grouped by experiment name,
game, policy type, and seed. The canonical “submission” sweep is reproduced by
the helper script described below; aggregate manifests and plots refresh
automatically.

## Environment setup

```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Install OpenSpiel manually if not already available:
# https://github.com/deepmind/open_spiel
```

The training scripts only require CPU PyTorch. Optional extras such as `tqdm`
enhance logging but are not mandatory.

## Reproducing the submission sweep

The helper script below re-creates every result included in the submission
package. Pass `--backend pyspiel` if you prefer the OpenSpiel environment; the
default uses the Rust backend.

```bash
python3 scripts/run_poker_suite.py \
  --output-dir results \
  --backend rust \
  --experiment-name submission_suite
```

This expands to:

1. Neural ARMAC on Kuhn & Leduc (seeds 0–4, 500 iterations, 128 episodes/iter).
2. CFR anchors on Kuhn & Leduc (1 000 iterations, seed 0).
3. Aggregation via `generate_results.py` and plot refresh through
   `create_plots.py`.

All artefacts appear under
`results/submission_suite/<game>/<policy>/seed_<n>/…timestamp….json`, and the
suite summary sits inside `results/submission_suite/summary/`.

## Key artefacts

- `results/<experiment>/<game>/<policy>/seed_*/…json` – raw logs per run.
- `results/<experiment>/summary/experiment_summary.json` – aggregate stats for the
  experiment suite executed via `run_poker_suite.py`.
- `results/experiment_summary.json` – aggregate over the entire `results/`
  directory.
- `results/plots/*.png`, `results/tables/performance_table.tex` – visualisations
  and LaTeX-ready tables generated from the manifests.
- `SUBMISSION_OVERVIEW.md` – one-page summary of experiments, commands, and
  final metrics.

## Repository layout

```
dual_rl_poker/
├── algs/            # ARMAC core, scheduler/meta-regret infrastructure
├── analysis/        # Notebooks and off-line tooling
├── configs/         # Reference configuration files
├── diagrams/        # Project diagrams
├── eval/            # OpenSpiel-based evaluators
├── experiments/     # Legacy experiment helpers (kept for reference)
├── games/           # OpenSpiel wrappers
├── nets/            # Neural network definitions used by ARMAC
├── results/         # Manifests, plots, and experiment logs
├── scripts/         # Utility scripts and automation
├── rust/            # PyO3 bindings for high-throughput environments
├── run_real_training.py  # Primary training loop for current experiments
└── generate_results.py   # Aggregates *_seed*.json files into manifests
```

## Additional tooling

- `generate_results.py` now accepts `--results-dir` and `--output` to summarise
  arbitrary directories (e.g. `results/submission_runs`).
- `create_plots.py` consumes the global manifest and regenerates every plot and
  LaTeX table required for reports/blog posts.
- `utils/rust_env.py` offers a parity checker against OpenSpiel when the Rust
  module is built locally (`python3 -m utils.rust_env kuhn_poker`).

## Next steps

The scheduler and Rust environment layers are implemented but not yet wired
into the production training loop. Future work will:

1. Integrate the per-state scheduler and meta-regret trainer into
   `run_real_training.py`.
2. Switch data collection to the Rust environments for higher throughput.
3. Extend experiments beyond small poker (e.g., to larger imperfect-information
   benchmarks) once the above components are validated.
