# Dual RL Poker Benchmarks

Reproducible self‐play experiments for Kuhn and Leduc poker built around the
ARMAC (actor + regret) paradigm. The repository currently ships:

- **Neural-tabular ARMAC** with logistic λ adaptation (no learned scheduler in
  the data path yet).
- **OpenSpiel CFR anchors** for both games to validate evaluation metrics.
- **Scheduler, meta-regret, and Rust environment infrastructure** ready for
  integration once higher-capacity experiments resume.

Fresh submission runs (500 iterations, 128 episodes per iteration, seeds 0–4)
live under `results/submission_runs/`, together with CFR baselines and a
machine-generated manifest.

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

The following commands re-create every result included in the submission
package. They match the exact pipeline executed for the `results/submission_runs`
manifest.

```bash
# 1. Neural-tabular ARMAC (Kuhn & Leduc, seeds 0..4, 500 iterations)
for game in kuhn_poker leduc_poker; do
  for seed in 0 1 2 3 4; do
    python3 run_real_training.py \
      --game "$game" \
      --iterations 500 \
      --episodes-per-iteration 128 \
      --seed "$seed" \
      --output-dir results/submission_runs \
      --tag submission
  done
done

# 2. CFR anchors (Kuhn & Leduc, 1 000 iterations)
python3 run_real_training.py \
  --game kuhn_poker \
  --iterations 1000 \
  --algorithm cfr \
  --seed 0 \
  --output-dir results/submission_runs \
  --tag submission

python3 run_real_training.py \
  --game leduc_poker \
  --iterations 1000 \
  --algorithm cfr \
  --seed 0 \
  --output-dir results/submission_runs \
  --tag submission

# 3. Summaries & plots
python3 generate_results.py \
  --results-dir results/submission_runs \
  --output results/submission_summary.json
python3 generate_results.py  # refresh global manifest (results/experiment_summary.json)
python3 create_plots.py      # refresh plots in results/plots/ and tables/
```

For convenience a `make submission` target performs the same sweep.

## Key artefacts

- `results/submission_runs/` – raw JSON logs for each run.
- `results/submission_summary.json` – aggregate stats for the submission sweep.
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
