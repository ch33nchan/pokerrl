# Dual RL Poker Benchmarks

Reproducible self‐play experiments for Kuhn and Leduc poker built around the
ARMAC (actor + regret) paradigm. The repository currently ships:

- **Neural-tabular ARMAC + MARM-K gate** – a bi-level meta-scheduler that mixes
  actor, regret, risk-averse, exploratory, and CFR-style experts using
  approximate best-response utilities and a regret-matching bandit target.
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
git checkout -b sriniii
python3.11 -m venv .venv311
source .venv311/bin/activate
pip3.11 install --upgrade pip
pip3.11 install -r requirements.txt
# Install OpenSpiel manually if not already available:
# https://github.com/deepmind/open_spiel
```

The training scripts only require CPU PyTorch. Optional extras such as `tqdm`
enhance logging but are not mandatory.

## Running MARM-K training (CPU/Rust backends)

The default training loop now ships with the Meta-Adaptive K-Expert Gate
(MARM-K). A typical Mac CPU run using the deterministic Rust backend looks like:

```bash
python3.11 run_real_training.py \
  --game leduc_poker \
  --backend rust \
  --episodes-per-iteration 128 \
  --iterations 500 \
  --experts actor,regret,ra,explore,cfr \
  --br-budget 64 \
  --meta-unroll 16 \
  --handoff-tau 0.15 \
  --handoff-patience 3 \
  --state-cluster round+position+pot \
  --manifest-path results/manifest.csv
```

Set `--backend pyspiel` if the Rust module has not been built yet. The gate
learns on-line via short-horizon meta-gradients, tracks a regret-matching bandit
target, and automatically freezes subgames into the CFR head once local
exploitability stabilises below `--handoff-tau` for `--handoff-patience`
evaluations.

## Reproducing the submission sweep

The helper script below re-creates every result included in the submission
package. Pass `--backend pyspiel` if you prefer the OpenSpiel environment; the
default uses the Rust backend.

```bash
python3.11 scripts/run_poker_suite.py \
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
`results/submission_suite/<game>/<policy>/seed_<n>/…timestamp….{json,csv}`, and
the suite summary sits inside `results/submission_suite/summary/`.

## Sequential workflow (Mac CPU / `sriniii` branch)

1. **Create/activate the virtual environment**
   ```bash
   git checkout -b sriniii  # only needed the first time
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   pip3.11 install --upgrade pip
   pip3.11 install -r requirements.txt
   ```
2. **Build the Rust backend (optional but recommended)**
   ```bash
   cargo build --release --manifest-path rust/Cargo.toml
   ```
3. **Run a single training job and capture JSON + CSV logs**
   ```bash
   python3.11 run_real_training.py \
     --game kuhn_poker \
     --backend rust \
     --iterations 500 \
     --episodes-per-iteration 128 \
     --experts actor,regret,ra,explore,cfr \
     --br-budget 64 \
     --meta-unroll 16 \
     --handoff-tau 0.15 \
     --handoff-patience 3 \
     --state-cluster round+position+pot \
     --manifest-path results/manifest.csv
   ```
   The command produces both `…json` and `…_history.csv` files alongside an
   updated `results/manifest.csv` entry.
4. **Aggregate finished runs into JSON/CSV summaries**
   ```bash
   python3.11 generate_results.py \
     --results-dir results/submission_suite \
     --output results/submission_suite/summary/experiment_summary.json
   ```
5. **Launch the full benchmark sweep (includes evaluation + plots)**
   ```bash
   python3.11 scripts/run_poker_suite.py \
     --output-dir results \
     --backend rust \
     --experiment-name submission_suite
   ```
   Each invocation appends to `results/manifest.csv`, ensuring CSV + JSON logs
   for every run in the suite.

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

- Extend the expert catalogue with specialised domain heads (e.g., bluff
  detectors, variance-aware policies) while retaining the sublinear gate regret
  guarantees.
- Tighten the approximate best-response estimator with batched Rust rollouts
  and cached value functions for larger games.
- Automate multi-game sweeps that compare fixed-λ baselines against the
  anytime-CFR handoff under realistic CPU budgets.
