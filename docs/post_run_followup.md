# Post-run follow-up checklist

After running `scripts/run_poker_suite.py` (or individual `run_real_training.py` jobs) locally, follow this sequence to verify artefacts, analyse metrics, and prepare further experiments.

## 1. Inspect aggregated metrics

```bash
python3.11 -m json.tool results/submission_suite/summary/experiment_summary.json | head
```

* Confirms the aggregator captured every run, seed, and configuration.
* Review `summary/experiment_summary.json` for mean, standard deviation, and confidence intervals.

## 2. Review per-run JSON and CSV logs

Each run writes paired files under `results/<experiment>/<game>/<policy>/seed_<n>/`:

* `<timestamp>.json` – configuration, final metrics, and metadata.
* `<timestamp>_history.csv` – per-iteration metrics (exploitability, NashConv, scheduler loss, wall-clock seconds).

Open them for deeper inspection with any spreadsheet or plotting tool. From the shell you can peek at the first rows via:

```bash
head results/submission_suite/kuhn_poker/neural_tabular/seed_0/*_history.csv
```

## 3. Check the global manifest

Every job appends a summary row to `results/manifest.csv`. Use it to compare seeds/games or to import into analysis notebooks.

```bash
python3.11 - <<'PY'
import csv
from itertools import islice

with open('results/manifest.csv', newline='') as fh:
    rows = list(csv.reader(fh))
for row in rows[:1]:
    print(' | '.join(row))
print('-' * 80)
for row in rows[-5:]:
    print(' | '.join(row))
PY
```

If you need a fresh manifest, delete the file and rerun the aggregator (step 4).

## 4. Regenerate summaries after adding new runs

Whenever you collect more runs (different hyperparameters or seeds), rerun the aggregator so the suite-level summaries and plots stay current:

```bash
python3.11 generate_results.py \
  --results-dir results/submission_suite \
  --output results/submission_suite/summary/experiment_summary.json
```

This refreshes `results/experiment_summary.json` and plots under `results/plots/` when invoked through `scripts/run_poker_suite.py`.

## 5. Optional sanity checks

* Verify Rust/OpenSpiel parity:
  ```bash
  python3.11 -m dual_rl_poker.utils.rust_env kuhn_poker
  ```
* Spot-check deterministic behaviour by re-running a seed and confirming identical JSON/CSV outputs.

## 6. Prepare for follow-on experiments

* Duplicate the command with new flags (e.g., `--meta-unroll`, `--br-budget`, `--experts`) to launch alternative configurations.
* Use `--experiment-name` to isolate artefacts per study.
* Keep the `--manifest-path` consistent so all runs append to the same CSV.

Following this checklist keeps artefacts organised, validates that the suite executed correctly, and readies the workspace for additional benchmarks.
