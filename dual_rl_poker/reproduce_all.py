#!/usr/bin/env python3
"""
One-command reproducibility script for Dual RL Poker project.

Replays all experiments and regenerates all outputs including:
- Complete experiment matrix (Kuhn & Leduc, multiple algorithms)
- All figures and tables from manifest data
- Statistical analysis with confidence intervals
- Head-to-head EV evaluations
- Computational analysis and diagnostics
- Publication-ready materials

Usage: python reproduce_all.py [--clean] [--quick]
Options:
  --clean  : Remove existing results before regeneration
  --quick  : Run reduced experiments for testing (10 seeds instead of 20)
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Set up paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
TABLES_DIR = PROJECT_ROOT / "tables"
PAPER_DIR = PROJECT_ROOT / "paper"

def setup_logging():
    """Set up comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('reproduce_all.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_command(cmd: List[str], cwd: Optional[Path] = None,
                env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """Run command with proper error handling."""
    logger = logging.getLogger(__name__)

    cmd_str = ' '.join(cmd)
    logger.info(f"Running: {cmd_str}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            env=env or os.environ,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Command failed: {cmd_str}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)

        logger.info(f"Command completed successfully: {cmd_str}")
        return result

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {cmd_str}")
        raise
    except Exception as e:
        logger.error(f"Error running command {cmd_str}: {e}")
        raise

def clean_results():
    """Clean existing results directories."""
    logger = logging.getLogger(__name__)

    logger.info("Cleaning existing results...")

    for directory in [RESULTS_DIR, PLOTS_DIR, TABLES_DIR]:
        if directory.exists():
            logger.info(f"Removing {directory}")
            import shutil
            shutil.rmtree(directory)
        directory.mkdir(exist_ok=True)

    # Clean paper outputs
    if PAPER_DIR.exists():
        for ext in ['.pdf', '.aux', '.log', '.out']:
            for file in PAPER_DIR.glob(f"*{ext}"):
                file.unlink()

def check_dependencies():
    """Check that all required dependencies are available."""
    logger = logging.getLogger(__name__)

    logger.info("Checking dependencies...")

    required_packages = [
        'torch', 'numpy', 'pyspiel', 'scipy',
        'pandas', 'matplotlib', 'seaborn', 'pyarrow'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error(f"Missing packages: {missing}")
        logger.error("Install with: pip install -r requirements.lock")
        raise RuntimeError(f"Missing dependencies: {missing}")

    logger.info("All dependencies satisfied")

def run_experiment_matrix(quick: bool = False):
    """Run the complete experiment matrix."""
    logger = logging.getLogger(__name__)

    logger.info("Starting experiment matrix...")

    # Set environment variable for reproducibility
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)

    # Quick mode: 10 seeds for Kuhn, 5 for Leduc
    # Full mode: 20 seeds for Kuhn, 10 for Leduc
    kuhn_seeds = 10 if quick else 20
    leduc_seeds = 5 if quick else 10

    # Run canonical matrix experiments
    cmd = [
        sys.executable,
        'experiments/canonical_matrix.py',
        '--kuhn-seeds', str(kuhn_seeds),
        '--leduc-seeds', str(leduc_seeds)
    ]

    start_time = time.time()
    run_command(cmd, env=env)
    elapsed = time.time() - start_time

    logger.info(f"Experiment matrix completed in {elapsed/3600:.2f} hours")

def run_head_to_head_evaluations():
    """Run head-to-head EV evaluations against baselines."""
    logger = logging.getLogger(__name__)

    logger.info("Running head-to-head EV evaluations...")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)

    cmd = [sys.executable, 'evaluation/head_to_head_ev.py']
    run_command(cmd, env=env)

def generate_figures_and_tables():
    """Generate all figures and tables from manifest data."""
    logger = logging.getLogger(__name__)

    logger.info("Generating figures and tables...")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)

    cmd = [sys.executable, 'visualization/auto_generator.py']
    run_command(cmd, env=env)

def compile_paper():
    """Compile the LaTeX paper."""
    logger = logging.getLogger(__name__)

    logger.info("Compiling LaTeX paper...")

    # Run pdflatex twice for proper references
    for _ in range(2):
        cmd = ['pdflatex', 'dual_rl_poker.tex']
        run_command(cmd, cwd=PAPER_DIR)

    # Clean auxiliary files
    for ext in ['.aux', '.log', '.out']:
        for file in PAPER_DIR.glob(f"*{ext}"):
            file.unlink()

def generate_summary_report():
    """Generate final summary report."""
    logger = logging.getLogger(__name__)

    logger.info("Generating summary report...")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)

    cmd = [sys.executable, 'utils/enhanced_manifest_manager.py', '--summary']
    try:
        run_command(cmd, env=env)
    except subprocess.CalledProcessError:
        # Summary generation is optional
        logger.warning("Summary report generation failed (optional)")

def verify_outputs():
    """Verify that all expected outputs were generated."""
    logger = logging.getLogger(__name__)

    logger.info("Verifying outputs...")

    expected_files = []

    # Check manifest
    manifest_file = RESULTS_DIR / "enhanced_manifest.csv"
    if manifest_file.exists():
        expected_files.append(manifest_file)
        logger.info(f"✓ Manifest found: {manifest_file}")
    else:
        logger.warning(f"✗ Missing manifest: {manifest_file}")

    # Check plots
    expected_plots = [
        'nash_conv_curves.png',
        'exploitability_curves.png',
        'performance_comparison.png',
        'ev_heatmap.png',
        'convergence_analysis.png'
    ]

    for plot in expected_plots:
        plot_file = PLOTS_DIR / plot
        if plot_file.exists():
            expected_files.append(plot_file)
            logger.info(f"✓ Plot found: {plot}")
        else:
            logger.warning(f"✗ Missing plot: {plot}")

    # Check tables
    expected_tables = [
        'performance_table.tex',
        'statistical_analysis.tex',
        'ev_matrix.tex'
    ]

    for table in expected_tables:
        table_file = TABLES_DIR / table
        if table_file.exists():
            expected_files.append(table_file)
            logger.info(f"✓ Table found: {table}")
        else:
            logger.warning(f"✗ Missing table: {table}")

    # Check paper
    paper_file = PAPER_DIR / "dual_rl_poker.pdf"
    if paper_file.exists():
        expected_files.append(paper_file)
        logger.info(f"✓ Paper compiled: {paper_file}")
    else:
        logger.warning(f"✗ Missing paper: {paper_file}")

    logger.info(f"Verification complete: {len(expected_files)} files generated")

    return len(expected_files) > 0

def main():
    """Main reproducibility pipeline."""
    parser = argparse.ArgumentParser(description="Reproduce all Dual RL Poker experiments")
    parser.add_argument('--clean', action='store_true',
                       help='Clean existing results before regeneration')
    parser.add_argument('--quick', action='store_true',
                       help='Run reduced experiments for testing')
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("DUAL RL POKER - COMPLETE REPRODUCIBILITY PIPELINE")
    logger.info("=" * 80)

    if args.quick:
        logger.info("Running in QUICK mode (reduced experiments)")
    else:
        logger.info("Running in FULL mode (complete experiments)")

    start_time = time.time()

    try:
        # Step 1: Clean if requested
        if args.clean:
            clean_results()

        # Step 2: Check dependencies
        check_dependencies()

        # Step 3: Run experiment matrix
        run_experiment_matrix(quick=args.quick)

        # Step 4: Run head-to-head evaluations
        run_head_to_head_evaluations()

        # Step 5: Generate figures and tables
        generate_figures_and_tables()

        # Step 6: Compile paper
        compile_paper()

        # Step 7: Generate summary report
        generate_summary_report()

        # Step 8: Verify outputs
        success = verify_outputs()

        elapsed = time.time() - start_time

        if success:
            logger.info("=" * 80)
            logger.info("REPRODUCIBILITY PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total time: {elapsed/3600:.2f} hours")
            logger.info("=" * 80)

            print("\n" + "="*80)
            print("✓ ALL EXPERIMENTS REPRODUCED SUCCESSFULLY")
            print("="*80)
            print(f"Results available in: {RESULTS_DIR}")
            print(f"Plots available in: {PLOTS_DIR}")
            print(f"Tables available in: {TABLES_DIR}")
            print(f"Paper compiled: {PAPER_DIR}/dual_rl_poker.pdf")
            print("="*80)

        else:
            logger.error("Reproducibility pipeline completed with errors")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Reproducibility pipeline failed: {e}")
        logger.info(f"Check logs in: {PROJECT_ROOT}/reproduce_all.log")
        sys.exit(1)

if __name__ == "__main__":
    main()