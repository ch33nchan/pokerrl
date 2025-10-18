"""Single source of truth manifest manager for experimental results.

Tracks all experimental runs with exact metrics, configurations, and reproducibility
information to ensure scientific rigor and reproducibility.
"""

import csv
import json
import hashlib
import time
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

try:  # Optional heavy dependencies used only for analysis helpers
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is optional
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore

logger = logging.getLogger(__name__)


class ManifestManager:
    """Manages the experimental results manifest with full reproducibility tracking."""

    def __init__(self, manifest_path: str = "results/manifest.csv"):
        """Initialize manifest manager.

        Args:
            manifest_path: Path to the manifest CSV file
        """
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Define manifest columns
        self.columns = [
            'algorithm', 'game', 'config_hash', 'seed', 'iteration',
            'nash_conv', 'exploitability', 'wall_clock_time', 'final_reward',
            'parameters', 'flops_per_forward', 'training_flops', 'model_size_mb',
            'timestamp', 'run_id', 'notes'
        ]

        # Initialize manifest if it doesn't exist
        if not self.manifest_path.exists():
            self._initialize_manifest()

        logger.info(f"Initialized manifest manager at {self.manifest_path}")

    def _initialize_manifest(self):
        """Create empty manifest with headers."""
        with open(self.manifest_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
        logger.info("Created new manifest file")

    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate unique hash for configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Unique hash string
        """
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _generate_run_id(self, algorithm: str, seed: int) -> str:
        """Generate unique run ID.

        Args:
            algorithm: Algorithm name
            seed: Random seed

        Returns:
            Unique run ID
        """
        timestamp = int(time.time())
        hash_input = f"{algorithm}_{seed}_{timestamp}"
        run_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        return f"run_{timestamp}_{run_hash}"

    def log_experiment(self,
                      algorithm: str,
                      game: str,
                      config: Dict[str, Any],
                      seed: int,
                      iteration: int,
                      nash_conv: float,
                      exploitability: float,
                      wall_clock_time: float,
                      final_reward: float,
                      parameters: int,
                      flops_per_forward: int,
                      training_flops: int,
                      model_size_mb: float,
                      notes: str = "") -> str:
        """Log experimental results to manifest.

        Args:
            algorithm: Algorithm name
            game: Game name
            config: Algorithm configuration
            seed: Random seed
            iteration: Training iteration
            nash_conv: Final NashConv value
            exploitability: Final exploitability value
            wall_clock_time: Total wall clock time in seconds
            final_reward: Final reward
            parameters: Number of model parameters
            flops_per_forward: FLOPs per forward pass
            training_flops: Total training FLOPs
            model_size_mb: Model size in MB
            notes: Optional notes

        Returns:
            Generated run ID
        """
        # Generate identifiers
        config_hash = self._generate_config_hash(config)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        run_id = self._generate_run_id(algorithm, seed)

        # Create entry
        entry = {
            'algorithm': algorithm,
            'game': game,
            'config_hash': config_hash,
            'seed': seed,
            'iteration': iteration,
            'nash_conv': nash_conv,
            'exploitability': exploitability,
            'wall_clock_time': wall_clock_time,
            'final_reward': final_reward,
            'parameters': parameters,
            'flops_per_forward': flops_per_forward,
            'training_flops': training_flops,
            'model_size_mb': model_size_mb,
            'timestamp': timestamp,
            'run_id': run_id,
            'notes': notes
        }

        # Write to manifest
        with open(self.manifest_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(entry)

        logger.info(f"Logged experiment {run_id}: {algorithm} on {game}")
        return run_id

    def get_experiments(self,
                       algorithm: Optional[str] = None,
                       game: Optional[str] = None,
                       config_hash: Optional[str] = None) -> pd.DataFrame:
        """Get experiments from manifest with optional filtering.

        Args:
            algorithm: Filter by algorithm name
            game: Filter by game name
            config_hash: Filter by configuration hash

        Returns:
            DataFrame of matching experiments
        """
        if pd is None:  # pragma: no cover - depends on optional pandas install
            raise ImportError(
                "pandas is required for ManifestManager.get_experiments(); "
                "install pandas or read the CSV manually."
            )

        df = pd.read_csv(self.manifest_path)

        # Apply filters
        if algorithm:
            df = df[df['algorithm'] == algorithm]
        if game:
            df = df[df['game'] == game]
        if config_hash:
            df = df[df['config_hash'] == config_hash]

        return df

    def get_best_experiment(self,
                           algorithm: Optional[str] = None,
                           game: Optional[str] = None,
                           metric: str = 'exploitability') -> Dict[str, Any]:
        """Get best experiment by specified metric.

        Args:
            algorithm: Filter by algorithm name
            game: Filter by game name
            metric: Metric to optimize (lower is better)

        Returns:
            Dictionary with best experiment info
        """
        df = self.get_experiments(algorithm=algorithm, game=game)

        if df.empty:
            return {}

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in manifest")

        # Get row with minimum metric value
        best_idx = df[metric].idxmin()
        best_experiment = df.loc[best_idx].to_dict()

        logger.info(f"Best {algorithm} experiment on {game}: {metric}={best_experiment[metric]:.6f}")
        return best_experiment

    def compare_algorithms(self,
                          algorithms: List[str],
                          game: str,
                          metric: str = 'exploitability') -> pd.DataFrame:
        """Compare algorithms on specified game and metric.

        Args:
            algorithms: List of algorithm names to compare
            game: Game name
            metric: Metric to compare

        Returns:
            Comparison DataFrame
        """
        if pd is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "pandas is required for ManifestManager.compare_algorithms(); "
                "install pandas to use this helper."
            )

        comparison_data = []

        for algorithm in algorithms:
            experiments = self.get_experiments(algorithm=algorithm, game=game)
            if not experiments.empty:
                # Get best performance
                best_idx = experiments[metric].idxmin()
                best = experiments.loc[best_idx]

                # Get statistics
                stats = {
                    'algorithm': algorithm,
                    'best_' + metric: best[metric],
                    'best_iteration': best['iteration'],
                    'mean_' + metric: experiments[metric].mean(),
                    'std_' + metric: experiments[metric].std(),
                    'num_runs': len(experiments)
                }
                comparison_data.append(stats)

        return pd.DataFrame(comparison_data)

    def generate_summary_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive summary report.

        Args:
            output_path: Optional path to save report

        Returns:
            Report string
        """
        if pd is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "pandas is required for ManifestManager.generate_summary_report(); "
                "install pandas to use this helper."
            )

        df = pd.read_csv(self.manifest_path)

        if df.empty:
            return "No experiments found in manifest."

        lines = []
        lines.append("Experimental Results Summary")
        lines.append("=" * 50)
        lines.append(f"Total experiments: {len(df)}")
        lines.append(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        lines.append("")

        # Algorithm breakdown
        lines.append("Algorithm Breakdown:")
        algo_counts = df['algorithm'].value_counts()
        for algorithm, count in algo_counts.items():
            lines.append(f"  {algorithm}: {count} experiments")
        lines.append("")

        # Game breakdown
        lines.append("Game Breakdown:")
        game_counts = df['game'].value_counts()
        for game, count in game_counts.items():
            lines.append(f"  {game}: {count} experiments")
        lines.append("")

        # Best performance by algorithm and game
        lines.append("Best Performance by Algorithm and Game:")
        for (algorithm, game), group in df.groupby(['algorithm', 'game']):
            best_idx = group['exploitability'].idxmin()
            best = group.loc[best_idx]
            lines.append(f"  {algorithm} on {game}: exploitability={best['exploitability']:.6f} (iter {best['iteration']})")
        lines.append("")

        # Performance comparison
        lines.append("Performance Comparison (lower exploitability is better):")
        for game in df['game'].unique():
            lines.append(f"\n{game}:")
            game_data = df[df['game'] == game]
            for algorithm in game_data['algorithm'].unique():
                algo_data = game_data[game_data['algorithm'] == algorithm]
                best = algo_data.loc[algo_data['exploitability'].idxmin()]
                mean_exp = algo_data['exploitability'].mean()
                std_exp = algo_data['exploitability'].std()
                lines.append(f"  {algorithm}: {best['exploitability']:.6f} (best), {mean_exp:.6f}±{std_exp:.6f} (mean±std)")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Summary report saved to {output_path}")

        return report

    def export_for_paper(self, output_path: str):
        """Export results formatted for paper inclusion.

        Args:
            output_path: Path to save formatted results
        """
        if pd is None or np is None:  # pragma: no cover - optional dependencies
            raise ImportError(
                "pandas and numpy are required for ManifestManager.export_for_paper(); "
                "install them to use this helper."
            )

        df = pd.read_csv(self.manifest_path)

        if df.empty:
            logger.warning("No experiments to export")
            return

        # Create paper-ready summary
        paper_data = []

        for (algorithm, game), group in df.groupby(['algorithm', 'game']):
            # Get best performance
            best_idx = group['exploitability'].idxmin()
            best = group.loc[best_idx]

            # Calculate statistics
            mean_exp = group['exploitability'].mean()
            std_exp = group['exploitability'].std()
            n_runs = len(group)

            entry = {
                'Algorithm': algorithm.replace('_', '-'),
                'Game': game.replace('_', '-'),
                'Best Exploitability': best['exploitability'],
                'Mean Exploitability': mean_exp,
                'Std Error': std_exp / np.sqrt(n_runs),
                'Best Iteration': best['iteration'],
                'Parameters': best['parameters'],
                'Training FLOPs': best['training_flops'],
                'Model Size (MB)': best['model_size_mb'],
                'Number of Runs': n_runs
            }
            paper_data.append(entry)

        paper_df = pd.DataFrame(paper_data)
        paper_df.to_csv(output_path, index=False)
        logger.info(f"Paper-ready results exported to {output_path}")

    def validate_manifest(self) -> List[str]:
        """Validate manifest for consistency and completeness.

        Returns:
            List of validation warnings
        """
        if pd is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "pandas is required for ManifestManager.validate_manifest(); "
                "install it to perform validation."
            )

        df = pd.read_csv(self.manifest_path)
        warnings = []

        # Check for missing values
        for col in self.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                warnings.append(f"Column '{col}' has {missing} missing values")

        # Check for duplicate run IDs
        duplicate_runs = df['run_id'].duplicated().sum()
        if duplicate_runs > 0:
            warnings.append(f"Found {duplicate_runs} duplicate run IDs")

        # Check exploitability monotonicity (should generally decrease)
        for (algorithm, game, seed), group in df.groupby(['algorithm', 'game', 'seed']):
            if len(group) > 1:
                group_sorted = group.sort_values('iteration')
                exp_increases = (group_sorted['exploitability'].diff() > 0.01).sum()
                if exp_increases > len(group) * 0.1:  # Allow some noise
                    warnings.append(f"{algorithm} on {game} (seed {seed}): {exp_increases} exploitability increases")

        if not warnings:
            logger.info("Manifest validation passed")
        else:
            logger.warning(f"Manifest validation found {len(warnings)} issues")

        return warnings
