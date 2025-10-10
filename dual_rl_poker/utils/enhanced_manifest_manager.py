"""
Enhanced manifest manager with complete metadata tracking.

Single source of truth for all experimental runs with exact specifications:
- Complete schema matching executive directive requirements
- Version tracking (Python, OpenSpiel, torch, commit hash)
- Exact evaluation metrics (NashConv, exploitability, EV)
- Computational cost analysis (FLOPs, parameter counts)
- Diagnostic coverage tracking
- CSV export with all required fields
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
import subprocess
import logging
from datetime import datetime
import platform


@dataclass
class RunMetadata:
    """Complete metadata for a single experimental run."""
    # Core identification
    run_id: str
    config_hash: str
    timestamp: str
    start_time: str
    end_time: str

    # Experiment configuration
    game: str
    traversal: str
    method: str
    seed: int
    iterations: int
    eval_every: int

    # Network architecture
    params_count: int
    flops_est: int
    hidden_dims: List[int]
    network_type: str

    # Optimization configuration
    optimizer_cfg: Dict[str, Any]
    replay_cfg: Dict[str, Any]
    update_cadence: int

    # Environment and versions
    python_version: str
    openspiel_version: str
    torch_version: str
    commit_hash: str
    platform_info: Dict[str, str]

    # Performance metrics (exact evaluation)
    final_exploitability: float
    final_nashconv: float
    best_exploitability: float
    best_nashconv: float
    mean_value: float

    # Convergence metrics
    steps_to_threshold: int  # T (e.g., Kuhn exploitability ≤ 0.1)
    time_to_threshold: float  # Time to reach threshold
    wall_clock_s: float

    # Head-to-head EV results
    ev_vs_tabular_cfr: float
    ev_vs_deep_cfr: float
    ev_vs_sd_cfr: float
    ev_std_error: float

    # Diagnostic coverage
    diagnostics_coverage: Dict[str, bool]
    gradient_norms_logged: bool
    advantage_stats_logged: bool
    policy_kl_logged: bool
    clipping_events_logged: bool
    timing_logged: bool

    # Statistical analysis
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    confidence_level: float
    holm_bonferroni_corrected: bool

    # Training dynamics
    final_training_loss: float
    training_convergence: bool
    nash_conv_spikes: int
    avg_gradient_norm: float

    # Computational analysis
    memory_usage_mb: float
    cpu_utilization: float
    training_flops: int
    evaluation_flops: int

    # Reproducibility
    deterministic_cudnn: bool
    random_seed_set: bool
    environment_frozen: bool


class EnhancedManifestManager:
    """
    Enhanced manifest manager for complete experimental tracking.

    Single source of truth with all required metadata per executive directive.
    """

    def __init__(self, output_path: str = "results"):
        """Initialize enhanced manifest manager.

        Args:
            output_path: Path to store manifest files
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Manifest storage
        self.manifest_path = self.output_path / "enhanced_manifest.csv"
        self.schema_path = self.output_path / "manifest_schema.json"

        # Initialize manifest file if it doesn't exist
        self._initialize_manifest()

        # Environment information
        self.env_info = self._collect_environment_info()

    def _initialize_manifest(self):
        """Initialize manifest file with proper schema."""
        if not self.manifest_path.exists():
            # Create empty manifest with headers
            df = pd.DataFrame(columns=[field.name for field in RunMetadata.__dataclass_fields__.values()])
            df.to_csv(self.manifest_path, index=False)
            self.logger.info(f"Created enhanced manifest at {self.manifest_path}")

        # Save schema
        schema = {field.name: field.type.__name__ for field in RunMetadata.__dataclass_fields__.values()}
        with open(self.schema_path, 'w') as f:
            json.dump(schema, f, indent=2)

    def _collect_environment_info(self) -> Dict[str, str]:
        """Collect complete environment information.

        Returns:
            Dictionary with environment details
        """
        env_info = {}

        # Python version
        env_info['python_version'] = platform.python_version()

        # Platform information
        env_info['platform'] = platform.platform()
        env_info['processor'] = platform.processor()
        env_info['architecture'] = platform.architecture()[0]

        # OpenSpiel version
        try:
            import pyspiel
            env_info['openspiel_version'] = pyspiel.version()
        except ImportError:
            env_info['openspiel_version'] = 'unknown'

        # PyTorch version
        try:
            import torch
            env_info['torch_version'] = torch.__version__
            env_info['cuda_available'] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                env_info['cuda_version'] = torch.version.cuda
        except ImportError:
            env_info['torch_version'] = 'unknown'

        # Git commit hash
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                env_info['commit_hash'] = result.stdout.strip()
            else:
                env_info['commit_hash'] = 'unknown'
        except (subprocess.SubprocessError, FileNotFoundError):
            env_info['commit_hash'] = 'unknown'

        # Additional environment variables
        import os
        env_info['cwd'] = os.getcwd()
        env_info['path'] = os.environ.get('PATH', '')[:200]  # Truncated for readability

        return env_info

    def log_experiment(self, **kwargs) -> str:
        """Log an experimental run with complete metadata.

        Args:
            **kwargs: All metadata fields for the run

        Returns:
            Run ID for the logged experiment
        """
        # Generate run ID if not provided
        run_id = kwargs.get('run_id')
        if not run_id:
            run_id = self._generate_run_id(kwargs)

        # Ensure required fields are present
        required_fields = [field.name for field in RunMetadata.__dataclass_fields__.values()]
        for field in required_fields:
            if field not in kwargs:
                kwargs[field] = self._get_default_value(field)

        # Add environment information
        kwargs.update(self.env_info)

        # Create metadata object
        metadata = RunMetadata(**kwargs)

        # Validate metadata
        self._validate_metadata(metadata)

        # Load existing manifest
        df = pd.read_csv(self.manifest_path)

        # Append new entry
        new_row = pd.DataFrame([asdict(metadata)])
        df = pd.concat([df, new_row], ignore_index=True)

        # Save updated manifest
        df.to_csv(self.manifest_path, index=False)

        self.logger.info(f"Logged experiment {run_id} to enhanced manifest")
        return run_id

    def _generate_run_id(self, config: Dict[str, Any]) -> str:
        """Generate unique run ID from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Unique run ID
        """
        # Create hash from key configuration parameters
        key_fields = ['game', 'algorithm', 'seed', 'iterations', 'hidden_dims']
        config_str = json.dumps({k: config.get(k) for k in key_fields}, sort_keys=True)
        hash_suffix = hashlib.sha256(config_str.encode()).hexdigest()[:8]

        # Create readable ID
        algorithm = config.get('algorithm', 'unknown')
        game = config.get('game', 'unknown')
        seed = config.get('seed', 0)

        run_id = f"{algorithm}_{game}_{seed}_{hash_suffix}"
        return run_id

    def _get_default_value(self, field_name: str) -> Any:
        """Get default value for a field.

        Args:
            field_name: Name of the field

        Returns:
            Default value
        """
        defaults = {
            'timestamp': datetime.now().isoformat(),
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'traversal': 'external_sampling',
            'params_count': 0,
            'flops_est': 0,
            'hidden_dims': [],
            'network_type': 'mlp',
            'optimizer_cfg': {},
            'replay_cfg': {},
            'update_cadence': 1,
            'iterations': 0,
            'eval_every': 100,
            'final_exploitability': float('inf'),
            'final_nashconv': float('inf'),
            'best_exploitability': float('inf'),
            'best_nashconv': float('inf'),
            'mean_value': 0.0,
            'steps_to_threshold': -1,
            'time_to_threshold': -1.0,
            'wall_clock_s': 0.0,
            'ev_vs_tabular_cfr': 0.0,
            'ev_vs_deep_cfr': 0.0,
            'ev_vs_sd_cfr': 0.0,
            'ev_std_error': 0.0,
            'bootstrap_ci_lower': 0.0,
            'bootstrap_ci_upper': 0.0,
            'confidence_level': 0.95,
            'holm_bonferroni_corrected': False,
            'final_training_loss': 0.0,
            'training_convergence': False,
            'nash_conv_spikes': 0,
            'avg_gradient_norm': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_utilization': 0.0,
            'training_flops': 0,
            'evaluation_flops': 0,
            'deterministic_cudnn': False,
            'random_seed_set': False,
            'environment_frozen': False,
            'diagnostics_coverage': {},
            'gradient_norms_logged': False,
            'advantage_stats_logged': False,
            'policy_kl_logged': False,
            'clipping_events_logged': False,
            'timing_logged': False
        }

        return defaults.get(field_name, None)

    def _validate_metadata(self, metadata: RunMetadata):
        """Validate metadata for completeness and correctness.

        Args:
            metadata: Metadata to validate
        """
        warnings = []

        # Check required fields
        if not metadata.run_id:
            warnings.append("Missing run_id")
        if not metadata.game:
            warnings.append("Missing game")
        if not metadata.method:
            warnings.append("Missing method")
        if metadata.seed < 0:
            warnings.append(f"Invalid seed: {metadata.seed}")
        if metadata.iterations <= 0:
            warnings.append(f"Invalid iterations: {metadata.iterations}")

        # Check evaluation metrics
        if metadata.final_exploitability == float('inf'):
            warnings.append("Missing final exploitability")
        if metadata.final_nashconv == float('inf'):
            warnings.append("Missing final NashConv")

        # Check computational tracking
        if not metadata.params_count:
            warnings.append("Missing parameter count")
        if not metadata.flops_est:
            warnings.append("Missing FLOPs estimate")

        # Check diagnostic coverage
        required_diagnostics = ['gradient_norms_logged', 'advantage_stats_logged',
                              'policy_kl_logged', 'timing_logged']
        for diag in required_diagnostics:
            if not getattr(metadata, diag):
                warnings.append(f"Missing diagnostic: {diag}")

        if warnings:
            self.logger.warning(f"Metadata validation warnings for {metadata.run_id}: {warnings}")

    def get_experiment(self, run_id: str) -> Optional[RunMetadata]:
        """Get experiment metadata by run ID.

        Args:
            run_id: Run identifier

        Returns:
            Experiment metadata or None if not found
        """
        df = pd.read_csv(self.manifest_path)
        row = df[df['run_id'] == run_id]

        if row.empty:
            return None

        # Convert row to RunMetadata object
        metadata_dict = row.iloc[0].to_dict()
        return RunMetadata(**metadata_dict)

    def get_experiments_by_filter(self, **filters) -> List[RunMetadata]:
        """Get experiments matching specified filters.

        Args:
            **filters: Filter criteria (e.g., game='kuhn_poker', method='deep_cfr')

        Returns:
            List of matching experiments
        """
        df = pd.read_csv(self.manifest_path)

        # Apply filters
        for key, value in filters.items():
            if key in df.columns:
                df = df[df[key] == value]

        # Convert to RunMetadata objects
        experiments = []
        for _, row in df.iterrows():
            metadata_dict = row.to_dict()
            experiments.append(RunMetadata(**metadata_dict))

        return experiments

    def generate_summary_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive summary report from manifest.

        Args:
            output_path: Path to save report (optional)

        Returns:
            Path to generated report
        """
        df = pd.read_csv(self.manifest_path)

        if df.empty:
            report = "No experiments found in manifest."
        else:
            report = self._generate_report_content(df)

        if output_path is None:
            output_path = self.output_path / "manifest_summary.txt"

        with open(output_path, 'w') as f:
            f.write(report)

        self.logger.info(f"Generated summary report at {output_path}")
        return str(output_path)

    def _generate_report_content(self, df: pd.DataFrame) -> str:
        """Generate report content from DataFrame.

        Args:
            df: Manifest DataFrame

        Returns:
            Report content as string
        """
        report = []
        report.append("ENHANCED EXPERIMENT MANIFEST SUMMARY")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total Experiments: {len(df)}")
        report.append("")

        # Summary by algorithm
        report.append("EXPERIMENTS BY ALGORITHM:")
        report.append("-" * 30)
        alg_counts = df['method'].value_counts()
        for alg, count in alg_counts.items():
            report.append(f"  {alg}: {count} experiments")
        report.append("")

        # Summary by game
        report.append("EXPERIMENTS BY GAME:")
        report.append("-" * 30)
        game_counts = df['game'].value_counts()
        for game, count in game_counts.items():
            report.append(f"  {game}: {count} experiments")
        report.append("")

        # Performance summary
        report.append("PERFORMANCE SUMMARY:")
        report.append("-" * 30)
        valid_metrics = df[df['final_exploitability'] != float('inf')]
        if not valid_metrics.empty:
            report.append(f"Best Exploitability: {valid_metrics['final_exploitability'].min():.6f}")
            report.append(f"Mean Exploitability: {valid_metrics['final_exploitability'].mean():.6f}")
            report.append(f"Worst Exploitability: {valid_metrics['final_exploitability'].max():.6f}")
        else:
            report.append("No valid exploitability metrics found")
        report.append("")

        # Computational summary
        report.append("COMPUTATIONAL SUMMARY:")
        report.append("-" * 30)
        if 'params_count' in df.columns:
            report.append(f"Total Parameters: {df['params_count'].sum():,}")
            report.append(f"Mean Parameters: {df['params_count'].mean():,.0f}")
        if 'wall_clock_s' in df.columns:
            report.append(f"Total Training Time: {df['wall_clock_s'].sum():.1f}s")
            report.append(f"Mean Training Time: {df['wall_clock_s'].mean():.1f}s")
        report.append("")

        # Diagnostic coverage
        report.append("DIAGNOSTIC COVERAGE:")
        report.append("-" * 30)
        diag_columns = ['gradient_norms_logged', 'advantage_stats_logged', 'policy_kl_logged', 'timing_logged']
        for col in diag_columns:
            if col in df.columns:
                coverage = df[col].sum() / len(df) * 100
                report.append(f"  {col}: {coverage:.1f}%")

        return "\n".join(report)

    def export_for_publication(self, output_path: str):
        """Export manifest data formatted for publication.

        Args:
            output_path: Path to save publication-ready data
        """
        df = pd.read_csv(self.manifest_path)

        # Select publication-relevant columns
        pub_columns = [
            'run_id', 'game', 'method', 'seed', 'iterations', 'eval_every',
            'params_count', 'flops_est', 'final_exploitability', 'final_nashconv',
            'best_exploitability', 'steps_to_threshold', 'time_to_threshold',
            'wall_clock_s', 'ev_vs_tabular_cfr', 'ev_vs_deep_cfr', 'ev_vs_sd_cfr',
            'bootstrap_ci_lower', 'bootstrap_ci_upper', 'confidence_level'
        ]

        # Filter available columns
        available_columns = [col for col in pub_columns if col in df.columns]
        pub_df = df[available_columns]

        # Save publication data
        pub_df.to_csv(output_path, index=False)
        self.logger.info(f"Exported {len(pub_df)} experiments for publication to {output_path}")

    def validate_manifest_integrity(self) -> List[str]:
        """Validate manifest integrity and completeness.

        Returns:
            List of validation warnings
        """
        df = pd.read_csv(self.manifest_path)
        warnings = []

        if df.empty:
            warnings.append("Manifest is empty")
            return warnings

        # Check for required columns
        required_columns = [field.name for field in RunMetadata.__dataclass_fields__.values()]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            warnings.append(f"Missing required columns: {missing_columns}")

        # Check for missing critical data
        if 'final_exploitability' in df.columns:
            infinite_exploitability = (df['final_exploitability'] == float('inf')).sum()
            if infinite_exploitability > 0:
                warnings.append(f"{infinite_exploitability} experiments missing final exploitability")

        if 'run_id' in df.columns:
            duplicate_runs = df['run_id'].duplicated().sum()
            if duplicate_runs > 0:
                warnings.append(f"{duplicate_runs} duplicate run IDs found")

        # Check diagnostic coverage
        diag_columns = ['gradient_norms_logged', 'advantage_stats_logged', 'policy_kl_logged', 'timing_logged']
        for col in diag_columns:
            if col in df.columns:
                missing_diagnostics = (~df[col]).sum()
                if missing_diagnostics > 0:
                    warnings.append(f"{missing_diagnostics} experiments missing {col}")

        return warnings