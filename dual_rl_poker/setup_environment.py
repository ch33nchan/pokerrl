#!/usr/bin/env python3
"""
Environment setup script for Dual RL Poker project.

This script creates a reproducible Python environment and verifies all dependencies.
"""

import subprocess
import sys
import venv
from pathlib import Path


def create_venv():
    """Create Python virtual environment."""
    venv_path = Path(__file__).parent / ".venv"
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        return venv_path

    print(f"Creating virtual environment at {venv_path}")
    venv.create(venv_path, with_pip=True)
    return venv_path


def install_dependencies():
    """Install pinned dependencies."""
    print("Installing pinned dependencies...")
    requirements_path = Path(__file__).parent / "requirements.lock"

    # Install torch CPU version first
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ], check=True)

    # Install other dependencies
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "-r", str(requirements_path)
    ], check=True)


def verify_environment():
    """Verify that all dependencies are correctly installed."""
    print("Verifying environment...")

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")

        import numpy
        print(f"✓ NumPy: {numpy.__version__}")

        import scipy
        print(f"✓ SciPy: {scipy.__version__}")

        import pandas
        print(f"✓ Pandas: {pandas.__version__}")

        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")

        import pyspiel
        print(f"✓ OpenSpiel: {pyspiel.version()}")

        import pyarrow
        print(f"✓ PyArrow: {pyarrow.__version__}")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def record_environment_info():
    """Record environment information for reproducibility."""
    import sys
    import platform
    import torch
    import pyspiel

    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "openspiel_version": pyspiel.version(),
        "cuda_available": torch.cuda.is_available(),
    }

    env_path = Path(__file__).parent / "environment_info.json"
    import json
    with open(env_path, 'w') as f:
        json.dump(env_info, f, indent=2)

    print(f"Environment info saved to {env_path}")


def main():
    """Main setup function."""
    print("Setting up Dual RL Poker environment...")

    # Create virtual environment
    venv_path = create_venv()

    # Install dependencies
    install_dependencies()

    # Verify installation
    if not verify_environment():
        print("Environment verification failed!")
        sys.exit(1)

    # Record environment info
    record_environment_info()

    print("Environment setup complete!")
    print("Activate with: source .venv/bin/activate")


if __name__ == "__main__":
    main()