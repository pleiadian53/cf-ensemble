# Installation Guide

This guide covers installation of the CF-Ensemble project on different platforms.

## Prerequisites

- Python >= 3.10, < 3.13
- [Mamba](https://mamba.readthedocs.io/) or [Conda](https://docs.conda.io/)
- [Poetry](https://python-poetry.org/) >= 1.6.0

## Local Installation (macOS/Apple Silicon)

For local development on macOS with Apple Silicon (M1/M2/M3):

```bash
# Clone the repository
git clone <repository-url>
cd cf-ensemble

# Create and activate environment
mamba env create -f environment.yml
mamba activate cfensemble

# Install package in development mode
poetry install
```

## RunPod/GPU VM Installation

For training on GPU-equipped VMs (e.g., RunPod):

```bash
# Clone the repository
git clone <repository-url>
cd cf-ensemble

# Create and activate environment with CUDA support
mamba env create -f environment-runpod.yml
mamba activate cfensemble

# Install package in development mode
poetry install
```

## Verify Installation

```python
# Test import
import cfensemble
print(cfensemble.__version__)

# Check if PyTorch is available (optional)
from cfensemble.optimization import PYTORCH_AVAILABLE
print(f"PyTorch available: {PYTORCH_AVAILABLE}")

# Run tests
pytest tests/
```

## Optional: PyTorch for GPU Acceleration

PyTorch is now included by default in both environments for the alternative gradient descent optimizer. However, if you need to reinstall or update it:

### macOS (Apple Silicon)
PyTorch is pre-configured to use MPS (Metal Performance Shaders) for GPU acceleration:

```bash
# Already included in environment.yml
# If needed separately:
mamba install pytorch>=2.1 -c conda-forge
```

### RunPod/GPU VMs
PyTorch with CUDA support is pre-configured:

```bash
# Already included in environment-runpod.yml
# Verify CUDA availability:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Comparison: ALS vs PyTorch
Both optimization methods are available:
- **ALS** (default): CPU-based, stable, no extra dependencies
- **PyTorch**: GPU-accelerated, flexible, requires PyTorch

See `docs/methods/als_vs_pytorch.md` for detailed comparison.

## Development Setup

For development work, install additional dev dependencies:

```bash
# Install with dev dependencies
poetry install --with dev

# Set up pre-commit hooks (optional)
poetry run black --check .
poetry run ruff check .
```

## Jupyter Notebook Setup

To use the notebooks:

```bash
# Install Jupyter kernel
python -m ipykernel install --user --name cfensemble --display-name "Python (cfensemble)"

# Start Jupyter
jupyter notebook notebooks/
```

## Troubleshooting

### macOS Issues

If you encounter issues with implicit or surprise packages on macOS:

```bash
# Install Xcode command line tools
xcode-select --install

# Reinstall problematic packages
pip install --no-cache-dir implicit surprise
```

### CUDA Issues on RunPod

If CUDA is not detected:

```bash
# Verify CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
mamba install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Platform-Specific Notes

### Local (macOS M1/M2/M3)
- Uses CPU/MPS acceleration
- Suitable for prototyping and small-scale experiments
- Limited memory (16GB)

### RunPod (NVIDIA GPU)
- Full CUDA support for large-scale training
- Use `environment-runpod.yml` for proper GPU configuration
- Monitor GPU memory usage with `nvidia-smi`
