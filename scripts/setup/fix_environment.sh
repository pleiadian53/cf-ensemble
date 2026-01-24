#!/bin/bash
# Quick fix for pip packages in existing cfensemble environment
# Usage: ./scripts/setup/fix_environment.sh

set -e

echo "Updating cfensemble environment..."
echo ""

# Install/update conda packages (including PyTorch)
echo "1. Installing/updating conda packages..."
mamba install -n cfensemble -y pytorch>=2.1 -c conda-forge

# Install/update pip packages
echo ""
echo "2. Installing/updating pip packages..."
mamba run -n cfensemble pip install \
    scikit-surprise \
    tensorboard>=2.15 \
    mlflow>=2.10 \
    black>=24.0 \
    ruff>=0.3 \
    mypy>=1.8

echo ""
echo "✅ Done! Environment updated."
echo ""
echo "Verify PyTorch installation:"
echo "  mamba activate cfensemble"
echo "  python -c 'import torch; print(f\"PyTorch {torch.__version__} installed\")'"
echo ""
echo "Next steps:"
echo "  pip install -e ."
echo "  ./scripts/run_tests.sh"
