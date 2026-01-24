#!/bin/bash
# Run tests in cfensemble environment
# Usage: ./scripts/run_tests.sh [test_file]

set -e

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Check if environment exists
if ! mamba env list | grep -q cfensemble; then
    echo "Error: cfensemble environment not found!"
    echo "Create it with: mamba env create -f environment.yml"
    exit 1
fi

# Run tests using mamba run (no activation needed)
if [ -z "$1" ]; then
    echo "Running all tests..."
    mamba run -n cfensemble pytest tests/ -v
else
    echo "Running tests in $1..."
    mamba run -n cfensemble pytest "$1" -v
fi
