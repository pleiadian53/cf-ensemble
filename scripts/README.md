# Scripts

This directory contains setup, testing, training, and analysis scripts for CF-ensemble experiments.

## Setup Scripts (`setup/`)

Environment and dependency management:

- `setup/fix_environment.sh`: Quick fix to install/update pip packages in existing environment

**Usage**:
```bash
# Update pip packages without recreating environment
./scripts/setup/fix_environment.sh
```

## Testing Scripts

Development and testing utilities:

- `run_tests.sh`: Convenience script to run tests in the cfensemble environment

**Usage**:
```bash
# Run all tests
./scripts/run_tests.sh

# Or run specific test file
./scripts/run_tests.sh tests/test_ensemble_data.py
```

## Training Scripts

Scripts for running ensemble generation and CF-based ensemble learning:

- `step1_generate.py`: Generate base model predictions
- `step1a_generate.py`: Alternative data generation approach
- `step2_order.py`: Order and prepare predictions
- `step2a_consolidate.py`: Consolidate prediction results
- `step3_baselines.py`: Run baseline ensemble methods
- `step4_CES_ens.py`: CF-Ensemble learning
- `step4_CES_run.py`: Execute CF-Ensemble pipeline

## Configuration Scripts

- `cf_spec.py`: CF-Ensemble specifications and configurations
- `cf_run.py`: Run CF-based ensemble experiments

## Ensemble Methods

- `CES_ens.py`: Collaborative Ensemble System implementation
- `CES_run.py`: CES execution pipeline
- `RL_ens.py`: Reinforcement learning-based ensemble

## Data Generation

- `generate.py`: Data generation utilities
- `generate_data.py`: Generate synthetic datasets for experiments

## Usage

```bash
# Example: Generate base model predictions
python scripts/step1_generate.py --config configs/experiment.yaml

# Run CF-Ensemble
python scripts/step4_CES_run.py --input predictions/ --output results/
```

See individual scripts for detailed usage and parameters.
