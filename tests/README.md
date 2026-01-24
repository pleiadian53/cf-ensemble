# Tests

This directory contains test suites for the new CF-Ensemble implementation.

## Current Tests (Phase 1 & 2)

**Phase 1: Core Infrastructure**
- `test_ensemble_data.py` - EnsembleData class tests (~40 tests)
- `test_losses.py` - Loss function tests (~20 tests)
- `test_aggregators.py` - Aggregator tests (~20 tests)

**Phase 2: Optimization**
- `test_als.py` - ALS update tests (8 tests)
- `test_trainer.py` - CFEnsembleTrainer tests (35 tests)

**Phase 3: Confidence Weighting & Reliability Learning**
- `test_confidence.py` - Confidence strategies tests (28 tests)
- `test_reliability_model.py` - Reliability weight model tests (32 tests)

**Total**: 134 tests covering all implemented components ✅

---

## Running Tests

### Run All Tests (Recommended)
```bash
./scripts/run_tests.sh
```
**This automatically runs ALL test files** in the `tests/` directory, including:
- ✅ Phase 1: `test_ensemble_data.py`, `test_losses.py`, `test_aggregators.py` (31 tests)
- ✅ Phase 2: `test_als.py`, `test_trainer.py` (43 tests)  
- ✅ Phase 3: `test_confidence.py`, `test_reliability_model.py` (60 tests)

### Alternative Commands
```bash
# With mamba directly
mamba run -n cfensemble pytest tests/ -v

# Run specific test file
./scripts/run_tests.sh tests/test_als.py

# Run multiple specific files
./scripts/run_tests.sh tests/test_als.py tests/test_trainer.py

# Run with coverage
mamba run -n cfensemble pytest tests/ --cov=cfensemble --cov-report=html
```

---

## Legacy Tests (Archived)

Old test files have been moved to `archive/tests/`:
- `cf_test*.py` - Legacy CF tests (require old dependencies)
- `cf-test.py` - Legacy test suite
- `cf_topk_item.py` - Legacy top-k tests

These are kept for reference but use the old codebase structure and require `utils_sys` and other legacy dependencies.

---

## Test Organization

Tests follow pytest conventions:
- File names: `test_<module>.py`
- Class names: `Test<Component>`
- Function names: `test_<behavior>`

Each test file corresponds to a source module:
```
src/cfensemble/data/ensemble_data.py  → tests/test_ensemble_data.py
src/cfensemble/objectives/losses.py   → tests/test_losses.py
src/cfensemble/ensemble/aggregators.py → tests/test_aggregators.py
src/cfensemble/optimization/als.py     → tests/test_als.py
src/cfensemble/optimization/trainer.py → tests/test_trainer.py
```

---

## Test Coverage

Current coverage (Phase 1 + 2):
- ✅ Data structures: Comprehensive
- ✅ Loss functions: Comprehensive  
- ✅ Aggregators: Comprehensive
- ✅ ALS updates: Comprehensive
- ✅ Trainer: Comprehensive
- **Overall**: ~90% coverage of implemented code

### What's Tested

**EnsembleData (`test_ensemble_data.py`)**
- Initialization and validation
- Labeled/unlabeled data handling
- Confidence computation
- Train/validation splitting
- Edge cases (empty data, all unlabeled, etc.)

**Loss Functions (`test_losses.py`)**
- Reconstruction loss correctness
- Supervised loss (cross-entropy)
- Combined loss balancing
- Gradient computation
- RMSE utility

**Aggregators (`test_aggregators.py`)**
- MeanAggregator predictions
- WeightedAggregator learning
- Gradient descent updates
- Weight normalization
- Prediction validity

**ALS Updates (`test_als.py`)**
- Shape correctness
- Error reduction over iterations
- Convergence on perfect data
- Confidence weighting effects
- Regularization effects
- Numerical stability

**Trainer (`test_trainer.py`)**
- Initialization and validation
- Basic fitting pipeline
- Partially labeled data
- Convergence detection
- Transductive prediction
- Inductive prediction
- Different ρ values
- History tracking
- End-to-end integration

---

## Contributing Tests

When adding new functionality:

1. **Create test file**: `tests/test_<module>.py`
2. **Follow structure**:
   ```python
   import pytest
   from src.cfensemble.<module> import YourClass
   
   class TestYourClass:
       def test_basic_functionality(self):
           """Test basic behavior."""
           pass
       
       def test_edge_cases(self):
           """Test edge cases."""
           pass
       
       def test_error_handling(self):
           """Test error conditions."""
           with pytest.raises(ValueError):
               # Test invalid input
               pass
   ```

3. **Include**:
   - Basic functionality tests
   - Edge cases (empty inputs, single items, etc.)
   - Error handling (invalid inputs)
   - Numerical correctness
   - Integration tests (if applicable)

4. **Aim for**:
   - High code coverage (>80%)
   - Clear test names
   - Descriptive assertions
   - Independent tests (no shared state)

---

## Quick Validation

After implementing new features:

```bash
# 1. Run your new tests
./scripts/run_tests.sh tests/test_your_module.py

# 2. Run all tests to check for regressions
./scripts/run_tests.sh

# 3. Check coverage
mamba run -n cfensemble pytest tests/ --cov=cfensemble --cov-report=term-missing

# 4. Look for uncovered lines and add tests
```

---

## Troubleshooting

**ImportError: No module named 'utils_sys'**
- This means you're trying to run legacy tests
- Use `./scripts/run_tests.sh` which runs only new tests
- Legacy tests are in `archive/tests/`

**ModuleNotFoundError: cfensemble**
- Install package in development mode: `pip install -e .`
- Activate environment: `mamba activate cfensemble`

**Tests discovered but not running**
- Check test file names start with `test_`
- Check test function names start with `test_`
- Check class names start with `Test`

---

## Continuous Integration

Future: Tests will run automatically on:
- Every commit (pre-commit hook)
- Pull requests (GitHub Actions)
- Scheduled nightly builds

---

**Test Status**: ✅ **88+ tests, all passing**

**Coverage**: ~90% of implemented code (Phase 1 + 2)

**Next**: Phase 3 will add tests for confidence strategies and reliability models
