# Legacy Tests (Archived)

These test files are from the original CF-Ensemble implementation and are kept for reference only.

## Files

- `cf_test.py` - Main legacy test suite
- `cf-test.py` - Alternative test format
- `cf_test_confidence.py` - Confidence estimation tests
- `cf_test_opt.py` - Optimization tests
- `cf_test_preference.py` - Preference learning tests
- `cf_test_similarity.py` - Similarity metric tests
- `cf_test_stacker.py` - Stacking method tests
- `cf_topk_item.py` - Top-k recommendation tests

## Why Archived?

These tests depend on:
- `utils_sys` module (not in new implementation)
- Old code structure in `src/cfensemble/models/`
- Legacy APIs that have been replaced

## New Tests

For the modernized implementation, see:
- `../../tests/test_ensemble_data.py`
- `../../tests/test_losses.py`
- `../../tests/test_aggregators.py`
- `../../tests/test_als.py`
- `../../tests/test_trainer.py`

## Relationship to New Implementation

The new implementation (Phase 1-2) replaces the functionality tested here with:
- Cleaner API design
- Better documentation
- More comprehensive tests
- Modern Python practices
- KD-inspired combined objective (the key innovation)

These legacy tests will be useful for:
- Reference when modernizing notebooks (Phase 8)
- Understanding old implementation details
- Comparison of approaches

## Running Legacy Tests

To run these (if needed for reference):
1. Would require restoring old dependencies
2. Would require `utils_sys` module
3. Not recommended - use new tests instead

---

**Status**: Archived for reference, not actively maintained

**See**: `../../tests/README.md` for current test suite
