# Data Directory

This directory is for storing datasets used in CF-Ensemble experiments.

## Structure

Organize your data as follows:

```
data/
├── raw/              # Original, immutable data
├── processed/        # Cleaned and processed data
├── predictions/      # Base model predictions
├── results/          # Experiment results
└── external/         # External datasets
```

## .gitignore

Note that data files are excluded from version control by default:
- `data/raw/` - Large raw datasets
- `data/processed/` - Processed data files
- `*.pkl`, `*.h5`, `*.hdf5` - Binary data files

## Data Formats

Common formats used:
- **CSV**: Tabular data
- **NPY/NPZ**: NumPy arrays
- **PKL**: Pickled Python objects
- **H5/HDF5**: Large matrices and arrays

## Example Usage

```python
import pandas as pd
import numpy as np

# Load raw data
data = pd.read_csv('data/raw/dataset.csv')

# Load predictions
predictions = np.load('data/predictions/base_models.npz')

# Save processed data
processed_data.to_pickle('data/processed/clean_data.pkl')
```

## Datasets

Document your datasets here:
- Name
- Source
- Size
- Description
- Citation (if applicable)
