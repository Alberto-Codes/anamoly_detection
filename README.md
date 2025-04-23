# Anomaly Detection System

A robust Python-based system for detecting anomalies in structured data, with feature importance analysis using SHAP values.

## Overview

This package provides tools for:
- Cleaning and preprocessing structured data
- Detecting anomalies in records using Isolation Forest
- Identifying key features/drivers contributing to anomalies using SHAP values
- Generating detailed reports and visualizations
- Handling messy and inconsistent data formats

## Features

- **Data Cleaning**
  - Automatic handling of missing values
  - Data type conversion and validation
  - Feature scaling and standardization

- **Anomaly Detection**
  - Isolation Forest algorithm
  - Configurable contamination rate
  - Batch processing support

- **Feature Analysis**
  - SHAP value computation for feature importance
  - Top-2 driver identification for each anomaly
  - Interactive visualization of feature contributions

- **Output Generation**
  - CSV file with all detected anomalies
  - Individual JSON files for each analyzed anomaly
  - Visualizations of anomaly distributions and drivers

## Installation

```bash
# Install the package in development mode
uv run pip install -e .
```

## Usage

1. Download the test dataset:
```bash
uv run python scripts/download_data.py
```

2. Run the anomaly detection:
```bash
uv run python scripts/test_anomaly_detection.py
```

### Output Format

The system generates the following outputs in the `output/` directory:

1. `all_anomalies.csv`
   - Contains all detected anomalies with their scores
   - Includes original feature values and anomaly labels

2. Individual JSON files (e.g., `anomaly_1281.json`)
   ```json
   {
     "index": 1281,
     "anomaly_score": 0.59,
     "top_drivers": [
       {
         "feature": "Amount",
         "importance": 1.19
       },
       {
         "feature": "V20",
         "importance": 0.89
       }
     ],
     "feature_values": {
       "Time": -1.98,
       "V1": -1.51,
       // ... all feature values
     }
   }
   ```

3. Visualizations
   - Distribution of anomaly scores
   - Feature importance plots for each anomaly

## Requirements

- Python >= 3.12
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- shap >= 0.42.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tqdm >= 4.66.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
