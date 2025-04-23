"""Test script for the anomaly detection system.

This script demonstrates the usage of the AnomalyDetector class with the
credit card fraud detection dataset.
"""

import json
import time
from pathlib import Path

import pandas as pd
from anamoly_detection import AnomalyDetector


def save_anomaly_details(
    original_data: pd.DataFrame,
    anomalies: pd.DataFrame,
    drivers: dict,
    output_dir: Path,
) -> None:
    """Save detailed anomaly information to JSON files and export enhanced dataset.

    Args:
        original_data: Original DataFrame before anomaly detection.
        anomalies: DataFrame containing anomaly detection results.
        drivers: Dictionary mapping anomaly indices to their top drivers.
        output_dir: Directory to save the output files.
    """
    output_dir.mkdir(exist_ok=True)

    # Create enhanced dataset with anomaly information
    enhanced_data = original_data.copy()
    enhanced_data["is_anomaly"] = anomalies["is_anomaly"]
    enhanced_data["anomaly_score"] = anomalies["anomaly_score"]
    
    # Add top driver information
    enhanced_data["top_driver_1"] = None
    enhanced_data["top_driver_1_importance"] = None
    enhanced_data["top_driver_2"] = None
    enhanced_data["top_driver_2_importance"] = None
    
    for idx, features in drivers.items():
        if len(features) >= 1:
            enhanced_data.at[idx, "top_driver_1"] = features[0][0]
            enhanced_data.at[idx, "top_driver_1_importance"] = features[0][1]
        if len(features) >= 2:
            enhanced_data.at[idx, "top_driver_2"] = features[1][0]
            enhanced_data.at[idx, "top_driver_2_importance"] = features[1][1]

    # Save enhanced dataset
    enhanced_data.to_csv(output_dir / "enhanced_dataset.csv", index=False)

    # Save all anomalies with their scores
    anomalies.to_csv(output_dir / "all_anomalies.csv", index=False)

    # Save detailed information for each anomaly
    for idx, features in drivers.items():
        anomaly_info = {
            "index": int(idx),
            "anomaly_score": float(anomalies.loc[idx, "anomaly_score"]),
            "top_drivers": [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in features
            ],
            "feature_values": {
                feature: float(anomalies.loc[idx, feature])
                for feature in anomalies.columns
                if feature not in ["anomaly_score", "is_anomaly"]
            },
        }

        # Save individual anomaly details
        with open(output_dir / f"anomaly_{idx}.json", "w") as f:
            json.dump(anomaly_info, f, indent=2)


def main(max_anomalies_to_analyze: int | None = None) -> None:
    """Run the anomaly detection test.

    Args:
        max_anomalies_to_analyze: Maximum number of anomalies to analyze for feature importance.
            If None, analyze all anomalies. Defaults to None.
    """
    # Initialize the detector
    detector = AnomalyDetector(contamination=0.01)  # Set low contamination for fraud detection

    # Load the data
    data_path = Path("data/creditcard.csv")
    if not data_path.exists():
        print("Please run download_data.py first to download the dataset")
        return

    print("Loading data...")
    original_data = pd.read_csv(data_path)
    data = detector.load_data(str(data_path))

    # Detect anomalies
    print("Detecting anomalies...")
    anomalies = detector.detect(data)

    # Print some statistics
    n_anomalies = anomalies["is_anomaly"].sum()
    print(f"\nFound {n_anomalies} anomalies in the dataset")
    print(f"Anomaly rate: {n_anomalies/len(anomalies):.2%}")

    # Get anomaly indices to analyze
    anomaly_indices = anomalies[anomalies["is_anomaly"]].index
    if max_anomalies_to_analyze is not None:
        anomaly_indices = anomaly_indices[:max_anomalies_to_analyze]
        print(f"\nAnalyzing feature importance for first {len(anomaly_indices)} anomalies...")
    else:
        print(f"\nAnalyzing feature importance for all {len(anomaly_indices)} anomalies...")

    # Get top drivers for anomalies
    start_time = time.time()
    drivers = detector.get_top_drivers(anomalies, n_features=2)
    elapsed_time = time.time() - start_time
    print(f"Feature importance analysis completed in {elapsed_time:.2f} seconds")

    # Save detailed results
    print("\nSaving detailed results...")
    save_anomaly_details(original_data, anomalies, drivers, Path("output"))

    # Plot results
    print("\nGenerating visualizations...")
    detector.plot_anomalies(anomalies, drivers)

    print("\nAnalysis complete! Check the output/ directory for detailed results.")


if __name__ == "__main__":
    # You can change this number to analyze more or fewer anomalies
    # Set to None to analyze all anomalies
    main(max_anomalies_to_analyze=10)  # Default to 10 for quick testing
