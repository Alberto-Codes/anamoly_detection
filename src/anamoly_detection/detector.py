"""Core anomaly detection functionality.

This module provides the main AnomalyDetector class for detecting and analyzing
anomalies in structured data using Isolation Forest and SHAP values.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Main class for anomaly detection and analysis.

    This class provides functionality for:
    - Loading and preprocessing data
    - Detecting anomalies using Isolation Forest
    - Analyzing feature importance using SHAP values
    - Visualizing results

    Attributes:
        contamination: The proportion of outliers in the data set.
        random_state: Random seed for reproducibility.
        scaler: StandardScaler instance for feature scaling.
        model: IsolationForest instance for anomaly detection.
        feature_names: List of feature names from the input data.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42) -> None:
        """Initialize the anomaly detector.

        Args:
            contamination: The proportion of outliers in the data set.
            random_state: Random seed for reproducibility.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )
        self.feature_names: Optional[List[str]] = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess data from a file.

        Args:
            file_path: Path to the data file (CSV, Excel, etc.)

        Returns:
            Preprocessed pandas DataFrame.

        Raises:
            ValueError: If the file format is not supported.
            Exception: If there is an error loading the data.
        """
        try:
            if file_path.endswith(".csv"):
                data = pd.read_csv(file_path)
            elif file_path.endswith((".xls", ".xlsx")):
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            self.feature_names = data.columns.tolist()
            return self._preprocess_data(data)

        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            raise

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data: handle missing values, scale features.

        Args:
            data: Input DataFrame.

        Returns:
            Preprocessed DataFrame.
        """
        data = data.fillna(data.mean())
        scaled_data = self.scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in the data.

        Args:
            data: Preprocessed DataFrame.

        Returns:
            DataFrame with anomaly scores and labels.
        """
        scores = self.model.fit_predict(data)
        results = data.copy()
        results["anomaly_score"] = -self.model.score_samples(data)
        results["is_anomaly"] = scores == -1
        return results

    def get_top_drivers(
        self,
        anomalies: pd.DataFrame,
        n_features: int = 2,
        max_samples: int = 10,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Get top n features contributing to each anomaly using SHAP values.

        Args:
            anomalies: DataFrame with anomaly scores.
            n_features: Number of top features to return.
            max_samples: Maximum number of anomalies to analyze.

        Returns:
            Dictionary mapping anomaly indices to list of (feature, importance) tuples.
        """
        anomaly_data = anomalies[anomalies["is_anomaly"]].drop(
            ["anomaly_score", "is_anomaly"], axis=1
        )

        if len(anomaly_data) > max_samples:
            anomaly_data = anomaly_data.iloc[:max_samples]
            logger.info(
                "Analyzing first %d anomalies out of %d",
                max_samples,
                len(anomalies[anomalies["is_anomaly"]]),
            )

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(anomaly_data)

        top_drivers = {}
        for idx, row_idx in enumerate(
            tqdm(anomaly_data.index, desc="Analyzing anomalies")
        ):
            feature_importance = list(
                zip(self.feature_names, np.abs(shap_values[idx]), strict=False)
            )
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_drivers[row_idx] = feature_importance[:n_features]

        return top_drivers

    def plot_anomalies(
        self,
        anomalies: pd.DataFrame,
        drivers: Dict[int, List[Tuple[str, float]]],
    ) -> None:
        """Visualize anomalies and their top drivers.

        Args:
            anomalies: DataFrame with anomaly scores.
            drivers: Dictionary of top drivers for each anomaly.
        """
        plt.figure(figsize=(12, 6))
        sns.histplot(data=anomalies, x="anomaly_score", hue="is_anomaly")
        plt.title("Distribution of Anomaly Scores")
        plt.show()

        for idx, features in drivers.items():
            plt.figure(figsize=(10, 5))
            features_df = pd.DataFrame(features, columns=["Feature", "Importance"])
            sns.barplot(data=features_df, x="Importance", y="Feature")
            plt.title(f"Top Drivers for Anomaly at Index {idx}")
            plt.show()
