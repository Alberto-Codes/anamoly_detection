"""
Anomaly Detection System for structured data with feature importance analysis.
"""

__version__ = "0.1.0"

from .detector import AnomalyDetector

__all__ = ["AnomalyDetector"]

def main() -> None:
    print("Hello from anamoly-detection!")
