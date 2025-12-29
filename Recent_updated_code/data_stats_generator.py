# data_stats_generator.py

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_data_stats(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Compute generic statistics for any dataset.
    Returns a JSON-serializable dict.
    """
    stats: Dict[str, Any] = {}

    # Basic info
    stats["n_samples"] = int(X.shape[0])
    stats["n_features"] = int(X.shape[1])
    stats["feature_names"] = list(X.columns)

    # Per-feature stats
    feature_stats = {}
    for col in X.columns:
        series = X[col]
        feature_stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "skew": float(series.skew()),
            "pct_zero": float((series == 0).mean()),
            "is_constant": bool(series.nunique(dropna=True) <= 1),
        }
    stats["features"] = feature_stats

    # Target stats
    stats["target"] = {
        "name": getattr(y, "name", "y"),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y, ddof=0)),
        "skew": float(pd.Series(y).skew()),
    }

    # Simple correlation matrix (if not too large)
    if X.shape[1] <= 50:
        corr = X.corr().fillna(0.0)
        stats["correlation_matrix"] = corr.to_dict()

    return stats