from __future__ import annotations

import numpy as np


def fit_ridge_regression(
    features: np.ndarray,
    targets: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta, covariance) for a ridge-regularized linear fit."""

    if features.size == 0:
        raise ValueError("Cannot fit ridge regression with an empty design matrix.")

    feature_dim = features.shape[1]
    covariance = features.T @ features + ridge * np.eye(feature_dim, dtype=float)
    theta = np.linalg.solve(covariance, features.T @ targets)
    return theta, covariance


def predict_linear_values(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return features @ theta


def confidence_width(
    features: np.ndarray,
    covariance: np.ndarray,
    beta: float,
) -> np.ndarray:
    covariance_inv = np.linalg.inv(covariance)
    widths = np.einsum("nd,df,nf->n", features, covariance_inv, features)
    return beta * np.sqrt(np.maximum(widths, 0.0))
