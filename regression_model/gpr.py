from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class GPRRegressionModel(BaseEstimator, RegressorMixin):
    """Gaussian-process regressor with configurable kernel family."""

    def __init__(
        self,
        *,
        kernel_type: str = "matern",
        length_scale: float = 1.0,
        constant_value: float = 1.0,
        alpha: float = 1e-3,
        matern_nu: float = 1.5,
        use_scaler: bool = True,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 5,
    ):
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.constant_value = constant_value
        self.alpha = alpha
        self.matern_nu = matern_nu
        self.use_scaler = use_scaler
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self._model = None
        self._is_fitted = False

    def _build_kernel(self):
        base = ConstantKernel(constant_value=self.constant_value, constant_value_bounds=(1e-3, 1e3))
        if self.kernel_type == "rbf":
            return base * RBF(length_scale=self.length_scale, length_scale_bounds=(1e-3, 1e3))
        if self.kernel_type == "matern":
            return base * Matern(
                length_scale=self.length_scale,
                length_scale_bounds=(1e-3, 1e3),
                nu=self.matern_nu,
            )
        raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X2 = np.asarray(X, dtype=float)
        y1 = np.asarray(y, dtype=float).reshape(-1)
        gpr = GaussianProcessRegressor(
            kernel=self._build_kernel(),
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
        )
        if self.use_scaler:
            self._model = Pipeline([("scaler", StandardScaler()), ("gpr", gpr)])
        else:
            self._model = gpr
        self._model.fit(X2, y1)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) before predict().")

        X2 = np.asarray(X, dtype=float)
        if return_std:
            mu, std = self._model.predict(X2, return_std=True)
            return np.asarray(mu, dtype=float).reshape(-1), np.clip(np.asarray(std, dtype=float).reshape(-1), 0.0, np.inf)

        mu = self._model.predict(X2)
        return np.asarray(mu, dtype=float).reshape(-1)
