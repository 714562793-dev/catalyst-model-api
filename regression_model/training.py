from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    data_path: str
    feature_cols: tuple[int, int, int, int, int] = (1, 2, 3, 4, 5)
    target_col: int = 6
    test_size: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    n_splits: int = 5
    n_jobs: int = 1
    refit_on_full_data: bool = True


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_noheader_dataset(cfg: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    path = Path(cfg.data_path)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, header=None)
    else:
        df = pd.read_csv(path, header=None)

    X = df.iloc[:, list(cfg.feature_cols)].to_numpy(dtype=float)
    y = df.iloc[:, cfg.target_col].to_numpy(dtype=float)
    return X, y


def ensure_output_dir(path: str) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def build_model(
    *,
    model_name: str,
    estimator,
    param_grid: dict[str, list[Any]],
    dataset_cfg: DatasetConfig,
    train_cfg: TrainingConfig,
    output_dir: str,
    model_filename: str,
) -> dict[str, Any]:
    """
    Shared training workflow for RF, XGBoost, GPR and SVM.

    Steps:
    1. Split the initial dataset into train and test subsets.
    2. Run grid search on the training subset with k-fold CV.
    3. Evaluate the best model with CV RMSE/R2 on the training subset.
    4. Evaluate the best model on the held-out test subset.
    5. Optionally refit the best model on the full dataset and save it.
    """
    X, y = load_noheader_dataset(dataset_cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=dataset_cfg.test_size,
    )

    cv = KFold(
        n_splits=train_cfg.n_splits,
        shuffle=True,
    )
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=train_cfg.n_jobs,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_
    cv_predictions = cross_val_predict(clone(best_estimator), X_train, y_train, cv=cv, n_jobs=train_cfg.n_jobs)
    cv_rmse = _rmse(y_train, cv_predictions)
    cv_r2 = float(r2_score(y_train, cv_predictions))

    best_estimator.fit(X_train, y_train)
    test_predictions = np.asarray(best_estimator.predict(X_test, return_std=False), dtype=float)
    test_rmse = _rmse(y_test, test_predictions)
    test_r2 = float(r2_score(y_test, test_predictions))

    final_model = clone(best_estimator)
    if train_cfg.refit_on_full_data:
        final_model.fit(X, y)
    else:
        final_model.fit(X_train, y_train)

    outdir = ensure_output_dir(output_dir)
    model_path = outdir / model_filename

    summary = {
        "model_name": model_name,
        "dataset_config": asdict(dataset_cfg),
        "training_config": asdict(train_cfg),
        "best_params": grid.best_params_,
        "best_cv_rmse_from_grid": float(-grid.best_score_),
        "cv_rmse": cv_rmse,
        "cv_r2": cv_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "model_path": str(model_path),
    }

    joblib.dump({"model": final_model, "meta": summary}, model_path)
    return summary
