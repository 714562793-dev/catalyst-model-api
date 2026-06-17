from __future__ import annotations

import argparse

from regression_model.training import DatasetConfig, TrainingConfig, build_model
from regression_model.xgb_ensemble import XGBEnsembleRegressionModel


# Small-sample oriented XGBoost search space:
# shallow trees + stronger regularization usually outperform aggressive boosting.
PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [2, 4],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [1.0, 3.0],
    "gamma": [0.0, 0.2],
    "reg_alpha": [0.0, 0.2],
    "reg_lambda": [1.0, 5.0],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build an XGBoost regression model with grid search and k-fold CV.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target-col", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-filename", default="xgboost_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_model(
        model_name="xgboost",
        estimator=XGBEnsembleRegressionModel(),
        param_grid=PARAM_GRID,
        dataset_cfg=DatasetConfig(
            data_path=args.data_path,
            target_col=args.target_col,
            test_size=args.test_size,
        ),
        train_cfg=TrainingConfig(
            n_splits=args.n_splits,
        ),
        output_dir=args.output_dir,
        model_filename=args.model_filename,
    )
    print(summary)


if __name__ == "__main__":
    main()
