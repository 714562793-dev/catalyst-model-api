from __future__ import annotations

import argparse

from regression_model.random_forest import RandomForestRegressionModel
from regression_model.training import DatasetConfig, TrainingConfig, build_model


# Small-sample oriented RF search space:
# keep trees reasonably large, but regularize via depth / split / leaf settings.
PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 3, 5, 7],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [1, 2, 3],
    "max_features": [1.0, "sqrt", 0.8],
    "bootstrap": [True],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a random-forest regression model with grid search and k-fold CV.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target-col", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-filename", default="random_forest_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_model(
        model_name="random_forest",
        estimator=RandomForestRegressionModel(),
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
