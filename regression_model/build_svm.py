from __future__ import annotations

import argparse

from regression_model.svm_ensemble import SVMEnsembleRegressionModel
from regression_model.training import DatasetConfig, TrainingConfig, build_model


# Small-sample oriented SVR search space:
# search kernel family, penalty strength, kernel width and epsilon-insensitive loss.
PARAM_GRID = {
    "kernel": ["rbf", "linear", "poly"],
    "C": [0.5, 1.0, 10.0, 50.0],
    "gamma": ["scale", 0.05, 0.1, 0.5],
    "epsilon": [0.01, 0.05, 0.1, 0.2],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build an SVR model with grid search and k-fold CV.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target-col", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-filename", default="svm_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_model(
        model_name="svm",
        estimator=SVMEnsembleRegressionModel(),
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
