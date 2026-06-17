from __future__ import annotations

import argparse

from regression_model.gpr import GPRRegressionModel
from regression_model.training import DatasetConfig, TrainingConfig, build_model


# Small-sample oriented GPR search space:
# focus on kernel family, characteristic length scale, signal amplitude and noise level.
PARAM_GRID = {
    "kernel_type": ["rbf", "matern"],
    "length_scale": [0.05, 0.1, 0.3, 1.0, 3.0],
    "constant_value": [0.3, 1.0, 3.0],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "matern_nu": [0.5, 1.5, 2.5],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Gaussian-process regression model with grid search and k-fold CV.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target-col", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-filename", default="gpr_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_model(
        model_name="gpr",
        estimator=GPRRegressionModel(),
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
