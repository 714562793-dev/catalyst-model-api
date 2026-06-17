from regression_model.gpr import GPRRegressionModel
from regression_model.random_forest import RandomForestRegressionModel
from regression_model.svm_ensemble import SVMEnsembleRegressionModel

try:
    from regression_model.xgb_ensemble import XGBEnsembleRegressionModel, XGB_AVAILABLE
except Exception:
    XGBEnsembleRegressionModel = None
    XGB_AVAILABLE = False

__all__ = [
    "GPRRegressionModel",
    "RandomForestRegressionModel",
    "SVMEnsembleRegressionModel",
    "XGBEnsembleRegressionModel",
    "XGB_AVAILABLE",
]
