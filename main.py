from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
import warnings
import copy

warnings.filterwarnings('ignore')

from regression_model.gpr import GPRRegressionModel
from regression_model.random_forest import RandomForestRegressionModel
from regression_model.svm_ensemble import SVMEnsembleRegressionModel
try:
    from regression_model.xgb_ensemble import XGBEnsembleRegressionModel
    XGB_AVAILABLE = True
except Exception:
    XGBEnsembleRegressionModel = None
    XGB_AVAILABLE = False

app = FastAPI(title="Catalyst Model Training API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 请求 / 响应模型 ──────────────────────────────────────────────

class DataPoint(BaseModel):
    composition: Dict[str, float]
    onsetPotential: Optional[float] = None
    currentDensity: Optional[float] = None

class TrainRequest(BaseModel):
    dataset: List[DataPoint]
    target: str                     # 'onset' | 'current'
    trainRatio: float = 0.8
    validationMethod: str = 'kfold' # 'kfold' | 'holdout' | 'loo'
    kFolds: int = 5

class ModelResultItem(BaseModel):
    modelName: str
    r2: float
    rmse: float
    cvMeanR2: Optional[float] = None
    cvStdR2: Optional[float] = None
    bestParams: Optional[Dict[str, Any]] = None

class TrainResponse(BaseModel):
    results: List[ModelResultItem]
    nTrain: int
    nTest: int
    elements: List[str]

class PredictRequest(BaseModel):
    compositions: List[Dict[str, float]]
    elements: List[str]
    target: str

class PredictResponse(BaseModel):
    predictions: List[float]
    stds: Optional[List[float]] = None

# ── 参数网格（针对小样本和 Railway 免费版优化，减少搜索空间）──────────────────────

GPR_PARAM_GRID = {
    "kernel_type": ["rbf"],
    "length_scale": [0.5, 2.0],
    "constant_value": [1.0],
    "alpha": [1e-3],
}

RF_PARAM_GRID = {
    "n_estimators": [100],
    "max_depth": [5, None],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": [1.0],
}

SVM_PARAM_GRID = {
    "kernel": ["rbf"],
    "C": [1.0, 10.0],
    "gamma": ["scale"],
    "epsilon": [0.1],
}

XGB_PARAM_GRID = {
    "n_estimators": [100],
    "max_depth": [2, 4],
    "learning_rate": [0.1],
    "min_child_weight": [1.0],
}

# ── 训练函数 ──────────────────────────────────────────────────────

class TrainedModelStore:
    """内存存储训练好的模型，用于后续预测。"""
    def __init__(self):
        self.models = {}  # key: f"{target}_{model_name}"
    
    def save(self, target: str, model_name: str, model):
        self.models[f"{target}_{model_name}"] = copy.deepcopy(model)
    
    def get(self, target: str, model_name: str):
        return self.models.get(f"{target}_{model_name}")

model_store = TrainedModelStore()

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train_single_model(target, model_name, estimator, param_grid, X_train, y_train, X_test, y_test, n_splits=5):
    """
    对齐用户 Python 代码的 build_model 流程：
    1. GridSearchCV 在训练集上搜索最优参数
    2. cross_val_predict 在训练集上计算 CV R²
    3. 最优模型在测试集上评估
    4. 全量 refit
    """
    try:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 1. GridSearchCV
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=True,
        )
        grid.fit(X_train, y_train)
        
        best_estimator = grid.best_estimator_
        
        # 2. cross_val_predict on training set (对齐 Python 的 cv_r2)
        cv_predictions = cross_val_predict(clone(best_estimator), X_train, y_train, cv=cv, n_jobs=1)
        cv_r2 = float(r2_score(y_train, cv_predictions))
        
        # 3. Test evaluation
        best_estimator.fit(X_train, y_train)
        test_pred = best_estimator.predict(X_test)
        test_r2 = float(r2_score(y_test, test_pred))
        test_rmse = _rmse(y_test, test_pred)
        
        # 4. Refit on full data
        final_model = clone(best_estimator)
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        final_model.fit(X_full, y_full)
        
        # 保存模型用于后续预测
        model_store.save(target, model_name, final_model)
        
        return {
            "modelName": model_name,
            "r2": round(test_r2, 4),
            "rmse": round(test_rmse, 4),
            "cvMeanR2": round(cv_r2, 4),
            "cvStdR2": None,  # cross_val_predict 给出的是总 R²，不是各 fold 的
            "bestParams": grid.best_params_,
        }, final_model
    except Exception as e:
        print(f"{model_name} training error: {e}")
        return {
            "modelName": model_name,
            "r2": 0.0,
            "rmse": 999.0,
            "cvMeanR2": None,
            "cvStdR2": None,
            "bestParams": {},
        }, None

# ── API 端点 ─────────────────────────────────────────────────────

@app.post("/train", response_model=TrainResponse)
async def train_models(req: TrainRequest):
    # 1. 过滤有效数据
    valid = [
        d for d in req.dataset
        if (req.target == 'onset' and d.onsetPotential is not None) or
           (req.target == 'current' and d.currentDensity is not None)
    ]
    if len(valid) < 4:
        raise HTTPException(400, "至少需要 4 条有效数据（含目标值）")
    
    # 2. 收集元素，固定顺序
    elements = sorted(set().union(*(d.composition.keys() for d in valid)))
    
    # 3. 构建 X / y
    X = np.array([[d.composition.get(el, 0.0) for el in elements] for d in valid])
    y = np.array([
        d.onsetPotential if req.target == 'onset' else d.currentDensity
        for d in valid
    ])
    
    # 4. 划分训练/测试集
    if req.validationMethod == 'loo' or len(X) < req.kFolds + 1:
        test_size = 0.2
    else:
        test_size = 1 - req.trainRatio
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=max(0.1, test_size), random_state=42
    )
    
    # 5. 确定 CV 折数
    n_splits = min(req.kFolds, len(X_train)) if req.validationMethod == 'kfold' else 5
    
    # 6. 训练各模型
    models_config = [
        ("GPR", GPRRegressionModel(), GPR_PARAM_GRID),
        ("RF", RandomForestRegressionModel(), RF_PARAM_GRID),
        ("SVM", SVMEnsembleRegressionModel(), SVM_PARAM_GRID),
    ]
    if XGB_AVAILABLE:
        models_config.insert(1, ("XGBoost", XGBEnsembleRegressionModel(), XGB_PARAM_GRID))
    
    results = []
    for name, estimator, param_grid in models_config:
        result, model = train_single_model(req.target, name, estimator, param_grid, X_train, y_train, X_test, y_test, n_splits)
        results.append(ModelResultItem(**result))
        # 保存模型到内存
        if model is not None:
            model_store.save(req.target, name, model)
    
    # 按 R² 从高到低排序
    results.sort(key=lambda r: r.r2, reverse=True)
    
    return TrainResponse(
        results=results,
        nTrain=len(X_train),
        nTest=len(X_test),
        elements=elements,
    )

@app.post("/predict")
async def predict(req: PredictRequest):
    """用训练好的模型预测候选集的性能。"""
    # 先尝试获取该 target 下已保存的模型
    model = None
    for name in ["GPR", "XGBoost", "RF", "SVM"]:
        model = model_store.get(req.target, name)
        if model is not None:
            break
    
    if model is None:
        raise HTTPException(400, f"没有找到 {req.target} 的训练模型，请先训练模型")
    
    X = np.array([[c.get(el, 0.0) for el in req.elements] for c in req.compositions])
    
    try:
        if hasattr(model, 'predict'):
            # 检查是否支持 return_std
            try:
                preds, stds = model.predict(X, return_std=True)
                return PredictResponse(
                    predictions=[float(p) for p in preds],
                    stds=[float(s) for s in stds]
                )
            except TypeError:
                # 不支持 return_std
                preds = model.predict(X)
                return PredictResponse(
                    predictions=[float(p) for p in preds],
                    stds=None
                )
        else:
            raise HTTPException(500, "模型没有 predict 方法")
    except Exception as e:
        raise HTTPException(500, f"预测错误: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "models": ["GPR", "XGBoost", "RF", "SVM"], "version": "2.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
