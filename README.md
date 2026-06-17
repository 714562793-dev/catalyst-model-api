# 催化剂模型训练 API - 部署指南

## 部署到 Render（推荐）

### 方式一：Render Blueprint（一键部署，推荐）

#### 步骤 1：推送到 GitHub

```bash
# 1. 在 GitHub 上创建新仓库（例如：catalyst-model-api）
# 2. 添加远程仓库并推送
git remote add origin https://github.com/你的用户名/catalyst-model-api.git
git branch -M main
git push -u origin main
```

#### 步骤 2：在 Render 上创建 Blueprint

1. 登录 [Render Dashboard](https://dashboard.render.com/)
2. 点击 **New +** → **Blueprint**
3. 连接你的 GitHub 账号，选择 `catalyst-model-api` 仓库
4. Render 会自动读取 `render.yaml` 配置并创建服务
5. 等待部署完成（约 2-3 分钟）
6. 获取服务 URL（如 `https://catalyst-model-api.onrender.com`）

### 方式二：手动创建 Web Service

1. 登录 [Render Dashboard](https://dashboard.render.com/)
2. 点击 **New +** → **Web Service**
3. 连接你的 GitHub 仓库
4. 配置如下：
   - **Name**: `catalyst-model-api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}`
   - **Environment Variable**: `PYTHON_VERSION = 3.11.0`
5. 点击 **Create Web Service**

### 步骤 3：配置前端

在前端 **页面 3（模型训练）** 的 **Python 后端 API 地址** 输入框中填入：

```
https://catalyst-model-api.onrender.com
```

然后勾选 **使用 Python 后端** 进行训练。

---

## 文件结构

```
├── main.py                  # FastAPI 入口
├── Procfile                 # Render 启动命令
├── render.yaml              # Render Blueprint 配置
├── requirements.txt         # Python 依赖
├── regression_model/        # 模型实现
│   ├── __init__.py
│   ├── gpr.py              # GPR 回归模型
│   ├── random_forest.py    # 随机森林模型
│   ├── svm_ensemble.py     # SVM 集成模型
│   ├── xgb_ensemble.py   # XGBoost 集成模型
│   ├── training.py         # 训练工具
│   └── build_*.py          # 构建脚本
└── .gitignore
```

---

## API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/train` | 训练模型 |
| POST | `/predict` | 预测候选性能 |

---

## 本地测试

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

访问 `http://localhost:8000/docs` 查看 Swagger 文档。
