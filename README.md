# 🛒 Retail Oracle: Predictive Cart Intelligence & Supply Chain Optimization

An end-to-end Machine Learning pipeline and real-time visualization platform designed to predict which grocery items a customer will reorder in their next session using the **Instacart dataset** (comprising over 3 million transactions).

By anticipating user carts, this project enables e-commerce personalization and drives downstream supply chain optimization, allowing retailers to pre-position high-demand inventory, reduce out-of-stock events, and streamline logistics.

**Repository:** [github.com/shubhamrangdal2001-hash/ML-Project-Instacart-Prediction](https://github.com/shubhamrangdal2001-hash/ML-Project-Instacart-Prediction)

---

## 🚀 Key Features

- **End-to-End ML Pipeline**: Seamless data ingestion, custom feature engineering, model training, evaluation, and inference.
- **Advanced Feature Engineering**: Implements 10+ mathematical features across three granular levels:
  - **User-level**: Purchase frequency, average basket size, reorder ratios, and average days between orders.
  - **Product-level**: Global popularity index, average cart placement position, and conditional reorder probability.
  - **Interaction-level**: User-product purchase counts, recency tracking, first/last purchase indices, and relative order rate.
- **Multi-Model Comparison**: Evaluates baseline models (Logistic Regression) against ensemble classifiers (Random Forest, XGBoost).
- **Temporal Train-Test Splitting**: Time-ordered session histories to avoid data leakage.
- **Serialized Model Package**: Pre-trained `model.pkl` bundles the champion model, feature lists, and global context for sub-second Streamlit and API startup.
- **Interactive Streamlit Dashboard**: Personalized user lookup, real-time feature hydration, threshold tuning, and recommendation visualizer.
- **Cold-Start Fallback**: Routes users without purchase history to global conditional reorder probability matrices.
- **Azure ML Deployment**: Managed online endpoint scoring via `azure_deployment/score.py` with warm-start and cold-start inference modes.
- **MLflow Tracking**: Experiment and model artifacts stored locally (`mlflow.db`, `model_mlflow/`).
- **Local Inference Verification**: `verify_inference.py` exercises the scoring service before cloud deployment.

---

## 📂 Project Structure

```text
Project_ML/
├── data/
│   ├── raw.zip                         # Full Instacart archive (Git LFS)
│   └── raw/                            # Extracted CSVs (large files gitignored)
│       ├── orders.csv
│       ├── order_products__prior.csv
│       ├── products.csv
│       └── ...
├── EDA/                                # Exploratory analysis notebooks & assets
├── Instacart_Predictive_Cart/
│   ├── instacart_pipeline/             # Core ML package
│   │   ├── data_ingestion.py
│   │   ├── main.py
│   │   ├── evaluation/
│   │   ├── feature_engineering/
│   │   └── modeling/
│   ├── app.py                          # Streamlit dashboard
│   ├── serialize_model.py              # Train & export model.pkl
│   ├── model.pkl                       # Serialized model + feature context
│   ├── Final_Business_Report.md
│   └── Retail_Oracle_Phase1_Design.md
├── azure_deployment/                   # Azure ML managed endpoint
│   ├── score.py                        # init() / run() scoring entrypoints
│   ├── deploy_custom.py                # Workspace deploy script
│   ├── conda_env.yml                   # Azure environment spec
│   ├── sample_request.json             # Example REST payload
│   └── job_logs/                       # Build & deployment diagnostics
├── model_mlflow/
│   └── model.pkl                       # MLflow-exported artifact copy
├── mlflow.db                           # Local MLflow tracking store
├── verify_inference.py                 # Local scoring smoke tests
├── ML_Project_Report.md                # Full technical project report
├── walkthrough.md                      # Documentation & delivery notes
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.8+ (3.11 recommended for Azure deployment parity)
- [Git LFS](https://git-lfs.github.com/) (required for `data/raw.zip`)

### 2. Clone the Repository

```bash
git clone https://github.com/shubhamrangdal2001-hash/ML-Project-Instacart-Prediction.git
cd ML-Project-Instacart-Prediction

git lfs install
git lfs pull
```

### 3. Create a Virtual Environment & Install Dependencies

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
pip install streamlit mlflow azure-ai-ml azure-identity
```

### 4. Prepare Data

**Option A — Extract the bundled archive (recommended after clone):**

```bash
# From project root; creates data/raw/*.csv
# Windows (PowerShell):
Expand-Archive -Path data/raw.zip -DestinationPath data -Force
# macOS/Linux:
unzip -o data/raw.zip -d data/
```

**Option B — Download Instacart CSVs manually** from [Kaggle Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data) into `data/raw/`:

- `orders.csv`
- `order_products__prior.csv`
- `products.csv`

> Large transaction files are listed in `.gitignore` to keep the repo lean. Reference files (`products.csv`, `aisles.csv`, `departments.csv`) may already be present; you still need `orders.csv` and `order_products__prior.csv` for training and inference.

### 5. (Optional) Build the Serialized Model Package

If `Instacart_Predictive_Cart/model.pkl` is missing or you want to retrain:

```bash
python Instacart_Predictive_Cart/serialize_model.py
```

This writes `model.pkl` containing the trained model, feature columns, global user/product features, and sampled order history used by the dashboard and Azure scorer.

---

## 📈 Running the System

### Run the Full Training Pipeline

Train models, generate features, evaluate performance, and print business metrics:

```bash
python Instacart_Predictive_Cart/instacart_pipeline/main.py
```

### Run the Streamlit Dashboard

```bash
streamlit run Instacart_Predictive_Cart/app.py
```

When `model.pkl` is present, the app loads the pre-trained package instantly. Otherwise it trains on sampled data at startup (slower).

### Verify Local Scoring (Azure-compatible)

Smoke-test warm-start and cold-start users against `azure_deployment/score.py`:

```bash
python verify_inference.py
```

Example JSON payload (see `azure_deployment/sample_request.json`):

```json
{"user_id": 112, "top_k": 5, "threshold": 0.15}
```

### Deploy to Azure ML (Managed Online Endpoint)

1. Sign in to Azure (`az login`) and ensure your subscription has an ML workspace.
2. Update workspace settings in `azure_deployment/deploy_custom.py` if needed (subscription, resource group, workspace name).
3. Deploy:

```bash
python azure_deployment/deploy_custom.py
```

The deployment registers `model.pkl`, builds the conda environment from `conda_env.yml`, and exposes a REST endpoint. Scoring logic lives in `azure_deployment/score.py` (`init` loads the pickle; `run` accepts JSON with `user_id`, `top_k`, and `threshold`).

---

## 📊 Business Metrics & Evaluation

Unlike standard classifiers that optimize accuracy on highly skewed data, this system prioritizes **Recall@K** and **Precision@K** (where $K=10$):

- **Recall@K**: Of the items the customer actually bought, what fraction appeared in the model's top-$K$ recommendations?
- **Precision@K**: Of the top-$K$ recommendations, how many were truly reordered?
- **AUC-ROC**: Ranking quality of the champion XGBoost model on held-out temporal splits.
- **Supply chain impact**: Localized basket forecasts support inventory pre-positioning 24–48 hours ahead.

For formulas, feature definitions, and architecture diagrams, see **[ML_Project_Report.md](ML_Project_Report.md)**.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [ML_Project_Report.md](ML_Project_Report.md) | In-depth technical report: ingestion, features, models, metrics, dashboard |
| [walkthrough.md](walkthrough.md) | Delivery walkthrough and documentation changelog |
| [Instacart_Predictive_Cart/Final_Business_Report.md](Instacart_Predictive_Cart/Final_Business_Report.md) | Business alignment and tradeoffs |
| [Instacart_Predictive_Cart/Retail_Oracle_Phase1_Design.md](Instacart_Predictive_Cart/Retail_Oracle_Phase1_Design.md) | System design notes |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|----------------|
| **Languages** | Python |
| **Data** | Pandas, NumPy |
| **ML** | XGBoost, Scikit-learn, LightGBM |
| **Serving** | Streamlit, Pickle serialization |
| **MLOps** | MLflow, Azure Machine Learning |
| **Cloud** | Azure Managed Online Endpoints, `azure-ai-ml` |
| **Version control** | Git LFS (dataset archive) |

---

## 📄 License & Data

Instacart dataset usage is subject to the [Kaggle competition terms](https://www.kaggle.com/c/instacart-market-basket-analysis/rules). Do not redistribute raw data beyond those terms.
