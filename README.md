# 🛒 Retail Oracle: Predictive Cart Intelligence & Supply Chain Optimization

An end-to-end Machine Learning pipeline and real-time visualization platform designed to predict which grocery items a customer will reorder in their next session using the **Instacart dataset** (comprising over 3 million transactions). 

By anticipating user carts, this project enables e-commerce personalization and drives downstream supply chain optimization, allowing retailers to pre-position high-demand inventory, reduce out-of-stock events, and streamline logistics.

---

## 🚀 Key Features

- **End-to-End ML Pipeline**: Seamless data ingestion, custom feature engineering, model training, evaluation, and inference.
- **Advanced Feature Engineering**: Implements 10+ mathematical features covering three granular levels:
  - **User-level**: Purchase frequency, average basket size, reorder ratios, and average days between orders.
  - **Product-level**: Global popularity index, average cart placement position, and conditional reorder probability.
  - **Interaction-level**: User-product purchase counts, recency tracking, first/last purchase indices, and relative order rate.
- **Multi-Model Comparison**: Evaluates and compares baseline models (Logistic Regression) against advanced ensemble classifiers (Random Forest, XGBoost).
- **Temporal Train-Test Splitting**: Validates performance without data leakage by using time-ordered session histories.
- **Interactive Streamlit Dashboard**: Provides personalized user lookup, real-time feature hydration, interactive threshold tuning, and recommendation visualizer.
- **Cold-Start Fallback Handler**: Gracefully routes users without historical purchase history to global conditional reorder probability matrices.

---

## 📂 Project Structure

```directory
Project_ML/
├── data/                                 # Raw and processed datasets (ignored by Git)
│   └── raw/
│       ├── orders.csv
│       ├── order_products__prior.csv
│       ├── products.csv
│       └── ...
├── EDA/                                  # Exploratory Data Analysis notebooks & visual assets
│   ├── instacart_eda_features.ipynb
│   └── eda_overview.png
├── Instacart_Predictive_Cart/            # Main application directory
│   ├── instacart_pipeline/               # Core machine learning package
│   │   ├── data_ingestion.py             # Instacart data loader with memory management
│   │   ├── main.py                       # Pipeline orchestration script
│   │   ├── evaluation/
│   │   │   └── business_metrics.py       # Custom Precision@K & Recall@K metrics
│   │   ├── feature_engineering/
│   │   │   ├── user_features.py          # User behavioral profiles
│   │   │   ├── product_features.py       # Global product popularity & reorders
│   │   │   └── interaction_features.py   # User-product synergistic features
│   │   └── modeling/
│   │       ├── baseline.py               # Logistic Regression model
│   │       └── advanced_models.py        # Random Forest & XGBoost classifiers
│   ├── app.py                            # Interactive Streamlit application
│   ├── Final_Business_Report.md          # Business alignment & tradeoff documentation
│   └── Retail_Oracle_Phase1_Design.md    # Low-level and high-level architecture designs
├── requirements.txt                      # Project dependencies
└── README.md                             # Project documentation (this file)
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Clone the Repository & Initialize Environment
```bash
# Navigate to project directory
cd Project_ML

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Place Data Files
Ensure your Instacart CSV files are placed under the standard raw data path:
`data/raw/` or `Instacart_Predictive_Cart/data/raw/`
- `orders.csv`
- `order_products__prior.csv`
- `products.csv`

---

## 📈 Running the System

### Run the Pipeline
To train models, generate features, evaluate performance, and run business metric reports:
```bash
python Instacart_Predictive_Cart/instacart_pipeline/main.py
```

### Run the Interactive Streamlit Dashboard
To spin up the web client dashboard for user prediction lookups and visualizations:
```bash
streamlit run Instacart_Predictive_Cart/app.py
```

---

## 📊 Business Metrics & Evaluation

Unlike standard classifiers that optimize general accuracy on highly skewed datasets, this system prioritizes **Recall@K** and **Precision@K** (where $K=10$):

- **Recall@K**: Answers: *"Of the items the customer organically bought, what percentage did our model successfully recommend in the top $K$ slots?"* This directly minimizes friction for repeat grocery purchases.
- **AUC-ROC**: Measures the champion XGBoost model's capability to rank actual positive reorders higher than negative alternatives.
- **Downstream Supply Chain Impact**: By predicting localized demand baskets 24-48 hours in advance, retail operators can cache products, reduce delivery turnarounds, and optimize storage allocation dynamically.

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: XGBoost, Scikit-learn
- **Dashboard / Frontend**: Streamlit
- **Logging & Verification**: Python logging framework
