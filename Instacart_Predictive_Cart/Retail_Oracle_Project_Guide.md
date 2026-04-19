# 🛒 Retail Oracle: Predictive Cart Intelligence - Project Guide

Welcome to **Retail Oracle**! This guide is designed to help anyone—whether you are a beginner Data Scientist or a new developer joining the team—understand how our predictive cart system works from end to end.

By the end of this document, you will be able to navigate the repository, understand how data turns into smart predictions, run the project locally, and begin contributing!

---

## 1️⃣ PROJECT OVERVIEW

### What problem does this solve?
When customers shop for groceries online, finding the products they want to buy again takes time. We want to predict **which products a customer will add to their cart on their next visit** to reduce friction and improve the shopping experience.

### Business Use-Case
* **Increased Revenue:** By automatically suggesting high-probability reorder items, we increase average cart sizes (upselling/cross-selling).
* **Customer Retention:** A frictionless, personalized shopping experience keeps customers coming back.

### Why it matters
Instead of relying on basic "Global Popularity" (suggesting milk and bread to everyone), this system learns personalized habits (e.g., "User #14 buys organic avocados every 2 weeks").

---

## 2️⃣ HOW THE SYSTEM WORKS (BIG PICTURE)

The architecture follows a standard, modular Machine Learning lifecycle:

`Raw Data` ➡️ `Data Ingestion` ➡️ `Feature Engineering` ➡️ `Model Training` ➡️ `Evaluation` ➡️ `Streamlit UI`

1. **Raw Data:** We start with real historical purchase data (orders, user behavior, product catalogs).
2. **Data Ingestion:** Securely load and clean this massive dataset.
3. **Feature Engineering:** Transform past purchases into actionable math (e.g., "calculate User X's average basket size").
4. **Model Training:** Feed the features into a Machine Learning model (XGBoost) to learn patterns.
5. **Evaluation:** Test the model's accuracy on unseen data using business metrics like Precision@10.
6. **Streamlit UI:** Host an interactive web app where we can input a User ID and see real-time recommendations.

---

## 3️⃣ FILE-BY-FILE EXPLANATION

Here is a breakdown of our workspace:

* **`data/`** 
  * The folder where raw Instacart CSVs are stored (e.g., `orders.csv`, `products.csv`).
* **`instacart_pipeline/data_ingestion.py`**
  * **What it does:** Reads the raw CSVs into Pandas DataFrames efficiently using chunks.
  * **Input/Output:** In = CSV files | Out = Cleaned Pandas DataFrames.
  * **Connection:** Called by almost every other file to get the raw numbers.
* **`instacart_pipeline/eda.py`**
  * **What it does:** Exploratory Data Analysis. Contains logic for visualizing distributions (like "most common hour of day for ordering").
* **`instacart_pipeline/feature_engineering/`**
  * **What it does:** The brain of knowledge creation. Separated into components:
    * `user_features.py`: Stats about the user (total orders, reorder ratios).
    * `product_features.py`: Stats about the item (how often is it reordered globally?).
    * `interaction_features.py`: User-to-Item stats (how often *this user* buys *this item*).
  * **Connection:** Used by `main.py` and `app.py` to prepare the data for the model.
* **`instacart_pipeline/modeling/`**
  * **What it does:** Houses the ML algorithms (`baseline.py`, `advanced_models.py`). Contains Random Forest and XGBoost logic.
  * **Input/Output:** In = Feature DataFrames | Out = Trained Model & Probabilities.
* **`instacart_pipeline/evaluation/`**
  * **What it does:** Measures if the model is actually good. Uses `business_metrics.py` (Recall@K).
* **`instacart_pipeline/main.py`**
  * **What it does:** The orchestrator. It runs the entire backend pipeline (simulate loading -> train -> evaluate metrics) without needing the UI.
* **`app.py`**
  * **What it does:** The Streamlit Frontend. It wraps the model in a beautiful web app.
* **`requirements.txt`**
  * **What it does:** Lists all the Python packages needed to run this project (e.g., `pandas`, `xgboost`, `streamlit`).

---

## 4️⃣ DATA FLOW EXPLANATION

How does a User's history become a Cart Prediction?

1. **Extraction:** `data_ingestion` pulls `orders.csv` and `order_products__prior.csv`.
2. **Transformation:** 
    * `UserFeatures` calculates: `user_id 1 -> 50 total orders`.
    * `ProductFeatures` calculates: `product_id 24 -> 80% global reorder rate`.
    * `InteractionFeatures` calculates: `user 1 + product 24 -> bought 12 times together`.
3. **Merging:** We merge these 3 tables together on `user_id` and `product_id`.
4. **Learning:** We attach a `reordered_target` label (1 if they reordered it, 0 otherwise) and feed this massive table to XGBoost.
5. **Inference:** When asking the UI for a prediction, we create this exact same math for the specific user, but ask XGBoost to output a `Reorder Probability` (0.0 to 1.0) instead of learning.

---

## 5️⃣ HOW TO RUN THE PROJECT

Follow these steps exactly in your terminal:

**1. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
```

**2. Activate Environment**
* Windows: `venv\Scripts\activate`
* Mac/Linux: `source venv/bin/activate`

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run Backend Pipeline Test**
*(This validates your data and trains models to check accuracy metrics)*
```bash
python instacart_pipeline/main.py
```

**5. Launch the Web Application**
```bash
streamlit run app.py
```

---

## 6️⃣ STREAMLIT APP EXPLANATION

**File:** `app.py`

* **What it does:** Provides a UI to select a generic user and view their next predicted cart items.
* **How user input flows:** The user enters a `User ID` on the sidebar. The app asks `app.py::generate_user_cart_predictions()` to pull that user's history and build their feature matrix.
* **The Magic:** The dynamically built feature matrix is fed to our cached XGBoost model (`predict_proba()`). The app filters the items based on the UI's `Threshold` slider, sorts by highest probability, takes the `Top-K`, and displays them in an easy-to-read table and bar chart.
* **Cold Starts:** If a user has NO history, the app gracefully traps this error and recommends globally popular items instead of crashing.

---

## 7️⃣ KEY ML LOGIC EXPLAINED

* **Feature Engineering Strategy:** We rely heavily on historical behavior. The best predictor of future behavior is past behavior. Interaction features (User + Product overlap) carry the most mathematical weight.
* **Model Choice (XGBoost):** We chose XGBoost (Gradient Boosting) because it handles tabular, heterogeneous data extremely well, deals inherently with missing values, and provides feature importance metrics.
* **Evaluation Metrics (Recall@K):** Accuracy is dangerous here! If we predict a user won't buy 99% of the catalog, we are 99% accurate but useless. We use **Precision/Recall at K** (e.g., Recall@10) because a real business only has limited space to show recommendations (K slots). We care about how many of the *actual* reordered items made it into our Top K slots.

---

## 8️⃣ COMMON ERRORS & FIXES

* **🚨 Missing Columns Error (`KeyError: 'user_id'`):**
  * *Fix:* The dataset wasn't merged correctly. Ensure that `InteractionFeatures` successfully output a dataframe before merging `UserFeatures`.
* **🚨 Path Issues (`FileNotFoundError`):**
  * *Fix:* `data_ingestion.py` expects data to be in `../data` or `./data`. If you're running Streamlit from the root folder, make sure your raw CSVs are strictly located inside `/data/raw`.
* **🚨 Model Loading Errors / Memory Crashes:**
  * *Fix:* Instacart data is huge. If Streamlit crashes your machine, the data sampling in `app.py::load_historical_data()` needs to be reduced (e.g., lower `nrows` from 250k to 50k).
* **🚨 Data Mismatch Scenarios:**
  * *Fix:* If you run inference on a user but predict 100% probabilities, you likely suffered Data Leakage. Never include the `reordered_target` in the `X_infer` dataset!

---

## 9️⃣ HOW TO EXTEND THE PROJECT

Want to make the project better? Here's where to start!

1. **Adding New Features:** Add "Time Since Last Order" to `user_features.py` (e.g., are they overdue for milk?). Add Time-of-Day features.
2. **Improving Model:** Swap XGBoost for LightGBM for faster training, or try a sequential Deep Learning representation (like RNNs/Transformers) for time-series forecasting.
3. **Improving UI:** Add actual product images to the Streamlit table. Show the user's *past* cart next to their *predicted* cart for visual proof.
4. **Scaling System:** Move feature engineering to an orchestration tool like Apache Airflow, push features to a Feature Store (Redis), and wrap the model in a FastAPI endpoint so any mobile app could hit it.

---

## 🔟 BEST PRACTICES

As you contribute, please adhere to these standards:

* **Modular Coding:** Keep functions distinct. Do not combine loading data and training a model in one massive function.
* **Consistent Schema:** Always ensure DataFrames returning from feature modules have the exact same column naming conventions to prevent merge failures.
* **Proper Logging:** Use Python's `logging` module (seen in `main.py`). Avoid `print()` statements. We need to know timestamps and severity levels (INFO vs ERROR) for production monitoring.
* **Version Control:** Commit often to Git. Make isolated branches for experimental models (e.g., `git checkout -b feature/lightgbm-model`).

---

**Happy Predicting! 🛒🚀**
