# Predictive Cart Intelligence System: Final Business Report

## 1. Executive Summary
The predictive cart intelligence model is designed to anticipate which specific items a user will reorder in their next session. 

By leveraging user patterns, product popularity, and user-product interactions, we have constructed an end-to-end ML pipeline capable of generating highly accurate top-K product lists for grocery shoppers.

## 2. Business Metric Alignment
We chose **Recall@K** and **Precision@K** over simple classification Accuracy.
- **Recall@K (K=10)** answers: "If the user organically added 5 things to their cart, how many of those 5 did our model correctly pre-populate in the top 10 recommended slots?"
- **Minimizing False Negatives**: For habitual buyers, missing a staple item creates friction. Recall ensures we cast a net wide enough (Top K slots) to catch their staples without cluttering the UI. Accuracy simply measures overall correct answers (mostly highly skewed towards class 0, as users don't buy 99% of the catalog), which makes it a poor metric for predicting exact active selections.

## 3. Modeling & Trade-offs
- **Baseline (Logistic Regression)**: Fast to compute but fails to capture complex non-linear combinations (like specific day-of-week interactions for a precise product).
- **Random Forest**: Better at capturing non-linear splits but has a massive memory overhead when trees grow deep over millions of rows, becoming unsustainable for Instacart-scale datasets.
- **XGBoost (Chosen Model)**: 
  - **Memory Efficiency**: Can act on sparse feature matrices and handles missing values natively without huge bloated datasets.
  - **Performance**: Provides superior predictive probabilities via gradient boosting.
  - **Interpretability via SHAP**: Can report *why* an item was predicted (e.g., "Usually bought every 7 days, and it's been exactly 6 days since the prior order").

## 4. Error Analysis Insights
- **Cold-Start Users**: Users with fewer than 5 total orders lack sufficient interaction history to build behavioral models. For them, the system must gracefully fall back to calculating `p_reorder_prob` (Global Product Popularity - heavily skewed towards Bananas and Organic Strawberries).
- **Rare Products**: Long-tail items with low `p_total_purchases` struggle to break the probability threshold against frequent staples. This is acceptable for predictive carts, as users usually do not rely on recommendations to search for highly obscure items.

## 5. Deployment & System Architecture Strategy
- **Batch Pipeline (Feature Store)**: Since purchasing patterns evolve with each daily session, features should be updated automatically in a batch pipeline (like PySpark/Databricks) and written to a central Feature Store nightly.
- **Batch Inference Layer**: The XGBoost model performs nightly batch inference for all active users, pre-calculating the probabilities to surface the Top-10 reorder items.
- **Low-Latency Serving**: The pre-calculated Top-10 list is stored in a Redis cache per `user_id`. When the user logs into the app, the frontend hits a microservice API to instantaneous retrieve the list.

## 6. Business Impact
- **Increased Basket Size**: Pre-populating the cart reduces the cognitive load of searching, encouraging users to discover and buy additional/adjacent items.
- **Improved Retention**: Reduces friction in the recurring purchase cycle.
- **Inventory Predictability**: Anticipating user carts with 24-hours notice allows supply chain caching to align highly-ordered products precisely where demand spikes, reducing Out-Of-Stock events.
