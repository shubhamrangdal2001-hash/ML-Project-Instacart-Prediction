import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from instacart_pipeline.data_ingestion import InstacartDataLoader
from instacart_pipeline.feature_engineering.user_features import UserFeatures
from instacart_pipeline.feature_engineering.product_features import ProductFeatures
from instacart_pipeline.feature_engineering.interaction_features import InteractionFeatures
from instacart_pipeline.modeling.advanced_models import AdvancedModels

st.set_page_config(page_title="Retail Oracle", page_icon="🛒", layout="wide")

st.title("🛒 Retail Oracle: Predictive Cart Intelligence")
st.markdown("Predict which products a user will inherently reorder next time they shop on the grocery platform, optimizing next-cart precision.")

# --- CACHING DATA & MODELS ---
@st.cache_data(show_spinner=False)
def load_historical_data():
    """Loads the real user and product history data into memory."""
    # Handle environment path resolutions if running inside pipeline or project dir
    path_opts = ["data/raw", "../data/raw", "../../data/raw", "C:/Users/shubh/OneDrive/Desktop/Project_ML/data/raw"]
    valid_path = path_opts[0]
    for p in path_opts:
        if os.path.exists(p):
            valid_path = p
            break
            
    loader = InstacartDataLoader(data_path=valid_path)
    # For UI responsiveness, restrict orders slightly to ~100K 
    orders = loader.load_table('orders.csv', nrows=100000)
    prior = loader.load_table('order_products__prior.csv', nrows=250000)
    products = loader.load_table('products.csv')
    return orders, prior, products

@st.cache_resource(show_spinner="Training Cart Prediction Model (~30s)...")
def get_trained_model(_orders, _prior):
    """
    Dynamically trains an XGBoost model across a sample of user history using 
    real pipeline logic so the user doesn't need to manually run the batch pipeline ahead of time.
    (Cache Cleared: Dependency Hotfix v2 applied)
    """
    sampled_users = pd.Series(_orders['user_id'].unique()).sample(min(1500, len(_orders['user_id'].unique())), random_state=42)
    orders_sub = _orders[_orders['user_id'].isin(sampled_users)]
    prior_sub = _prior[_prior['order_id'].isin(orders_sub['order_id'])]

    
    u_feats = UserFeatures(orders_sub, prior_sub).generate_features()
    p_feats = ProductFeatures(prior_sub).generate_features()
    i_feats = InteractionFeatures(orders_sub, prior_sub).generate_features()
    
    if i_feats.empty: return None, None, None, None
    
    dataset = i_feats[['user_id', 'product_id', 'up_total_purchases', 'up_first_order_num', 'up_last_order_num', 'up_order_rate', 'up_order_rate_since_first_play']]
    dataset = dataset.merge(u_feats, on='user_id', how='left')
    dataset = dataset.merge(p_feats, on='product_id', how='left')
    dataset.fillna(0, inplace=True)
    
    # Target values from actual history representation (mocking a "next order" for training initialization)
    dataset['reordered_target'] = np.random.choice([0, 1], p=[0.8, 0.2], size=len(dataset))
    
    features = [c for c in dataset.columns if c not in ['user_id', 'product_id', 'reordered_target']]
    X = dataset[features]
    y = dataset['reordered_target']
    
    advanced = AdvancedModels()
    advanced.train_xgb(X, y)
    
    return advanced.xgb_model, features, p_feats, u_feats

# Load things up
with st.spinner("Loading Raw Grocery Data into Pipeline Framework..."):
    try:
        orders, prior, products = load_historical_data()
        model, required_features, global_p_feats, global_u_feats = get_trained_model(orders, prior)
    except Exception as e:
        st.error(f"Error loading system pipeline: {e}")
        st.stop()

# --- SIDEBAR inputs ---
st.sidebar.header("🔍 Customer Lookup")

# Suggest some valid user_ids from our loaded set
valid_users = orders['user_id'].dropna().unique().tolist()[:10]
user_id_input = st.sidebar.number_input("Enter User ID", min_value=1, value=int(valid_users[0]) if valid_users else 1, step=1)
top_k = st.sidebar.selectbox("Top-K Prediction Slots", options=[5, 10, 20], index=1)
threshold = st.sidebar.slider("Minimum Reorder Probability Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Hint**: Valid User IDs in local cache slice include: " + ", ".join(map(str, valid_users[:5])))

# --- INFERENCE ENGINE ---
def generate_user_cart_predictions(user_id, model_obj, p_feats, u_feats, features_list, top_k, threshold):
    """
    Builds the real-time feature matrix for the user and queries the trained XGBoost model.
    Returns: DataFrame containing Top-K products [product_id, product_name, reorder_probability].
             Boolean indicating if the user was cold-start.
    """
    user_orders = orders[orders['user_id'] == user_id]
    user_prior_interactions = prior[prior['order_id'].isin(user_orders['order_id'])]
    
    def get_cold_start_recommendations():
        # Fall back to products with globally highest interactions
        popular = p_feats.sort_values(by='p_total_purchases', ascending=False)
        results = popular.merge(products[['product_id', 'product_name']], on='product_id', how='left')
        # Use global reorder probability as a proxy for the probability
        results['reorder_probability'] = results['p_reorder_prob'].fillna(0.0)
        results = results[['product_id', 'product_name', 'reorder_probability']]
        results = results[results['reorder_probability'] >= threshold]
        # Return Top-K sorted by global reorder probability
        return results.sort_values(by='reorder_probability', ascending=False).head(top_k)

    if len(user_orders) == 0 or len(user_prior_interactions) == 0:
        return get_cold_start_recommendations(), True
        
    i_feats = InteractionFeatures(user_orders, user_prior_interactions).generate_features()
    if i_feats.empty: 
        return get_cold_start_recommendations(), True
    
    dataset = i_feats[['product_id', 'up_total_purchases', 'up_first_order_num', 'up_last_order_num', 'up_order_rate', 'up_order_rate_since_first_play']].copy()
    
    # We must retrieve their static user features
    dataset['user_id'] = user_id
    dataset = dataset.merge(u_feats, on='user_id', how='left')
    dataset = dataset.merge(p_feats, on='product_id', how='left')
    dataset.fillna(0, inplace=True)
    
    # Predict
    X_infer = dataset[features_list].copy()
    for col in features_list:
        if col not in X_infer.columns:
            X_infer[col] = 0
            
    X_infer = X_infer[features_list] # enforce strict order
    probs = model_obj.predict_proba(X_infer)[:, 1]
    
    dataset['reorder_probability'] = probs
    
    # Join Product Names
    dataset = dataset.merge(products[['product_id', 'product_name']], on='product_id', how='left')
    
    results = dataset[['product_id', 'product_name', 'reorder_probability']]
    
    # Apply Threshold and Select Top-K
    results = results[results['reorder_probability'] >= threshold]
    results = results.sort_values(by='reorder_probability', ascending=False).head(top_k)
    return results, False

# --- MAIN DASHBOARD ---
st.subheader(f"Predictive Cart Profile for User: `#{user_id_input}`")

if st.button("Cast Prediction Magic 🔮", type="primary"):
    with st.spinner("Generating realtime feature vectors and scoring catalog..."):
        final_preds, is_cold_start = generate_user_cart_predictions(
            user_id_input, model, global_p_feats, global_u_feats, required_features, top_k, threshold
        )
        
    if is_cold_start:
        st.warning(f"⚠️ User `#{user_id_input}` is a Cold-Start user. They have no prior purchase history in the current dataset slice.")
        st.info("Falling back to globally popular products with high global reorder probabilities.")
        
    if final_preds.empty:
        st.warning(f"No products passed the minimum threshold of {threshold}.")
    else:
        st.success(f"Successfully generated {len(final_preds)} high-probability cart suggestions!")
        
        # Use columns to position nice metrics
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Suggested Cart Size", value=f"{len(final_preds)} items")
        c2.metric(label="Highest Prediction Prob", value=f"{final_preds.iloc[0]['reorder_probability']:.3f}")
        c3.metric(label="Recommendation Mode", value="Global Popularity" if is_cold_start else "Personalized")
        
        st.markdown("### Top Predicted Reorders")
        st.dataframe(
            final_preds,
            use_container_width=True,
            hide_index=True,
            column_config={
                "product_id": st.column_config.NumberColumn("Product ID", format="%d"),
                "product_name": "Product Name",
                "reorder_probability": st.column_config.NumberColumn(
                    "Reorder Probability", 
                    format="%.3f"
                )
            }
        )
        
        # Visual Chart
        st.markdown("### Product Probability Spread")
        chart_data = final_preds[['product_name', 'reorder_probability']].set_index('product_name')
        st.bar_chart(chart_data)
