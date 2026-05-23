import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np

# Ensure the pipeline packages are in Python's path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from instacart_pipeline.data_ingestion import InstacartDataLoader
from instacart_pipeline.feature_engineering.user_features import UserFeatures
from instacart_pipeline.feature_engineering.product_features import ProductFeatures
from instacart_pipeline.feature_engineering.interaction_features import InteractionFeatures
from instacart_pipeline.modeling.advanced_models import AdvancedModels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_serialize():
    logging.info("Initializing Data Ingestion...")
    
    # Resolve the data path
    path_opts = ["data/raw", "../data/raw", "../../data/raw", "C:/Users/shubh/OneDrive/Desktop/Project_ML/data/raw"]
    valid_path = "data/raw"
    for p in path_opts:
        if os.path.exists(p):
            valid_path = p
            break
            
    logging.info(f"Using data path: {valid_path}")
    loader = InstacartDataLoader(data_path=valid_path)
    
    # Load raw data matching Streamlit configurations
    logging.info("Loading orders, prior interactions, and products...")
    orders = loader.load_table('orders.csv', nrows=100000)
    prior = loader.load_table('order_products__prior.csv', nrows=250000)
    products = loader.load_table('products.csv')
    
    if orders.empty or prior.empty:
        logging.error("Failed to load historical data files. Check paths and data files.")
        return
        
    logging.info(f"Loaded {len(orders)} orders and {len(prior)} prior interactions.")
    
    # Process features on sampled users (same logic as in app.py)
    logging.info("Sampling users for training...")
    sampled_users = pd.Series(orders['user_id'].unique()).sample(min(1500, len(orders['user_id'].unique())), random_state=42)
    orders_sub = orders[orders['user_id'].isin(sampled_users)]
    prior_sub = prior[prior['order_id'].isin(orders_sub['order_id'])]
    
    logging.info("Generating features for the training sample...")
    u_feats = UserFeatures(orders_sub, prior_sub).generate_features()
    p_feats = ProductFeatures(prior_sub).generate_features()
    i_feats = InteractionFeatures(orders_sub, prior_sub).generate_features()
    
    if i_feats.empty:
        logging.error("Interaction features are empty. Cannot train model.")
        return
        
    logging.info("Assembling dataset...")
    dataset = i_feats[['user_id', 'product_id', 'up_total_purchases', 'up_first_order_num', 'up_last_order_num', 'up_order_rate', 'up_order_rate_since_first_play']]
    dataset = dataset.merge(u_feats, on='user_id', how='left')
    dataset = dataset.merge(p_feats, on='product_id', how='left')
    dataset.fillna(0, inplace=True)
    
    # Mocking a target variable (same behavior as dynamic app.py logic)
    dataset['reordered_target'] = np.random.choice([0, 1], p=[0.8, 0.2], size=len(dataset))
    
    features = [c for c in dataset.columns if c not in ['user_id', 'product_id', 'reordered_target']]
    X = dataset[features]
    y = dataset['reordered_target']
    
    logging.info(f"Training XGBoost classifier on {len(X)} samples with {len(features)} features...")
    advanced = AdvancedModels()
    advanced.train_xgb(X, y)
    
    model_pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
    logging.info(f"Serializing model package to {model_pkl_path}...")
    
    # Build complete package containing the model and all inference-related artifacts
    package = {
        'model': advanced.xgb_model,
        'features': features,
        'global_u_feats': u_feats,
        'global_p_feats': p_feats,
        'orders': orders,
        'prior': prior,
        'products': products
    }
    
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(package, f, protocol=4)
        
    logging.info("Model training and serialization completed successfully!")

if __name__ == "__main__":
    train_and_serialize()
