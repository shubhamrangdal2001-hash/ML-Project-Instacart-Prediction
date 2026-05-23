import os
import json
import pickle
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def init():
    global model, features_list, global_u_feats, global_p_feats, orders, prior, products
    
    # Azure ML container mounts model files under AZUREML_MODEL_DIR
    model_dir = os.getenv("AZUREML_MODEL_DIR", "")
    pkl_path = os.path.join(model_dir, "model.pkl")
    
    if not os.path.exists(pkl_path):
        # Fallback to search recursively for model.pkl
        for root, dirs, files in os.walk(model_dir):
            if "model.pkl" in files:
                pkl_path = os.path.join(root, "model.pkl")
                break
                
    # Local fallback for development testing
    if not os.path.exists(pkl_path):
        local_opts = ["Instacart_Predictive_Cart/model.pkl", "model.pkl", "../model.pkl"]
        for path in local_opts:
            if os.path.exists(path):
                pkl_path = path
                break
                
    logging.info(f"Loading serialized model package from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        package = pickle.load(f)
        
    model = package["model"]
    features_list = package["features"]
    global_u_feats = package["global_u_feats"]
    global_p_feats = package["global_p_feats"]
    orders = package["orders"]
    prior = package["prior"]
    products = package.get("products", pd.DataFrame())
    
    logging.info("Model and feature references loaded successfully into scoring environment.")

def run(raw_data):
    logging.info("Executing scoring request...")
    try:
        data = json.loads(raw_data)
        # Support MLflow format: {"dataframe_records": [{"user_id": 112, ...}]}
        if isinstance(data, dict) and "dataframe_records" in data:
            records = data["dataframe_records"]
            if isinstance(records, list) and len(records) > 0:
                data = records[0]
            else:
                return json.dumps({"error": "Empty or invalid dataframe_records"})
        
        user_id = int(data["user_id"])
        top_k = int(data.get("top_k", 10))
        threshold = float(data.get("threshold", 0.0))
    except Exception as e:
        return json.dumps({"error": f"Invalid input format: {str(e)}"})
        
    try:
        # Filter raw user history slices
        user_orders = orders[orders['user_id'] == user_id]
        user_prior_interactions = prior[prior['order_id'].isin(user_orders['order_id'])]
        
        # Cold start fallback function
        def get_cold_start():
            popular = global_p_feats.sort_values(by='p_total_purchases', ascending=False)
            if not products.empty:
                results = popular.merge(products[['product_id', 'product_name']], on='product_id', how='left')
            else:
                results = popular.copy()
                results['product_name'] = "Unknown Product"
                
            results['reorder_probability'] = results['p_reorder_prob'].fillna(0.0)
            results = results[['product_id', 'product_name', 'reorder_probability']]
            results = results[results['reorder_probability'] >= threshold]
            results = results.sort_values(by='reorder_probability', ascending=False).head(top_k)
            return results, True
            
        if len(user_orders) == 0 or len(user_prior_interactions) == 0:
            results, is_cold = get_cold_start()
        else:
            # Generate interaction features for this specific user
            merged = user_prior_interactions.merge(user_orders[['order_id', 'user_id', 'order_number']], on='order_id')
            up_purchases = merged.groupby(['user_id', 'product_id']).size().to_frame('up_total_purchases')
            up_first_order = merged.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('up_first_order_num')
            up_last_order = merged.groupby(['user_id', 'product_id'])['order_number'].max().to_frame('up_last_order_num')
            user_total_orders = user_orders['order_number'].max()
            
            up_features = up_purchases.join(up_first_order).join(up_last_order).reset_index()
            up_features['up_order_rate'] = up_features['up_total_purchases'] / (user_total_orders - up_features['up_first_order_num'] + 1)
            up_features['up_order_rate_since_first_play'] = up_features['up_total_purchases'] / user_total_orders
            
            # Merge global profiles
            dataset = up_features.merge(global_u_feats, on='user_id', how='left')
            dataset = dataset.merge(global_p_feats, on='product_id', how='left')
            dataset.fillna(0, inplace=True)
            
            # Create feature matrix matching original columns
            X_infer = dataset.copy()
            for col in features_list:
                if col not in X_infer.columns:
                    X_infer[col] = 0
            X_infer = X_infer[features_list]
            
            probs = model.predict_proba(X_infer)[:, 1]
            dataset['reorder_probability'] = probs
            
            if not products.empty:
                dataset = dataset.merge(products[['product_id', 'product_name']], on='product_id', how='left')
            else:
                dataset['product_name'] = "Unknown Product"
                
            results = dataset[['product_id', 'product_name', 'reorder_probability']]
            results = results[results['reorder_probability'] >= threshold]
            results = results.sort_values(by='reorder_probability', ascending=False).head(top_k)
            is_cold = False
            
        # Return serialized results
        output = {
            "user_id": user_id,
            "mode": "Global Popularity" if is_cold else "Personalized",
            "predictions": results.to_dict(orient="records")
        }
        return json.dumps(output)
        
    except Exception as e:
        return json.dumps({"error": f"Inference execution failed: {str(e)}"})
