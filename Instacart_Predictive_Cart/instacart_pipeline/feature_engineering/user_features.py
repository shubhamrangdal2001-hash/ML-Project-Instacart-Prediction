import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UserFeatures:
    def __init__(self, orders_df, prior_df):
        self.orders_df = orders_df
        self.prior_df = prior_df
        
    def generate_features(self):
        logging.info("Generating user-level features...")
        if self.prior_df.empty or self.orders_df.empty:
            logging.warning("Dataframes are empty, cannot generate user features.")
            return pd.DataFrame()
            
        merged = self.prior_df.merge(self.orders_df[['order_id', 'user_id', 'days_since_prior_order']], on='order_id')
        
        # 1. Total number of orders per user
        user_orders = self.orders_df.groupby('user_id')['order_number'].max().to_frame('u_total_orders')
        
        # 2. Avg basket size
        basket_size = merged.groupby(['user_id', 'order_id'])['add_to_cart_order'].max().reset_index()
        avg_basket_size = basket_size.groupby('user_id')['add_to_cart_order'].mean().to_frame('u_avg_basket_size')
        
        # 3. Overall reorder ratio for the user
        user_reorder_ratio = merged.groupby('user_id')['reordered'].mean().to_frame('u_reorder_ratio')
        
        # 4. Avg days between orders
        avg_days_between = self.orders_df.groupby('user_id')['days_since_prior_order'].mean().to_frame('u_avg_days_between_orders')
        
        user_features = user_orders.join(avg_basket_size).join(user_reorder_ratio).join(avg_days_between).reset_index()
        
        logging.info(f"Generated user features for {len(user_features)} users.")
        return user_features
