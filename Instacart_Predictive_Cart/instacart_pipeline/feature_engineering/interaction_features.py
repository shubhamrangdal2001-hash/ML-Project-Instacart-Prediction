import pandas as pd
import logging

class InteractionFeatures:
    def __init__(self, orders_df, prior_df):
        self.orders_df = orders_df
        self.prior_df = prior_df
        
    def generate_features(self):
        logging.info("Generating user-product interaction features...")
        if self.prior_df.empty or self.orders_df.empty: return pd.DataFrame()
        
        merged = self.prior_df.merge(self.orders_df[['order_id', 'user_id', 'order_number']], on='order_id')
        
        # 1. Number of times user bought product
        up_purchases = merged.groupby(['user_id', 'product_id']).size().to_frame('up_total_purchases')
        
        # 2. First order number where user bought the product
        up_first_order = merged.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('up_first_order_num')
        
        # 3. Last order number where user bought the product
        up_last_order = merged.groupby(['user_id', 'product_id'])['order_number'].max().to_frame('up_last_order_num')
        
        # 4. User total orders
        user_total_orders = self.orders_df.groupby('user_id')['order_number'].max().to_frame('u_total_orders').reset_index()
        
        up_features = up_purchases.join(up_first_order).join(up_last_order).reset_index()
        up_features = up_features.merge(user_total_orders, on='user_id')
        
        # 5. Up reorder frequency (purchases / possible orders since first bought)
        # possible orders = total_orders - first_order_num + 1
        up_features['up_order_rate'] = up_features['up_total_purchases'] / (up_features['u_total_orders'] - up_features['up_first_order_num'] + 1)
        up_features['up_order_rate_since_first_play'] = up_features['up_total_purchases'] / up_features['u_total_orders']
        
        up_features.drop('u_total_orders', axis=1, inplace=True)
        return up_features
