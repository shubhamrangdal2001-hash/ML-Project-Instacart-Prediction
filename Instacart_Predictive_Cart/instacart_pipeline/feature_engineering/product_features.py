import pandas as pd
import logging

class ProductFeatures:
    def __init__(self, prior_df):
        self.prior_df = prior_df
        
    def generate_features(self):
        logging.info("Generating product-level features...")
        if self.prior_df.empty: return pd.DataFrame()
        
        # 1. Total times the product was purchased
        p_total_purchases = self.prior_df.groupby('product_id').size().to_frame('p_total_purchases')
        
        # 2. Reorder probability (times reordered / times purchased)
        p_reorder_prob = self.prior_df.groupby('product_id')['reordered'].mean().to_frame('p_reorder_prob')
        
        # 3. Avg position in cart
        p_avg_cart_pos = self.prior_df.groupby('product_id')['add_to_cart_order'].mean().to_frame('p_avg_cart_position')
        
        prod_features = p_total_purchases.join(p_reorder_prob).join(p_avg_cart_pos).reset_index()
        
        return prod_features
