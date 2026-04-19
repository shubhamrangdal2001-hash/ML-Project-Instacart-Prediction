import pandas as pd
import logging

class ErrorAnalysis:
    def __init__(self, test_df, predictions_df):
        """
        test_df: Original test dataframe with user and product characteristics.
        predictions_df: Dataframe with 'user_id', 'product_id', 'actual', 'predicted_prob' Ensure matching index.
        """
        self.merged = test_df.merge(predictions_df, on=['user_id', 'product_id'])
        
    def analyze_cold_starts(self):
        logging.info("Analyzing performance on cold-start users (users with few total orders)...")
        if 'u_total_orders' not in self.merged.columns: return
        
        cold_users = self.merged[self.merged['u_total_orders'] <= 5]
        regular_users = self.merged[self.merged['u_total_orders'] > 5]
        
        c_prob = cold_users['predicted_prob'].mean()
        r_prob = regular_users['predicted_prob'].mean()
        logging.info(f"Cold users Avg Predicted Prob: {c_prob:.4f}")
        logging.info(f"Regular users Avg Predicted Prob: {r_prob:.4f}")
        return c_prob, r_prob

    def analyze_rare_items(self):
        logging.info("Analyzing predictability of rare items...")
        if 'p_total_purchases' not in self.merged.columns: return
        
        rare_items = self.merged[self.merged['p_total_purchases'] < 50]
        popular_items = self.merged[self.merged['p_total_purchases'] >= 50]
        
        rp = rare_items['predicted_prob'].mean()
        pp = popular_items['predicted_prob'].mean()
        logging.info(f"Rare items Avg Predicted Prob: {rp:.4f}")
        logging.info(f"Popular items Avg Predicted Prob: {pp:.4f}")
        return rp, pp
