import numpy as np
import pandas as pd
import logging

class BusinessMetrics:
    @staticmethod
    def _get_top_k(user_probs, k):
        """ Return top k predicted product indices for a user """
        return user_probs.nlargest(k).index.tolist()
        
    @staticmethod
    def precision_recall_at_k(y_true_df, y_prob_df, k=10):
        """
        Calculates precision and recall aligned with business metrics mapping to the end-user's basket.
        y_true_df: ['user_id', 'product_id', 'reordered']
        y_prob_df: ['user_id', 'product_id', 'score']
        """
        logging.info(f"Calculating Precision@{k} and Recall@{k}...")
        results = []
        
        grouped_true = y_true_df.groupby('user_id')
        grouped_prob = y_prob_df.groupby('user_id')
        
        for user_id, group in grouped_prob:
            if user_id not in grouped_true.groups: continue
            
            top_k_preds = group.nlargest(k, 'score')['product_id'].tolist()
            actuals = grouped_true.get_group(user_id)
            actual_reorders = actuals[actuals['reordered'] == 1]['product_id'].tolist()
            
            if not actual_reorders: continue # Avoid division by zero if user bought mostly new items
                
            hits = len(set(top_k_preds).intersection(set(actual_reorders)))
            
            precision = hits / k
            recall = hits / len(actual_reorders)
            
            results.append((precision, recall))
            
        if not results: return 0.0, 0.0
            
        avg_precision = np.mean([r[0] for r in results])
        avg_recall = np.mean([r[1] for r in results])
        
        logging.info(f"Precision@{k}: {avg_precision:.4f} | Recall@{k}: {avg_recall:.4f}")
        return avg_precision, avg_recall
