import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class TextFeatures:
    def __init__(self, products_df):
        self.products_df = products_df
        
    def generate_features(self, n_components=10):
        logging.info("Generating TF-IDF and SVD text features from product names...")
        if self.products_df.empty or 'product_name' not in self.products_df.columns: 
            return pd.DataFrame()
            
        # Clean product names
        product_names = self.products_df['product_name'].fillna('')
        
        # TF-IDF
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf.fit_transform(product_names)
        
        # SVD for dimensionality reduction (word2vec alternative directly from tfidf)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_matrix = svd.fit_transform(tfidf_matrix)
        
        # Create columns
        col_names = [f"text_svd_{i}" for i in range(n_components)]
        text_features = pd.DataFrame(svd_matrix, columns=col_names)
        text_features['product_id'] = self.products_df['product_id']
        
        return text_features
