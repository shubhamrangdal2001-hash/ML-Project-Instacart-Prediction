import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
    def train(self, X_train, y_train):
        logging.info("Training Baseline Model (Logistic Regression)...")
        self.model.fit(X_train, y_train)
        logging.info("Training complete.")
        
    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        logging.info(f"Baseline Accuracy: {acc:.4f}")
        logging.info(f"Baseline AUC: {auc:.4f}")
        
        return acc, auc, preds, probs
