import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedModels:
    def __init__(self):
        # We cap max_depth and restrict n_jobs to handle memory appropriately
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        self.xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
        
    def train_rf(self, X_train, y_train):
        logging.info("Training Random Forest Classifier...")
        self.rf_model.fit(X_train, y_train)
        
    def train_xgb(self, X_train, y_train):
        logging.info("Training XGBoost Classifier...")
        self.xgb_model.fit(X_train, y_train)
        
    def evaluate(self, model_name, X_test, y_test):
        model = self.rf_model if model_name == 'rf' else self.xgb_model
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        logging.info(f"{model_name.upper()} Accuracy: {acc:.4f}")
        logging.info(f"{model_name.upper()} AUC: {auc:.4f}")
        
        return acc, auc, preds, probs
