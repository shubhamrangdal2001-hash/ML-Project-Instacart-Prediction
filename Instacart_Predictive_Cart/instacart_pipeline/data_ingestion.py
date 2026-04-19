import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reduce_mem_usage(df):
    """
    Skipping aggressive pandas memory reduction because it throws UfuncTypeError on string series min/max natively.
    """
    return df

class InstacartDataLoader:
    def __init__(self, data_path="./data"):
        self.data_path = data_path
        
    def load_table(self, file_name, usecols=None, chunksize=None, nrows=None):
        path = f"{self.data_path}/{file_name}"
        logging.info(f"Loading {file_name} from {path}...")
        
        try:
            if chunksize:
                chunks = []
                for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, nrows=nrows):
                    chunks.append(reduce_mem_usage(chunk))
                return pd.concat(chunks, axis=0)
            else:
                df = pd.read_csv(path, usecols=usecols, nrows=nrows)
                return reduce_mem_usage(df)
        except Exception as e:
            logging.error(f"Failed to load {file_name}: {e}")
            return pd.DataFrame()
            
    def load_all_data(self):
        """
        Loads all required datasets for the Predictive Cart system.
        Returns multiple dataframes.
        """
        orders = self.load_table('orders.csv')
        # order_products__prior.csv handles past behavior. Extremely large file.
        order_products_prior = self.load_table('order_products__prior.csv', chunksize=1000000)
        
        # Train dataset acting as the target/label context
        order_products_train = self.load_table('order_products__train.csv')
            
        products = self.load_table('products.csv')
        aisles = self.load_table('aisles.csv')
        departments = self.load_table('departments.csv')
        
        return orders, order_products_prior, order_products_train, products, aisles, departments

if __name__ == "__main__":
    loader = InstacartDataLoader()
    # orders, prior, train, products, aisles, depts = loader.load_all_data()
