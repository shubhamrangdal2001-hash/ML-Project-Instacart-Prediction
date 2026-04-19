import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataExplorer:
    def __init__(self, output_dir="./eda_outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_order_frequency_by_dow(self, orders_df):
        if orders_df.empty: return
        plt.figure(figsize=(10,6))
        sns.countplot(x='order_dow', data=orders_df, color='skyblue')
        plt.title('Order Frequency by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Count of Orders')
        plt.savefig(f"{self.output_dir}/order_frequency_dow.png")
        plt.close()
        logging.info("Saved plot: order_frequency_dow.png")
        
    def plot_order_frequency_by_hour(self, orders_df):
        if orders_df.empty: return
        plt.figure(figsize=(10,6))
        sns.countplot(x='order_hour_of_day', data=orders_df, color='salmon')
        plt.title('Order Frequency by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Count of Orders')
        plt.savefig(f"{self.output_dir}/order_frequency_hour.png")
        plt.close()
        logging.info("Saved plot: order_frequency_hour.png")
        
    def plot_days_since_prior_order(self, orders_df):
        if orders_df.empty or 'days_since_prior_order' not in orders_df.columns: return
        plt.figure(figsize=(10,6))
        sns.countplot(x='days_since_prior_order', data=orders_df, color='lightgreen')
        plt.title('Days Since Prior Order')
        plt.xlabel('Days')
        plt.ylabel('Count of Orders')
        plt.xticks(rotation=45)
        plt.savefig(f"{self.output_dir}/days_since_prior_order.png")
        plt.close()
        logging.info("Saved plot: days_since_prior_order.png")
        
    def plot_reorder_rate_by_department(self, merged_df):
        """ Expects a dataframe with 'department' and 'reordered' columns """
        if 'reordered' not in merged_df.columns or 'department' not in merged_df.columns:
            logging.warning("Missing required columns for reorder_rate_by_department plot.")
            return
            
        reorder_rates = merged_df.groupby('department')['reordered'].mean().sort_values(ascending=False)
        plt.figure(figsize=(12,8))
        sns.barplot(x=reorder_rates.values, y=reorder_rates.index, palette='viridis')
        plt.title('Reorder Rate by Department')
        plt.xlabel('Reorder Rate')
        plt.ylabel('Department')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/reorder_rate_by_dept.png")
        plt.close()
        logging.info("Saved plot: reorder_rate_by_dept.png")

    def run_full_eda(self, orders, order_products, products, departments):
        logging.info("Running full EDA...")
        self.plot_order_frequency_by_dow(orders)
        self.plot_order_frequency_by_hour(orders)
        self.plot_days_since_prior_order(orders)
        
        if not order_products.empty and not products.empty and not departments.empty:
            merged = order_products.merge(products, on='product_id').merge(departments, on='department_id')
            self.plot_reorder_rate_by_department(merged)
        logging.info("EDA completed.")
