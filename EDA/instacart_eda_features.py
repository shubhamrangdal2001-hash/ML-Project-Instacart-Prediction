"""
============================================================
  PROJECT: The Retail Oracle - Instacart Reorder Prediction
  FILE: instacart_eda_features.py
  COVERS: Step 1 (Data Loading) → Step 6 (Feature Engineering)
============================================================
"""

# ============================================================
# STEP 1: IMPORTS & SETUP
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("✅ Imports done")

# ============================================================
# STEP 2: DATA LOADING
# ============================================================
"""
Dataset: Instacart Market Basket Analysis
Tables:
  - aisles.csv         → 134 aisles
  - departments.csv    → 21 departments
  - products.csv       → 49,688 products
  - orders.csv         → 3,421,083 orders from 206,209 users
  - order_products__train.csv → 1,384,617 rows (train labels)
  
eval_set in orders:
  - 'prior'  → all historical orders  (3,214,874 rows)
  - 'train'  → last order for training users (131,209 rows)
  - 'test'   → last order for test users (75,000 rows)
"""

DATA_PATH = r"C:\Users\shubh\OneDrive\Desktop\Project_ML\data\raw\\"


aisles      = pd.read_csv(DATA_PATH + "aisles.csv")
departments = pd.read_csv(DATA_PATH + "departments.csv")
products    = pd.read_csv(DATA_PATH + "products.csv")
orders      = pd.read_csv(DATA_PATH + "orders.csv")
order_products_train = pd.read_csv(DATA_PATH + "order_products__train.csv")
order_products_prior = pd.read_csv(DATA_PATH + "order_products__prior.csv")

print(f"aisles:               {aisles.shape}")
print(f"departments:          {departments.shape}")
print(f"products:             {products.shape}")
print(f"orders:               {orders.shape}")
print(f"order_products_train: {order_products_train.shape}")
print(f"order_products_prior: {order_products_prior.shape}")

# ============================================================
# STEP 3: DATA UNDERSTANDING & SCHEMA
# ============================================================
"""
SCHEMA OVERVIEW:
---------------------------------------------------------
orders.csv
  order_id              → unique order identifier
  user_id               → customer id
  eval_set              → prior / train / test
  order_number          → sequence number (1=first order)
  order_dow             → day of week (0=Sunday, 6=Saturday)
  order_hour_of_day     → hour (0-23)
  days_since_prior_order→ NULL for first order

order_products__train.csv / order_products__prior.csv
  order_id              → links to orders.csv
  product_id            → links to products.csv
  add_to_cart_order     → 1st, 2nd... item added to cart
  reordered             → TARGET: 1=bought before, 0=new

products.csv
  product_id, product_name, aisle_id, department_id

aisles.csv      → aisle_id, aisle
departments.csv → department_id, department
"""

# Separate splits
prior_orders = orders[orders['eval_set'] == 'prior']
train_orders = orders[orders['eval_set'] == 'train']
test_orders  = orders[orders['eval_set'] == 'test']

print(f"\nPrior orders:  {prior_orders.shape[0]:,}")
print(f"Train orders:  {train_orders.shape[0]:,}")
print(f"Test orders:   {test_orders.shape[0]:,}")
print(f"Unique users:  {orders['user_id'].nunique():,}")

# ============================================================
# STEP 4: MISSING VALUE ANALYSIS
# ============================================================
print("\n=== Missing Value Report ===")
for name, df in [("orders", orders), ("order_products_train", order_products_train),
                 ("products", products)]:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"\n{name}:")
        for col, count in missing.items():
            print(f"  {col}: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        print(f"\n{name}: No missing values ✅")

"""
KEY FINDING: days_since_prior_order has 206,209 NULLs
→ These are first-ever orders (order_number == 1) — expected, not a data quality issue
→ Strategy: Fill with 0 OR create a flag: is_first_order = 1
"""

# Verify: all NULLs are order_number == 1
null_days = orders[orders['days_since_prior_order'].isnull()]
assert (null_days['order_number'] == 1).all(), "NULLs not all first orders!"
print("\n✅ Confirmed: All NULLs in days_since_prior_order are first orders (order_number=1)")

# Fix: fill with 0 and add flag
orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0)
orders['is_first_order'] = (orders['order_number'] == 1).astype(int)

# ============================================================
# STEP 5: EXPLORATORY DATA ANALYSIS
# ============================================================

# -----------------------------------------------------------
# 5A. TARGET VARIABLE ANALYSIS
# -----------------------------------------------------------
reorder_rate = order_products_train['reordered'].mean()
print(f"\nTarget (reordered) distribution:")
print(f"  Reordered (1): {order_products_train['reordered'].sum():,}  ({reorder_rate*100:.1f}%)")
print(f"  New (0):       {(order_products_train['reordered']==0).sum():,}  ({(1-reorder_rate)*100:.1f}%)")
print(f"\n⚠️  Mild class imbalance: {reorder_rate:.2f} vs {1-reorder_rate:.2f}")
print("→ Use F1, Precision-Recall, ROC-AUC as evaluation metrics, not just Accuracy")

# -----------------------------------------------------------
# 5B. ORDER BEHAVIOR ANALYSIS
# -----------------------------------------------------------
# Orders per user
orders_per_user = prior_orders.groupby('user_id')['order_id'].count()
print(f"\nOrders per user:")
print(f"  Min: {orders_per_user.min()}, Median: {orders_per_user.median()}, Max: {orders_per_user.max()}")
print(f"  >10 orders: {(orders_per_user > 10).mean()*100:.1f}% of users")

# Day-of-week patterns
dow_map = {0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'}
dow_counts = orders['order_dow'].map(dow_map).value_counts().reindex(
    ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])
print(f"\nTop order day: {dow_counts.idxmax()} ({dow_counts.max():,} orders)")
print(f"Peak hour: {orders['order_hour_of_day'].value_counts().idxmax()}:00")

# Days between orders
dspo = prior_orders['days_since_prior_order'].replace(0, np.nan).dropna()
print(f"\nDays between orders:")
print(f"  Mean: {dspo.mean():.1f},  Median: {dspo.median():.0f}")
print(f"  Weekly shoppers (7d ±1): {((dspo>=6)&(dspo<=8)).mean()*100:.1f}%")
print(f"  Monthly shoppers (28-30d): {((dspo>=28)&(dspo<=30)).mean()*100:.1f}%")

# -----------------------------------------------------------
# 5C. PRODUCT & BASKET ANALYSIS
# -----------------------------------------------------------
# Products per order in train
products_per_order = order_products_train.groupby('order_id')['product_id'].count()
print(f"\nProducts per order:")
print(f"  Median: {products_per_order.median():.0f}, Mean: {products_per_order.mean():.1f}")
print(f"  Max: {products_per_order.max()}")

# Cart position analysis
cp_reorder = order_products_train.groupby('add_to_cart_order')['reordered'].mean()
print(f"\nReorder rate by cart position:")
print(f"  Position 1: {cp_reorder.get(1, 0):.3f}")
print(f"  Position 5: {cp_reorder.get(5, 0):.3f}")
print(f"  Position 10: {cp_reorder.get(10, 0):.3f}")
print("→ Items added early in cart are more likely to be reorders")

# -----------------------------------------------------------
# 5D. DEPARTMENT & AISLE ANALYSIS
# -----------------------------------------------------------
prod_dept = products.merge(departments, on='department_id')
op_dept = order_products_train.merge(prod_dept[['product_id','department']], on='product_id')
dept_volume = op_dept['department'].value_counts()
print(f"\nTop 5 departments by order volume:")
print(dept_volume.head(5).to_string())

dept_reorder = op_dept.groupby('department')['reordered'].mean().sort_values(ascending=False)
print(f"\nTop 5 departments by reorder rate:")
print(dept_reorder.head(5).round(3).to_string())

# ============================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================
"""
FEATURE STRATEGY: 3-level feature groups

Group A: USER FEATURES
  → Who is this user? How do they shop?
  
Group B: PRODUCT FEATURES
  → How popular is this product? How often is it reordered?
  
Group C: USER-PRODUCT INTERACTION FEATURES
  → How does THIS user relate to THIS product?
  
Target for each (user, product) pair in train set:
  → reordered = 1 if the product appears in their last order
"""

print("\n=== FEATURE ENGINEERING ===")

# -----------------------------------------------------------
# 6A. USER FEATURES
# -----------------------------------------------------------
print("\n>> Group A: User Features")

# Merge prior order_products with prior orders to get user context
prior_op = order_products_prior.merge(
    prior_orders[['order_id','user_id','order_number','order_dow',
                  'order_hour_of_day','days_since_prior_order']],
    on='order_id'
)

# Basic aggregation
user_feat = prior_orders.groupby('user_id').agg(
    user_total_orders        = ('order_id', 'count'),
    user_avg_days_between    = ('days_since_prior_order', 'mean'),
    user_std_days_between    = ('days_since_prior_order', 'std'),
    user_min_days_between    = ('days_since_prior_order', 'min'),
    user_max_days_between    = ('days_since_prior_order', 'max'),
    user_avg_order_hour      = ('order_hour_of_day', 'mean'),
    user_std_order_hour      = ('order_hour_of_day', 'std'),
).reset_index()

# Total products and unique products purchased
user_prod_stats = prior_op.groupby('user_id').agg(
    user_total_products      = ('product_id', 'count'),
    user_unique_products     = ('product_id', 'nunique'),
    user_total_reorders      = ('reordered', 'sum'),
).reset_index()

user_prod_stats['user_reorder_rate'] = (
    user_prod_stats['user_total_reorders'] /
    user_prod_stats['user_total_products']
)
user_prod_stats['user_avg_basket_size'] = (
    user_prod_stats['user_total_products'] /
    user_prod_stats['user_total_products'].clip(lower=1)  # proxy
)

user_feat = user_feat.merge(user_prod_stats, on='user_id', how='left')

# Fill NaN in std (single-order users have no std)
user_feat['user_std_days_between'] = user_feat['user_std_days_between'].fillna(0)
user_feat['user_std_order_hour']   = user_feat['user_std_order_hour'].fillna(0)

print(f"  ✅ User features: {user_feat.shape[1]} columns, {user_feat.shape[0]:,} users")
print(f"  Columns: {list(user_feat.columns)}")

# -----------------------------------------------------------
# 6B. PRODUCT FEATURES
# -----------------------------------------------------------
print("\n>> Group B: Product Features")

prod_feat = prior_op.groupby('product_id').agg(
    product_total_orders     = ('order_id', 'count'),
    product_reorder_rate     = ('reordered', 'mean'),
    product_avg_cart_pos     = ('add_to_cart_order', 'mean'),
    product_std_cart_pos     = ('add_to_cart_order', 'std'),
    product_unique_users     = ('user_id', 'nunique'),
).reset_index()

prod_feat['product_std_cart_pos'] = prod_feat['product_std_cart_pos'].fillna(0)

# Attach metadata
prod_feat = prod_feat.merge(
    products.merge(departments, on='department_id').merge(aisles, on='aisle_id'),
    on='product_id', how='left'
)

# Department and aisle level reorder rates (smoothed signal)
dept_reorder_rates = (
    prod_feat.groupby('department')['product_reorder_rate']
    .mean().reset_index()
    .rename(columns={'product_reorder_rate': 'dept_avg_reorder_rate'})
)
aisle_reorder_rates = (
    prod_feat.groupby('aisle')['product_reorder_rate']
    .mean().reset_index()
    .rename(columns={'product_reorder_rate': 'aisle_avg_reorder_rate'})
)
prod_feat = prod_feat.merge(dept_reorder_rates, on='department', how='left')
prod_feat = prod_feat.merge(aisle_reorder_rates, on='aisle', how='left')

print(f"  ✅ Product features: {prod_feat.shape[1]} columns, {prod_feat.shape[0]:,} products")
key_cols = ['product_id','product_total_orders','product_reorder_rate',
            'product_avg_cart_pos','product_unique_users',
            'dept_avg_reorder_rate','aisle_avg_reorder_rate']
print(f"  Key columns: {key_cols}")

# -----------------------------------------------------------
# 6C. USER-PRODUCT INTERACTION FEATURES
# -----------------------------------------------------------
print("\n>> Group C: User-Product Interaction Features")

# For each (user, product) pair in prior orders
up_feat = prior_op.groupby(['user_id', 'product_id']).agg(
    up_times_ordered         = ('order_id', 'count'),
    up_reorder_count         = ('reordered', 'sum'),
    up_avg_cart_position     = ('add_to_cart_order', 'mean'),
    up_first_order_number    = ('order_number', 'min'),
    up_last_order_number     = ('order_number', 'max'),
).reset_index()

# Derived features
up_feat['up_reorder_rate'] = up_feat['up_reorder_count'] / up_feat['up_times_ordered']
up_feat['up_orders_span']  = up_feat['up_last_order_number'] - up_feat['up_first_order_number']
up_feat['up_buy_frequency'] = up_feat['up_times_ordered'] / (up_feat['up_orders_span'] + 1)

# How many orders ago was last purchase? (recency)
user_last_order = prior_orders.groupby('user_id')['order_number'].max().reset_index()
user_last_order.columns = ['user_id', 'user_max_order']
up_feat = up_feat.merge(user_last_order, on='user_id', how='left')
up_feat['up_orders_since_last'] = (
    up_feat['user_max_order'] - up_feat['up_last_order_number']
)

# Probability of reorder based on orders since last purchase (simple smoothing)
# Hypothesis: items not ordered recently are less likely to be reordered
up_feat['up_recency_score'] = 1 / (up_feat['up_orders_since_last'] + 1)

print(f"  ✅ User-Product features: {up_feat.shape[1]} columns, {up_feat.shape[0]:,} pairs")
print(f"  Columns: {list(up_feat.columns)}")

# -----------------------------------------------------------
# 6D. ASSEMBLE FINAL FEATURE MATRIX
# -----------------------------------------------------------
print("\n>> Assembling Final Feature Matrix...")

# Get all (user, product) pairs to predict for train set
# For each user in train_orders, we predict which prior products they'll reorder
train_users = train_orders[['order_id','user_id']].copy()

# Candidate products = all products user has ordered in prior
candidates = up_feat[['user_id','product_id']].copy()
candidates = candidates.merge(train_users, on='user_id', how='inner')

# Attach features
final_df = candidates.merge(up_feat.drop('user_max_order', axis=1), 
                            on=['user_id','product_id'], how='left')
final_df = final_df.merge(user_feat, on='user_id', how='left')
final_df = final_df.merge(prod_feat[key_cols], on='product_id', how='left')

# Attach labels
labels = order_products_train[['order_id','product_id','reordered']].copy()
final_df = final_df.merge(labels, on=['order_id','product_id'], how='left')
final_df['reordered'] = final_df['reordered'].fillna(0).astype(int)

print(f"\n  ✅ Final feature matrix: {final_df.shape[1]} columns, {final_df.shape[0]:,} rows")
print(f"  Target distribution:")
print(f"    Positive (reordered=1): {final_df['reordered'].sum():,} ({final_df['reordered'].mean()*100:.1f}%)")
print(f"    Negative (reordered=0): {(final_df['reordered']==0).sum():,}")

# -----------------------------------------------------------
# 6E. FEATURE IMPORTANCE HYPOTHESES
# -----------------------------------------------------------
"""
TOP FEATURE HYPOTHESES (ranked by expected predictive power):

1. up_times_ordered       → More purchases = higher chance of reorder (HIGH)
2. up_reorder_rate        → Direct measure of user-product loyalty (HIGH)
3. product_reorder_rate   → Global popularity of product being reordered (HIGH)
4. up_orders_since_last   → Recency — if bought recently, likely to reorder (HIGH)
5. up_buy_frequency       → How regularly the user buys this product (MEDIUM)
6. user_reorder_rate      → Habitual buyer tends to reorder more (MEDIUM)
7. product_avg_cart_pos   → Items added early = habit = reorder (MEDIUM)
8. up_avg_cart_position   → Same but user-specific (MEDIUM)
9. user_total_orders      → More experienced users have clearer preferences (LOW)
10. dept_avg_reorder_rate → Some departments (personal care) are high-reorder (LOW)
"""

# -----------------------------------------------------------
# 6F. SAVE PROCESSED DATA
# -----------------------------------------------------------
print("\n>> Saving processed files...")
user_feat.to_csv('data/processed/user_features.csv', index=False)
prod_feat.to_csv('data/processed/product_features.csv', index=False)
up_feat.to_csv('data/processed/user_product_features.csv', index=False)
final_df.to_csv('data/processed/final_features_train.csv', index=False)
print("✅ All files saved to data/processed/")

print("\n" + "="*60)
print("COMPLETED: Data Loading → EDA → Feature Engineering")
print("NEXT STEPS: Baseline Model → Model Comparison → Evaluation")
print("="*60)
