#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from sklearn.linear_model import Ridge
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')


#Import Dataset
df = pd.read_csv('Mode_Craft_Ecommerce_Data - Online_Retail.csv', dtype={'InvoiceNo': str}, low_memory=False)
df.index = pd.to_datetime(df.InvoiceDate)
df = df.dropna(how='all')
df.tail(10)

#Create Columns
#Set testing model

def create_features(df):
    """
    Create time series features based on time series index.
    Revenue (Quantity x Unit Price)
    Day of Week

    Weekday or Weekend
    Quarter
    Product Name 
    """
    df = df.copy()
    df['revenue'] = df['Quantity'] * df['UnitPrice']
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df.index.dayofweek >= 5
    df['InvoiceDate'] = df.index
    return df

df = create_features(df)

df.tail(10)

#Cleaning

# Step 1: Remove rows with UnitPrice == 0
df = df[df['UnitPrice'] != 0]

# Step 2: Build cancel_map 
cancel_map = defaultdict(set)

for _, row in df.iterrows():
    invoice = (str)(row['InvoiceNo'])
    customer_id = row['CustomerID']
    if (invoice).startswith('C'):
        cancel_map[customer_id].add(row['StockCode'])

# Step 3: Remove rows where (InvoiceNo, StockCode) match the cancelled pair
def is_cancelled_pair(row):
    customer_id = row['CustomerID']
    stock = row['StockCode']
    return customer_id in cancel_map and stock in cancel_map[customer_id]

df = df[~df.apply(is_cancelled_pair, axis=1)]

# Step 4: Remove cancellation rows and InvoiceNo with "B"
df = df[~df['InvoiceNo'].str.startswith('C')]
df = df[~df['InvoiceNo'].str.contains('B', case=False)]
# 0.1% and 99.9% bounds
low_q = df['revenue'].quantile(0.00001)
high_q = df['revenue'].quantile(0.99999)

# Filter only extreme outliers
df = df[(df['revenue'] >= low_q) & (df['revenue'] <= high_q)]


