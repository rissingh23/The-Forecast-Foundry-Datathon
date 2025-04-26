import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('Mode_Craft_Ecommerce_Data - Online_Retail.csv')
print(df)
df = df.set_index('InvoiceDate')


def create_features(df):
    """
    Create time series features based on time series index.
    Revenue (Quantity x Unit Price)
    Day of Week
    Time of Day
    Weekday or Weekend
    Quarter
    Product Name 
    """
    df = df.copy()
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    df['timeofday'] = df.index.timeofday
    df['is_weekend'] = df.index.dayofweek >= 5
    return df

df = create_features(df)

# fig, ax = plt.subplots(figsize=(10, 8))
# sns.boxplot(data=df, x='hour', y='PJME_MW')
# ax.set_title('MW by Hour')
# plt.show()