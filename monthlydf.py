from df import df
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

df['days'] = df['InvoiceDate'].dt.date
daily_revenue_df = df.groupby('days')['revenue'].sum().reset_index()

df['week'] = df['InvoiceDate'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_revenue_df = df.groupby('week')['revenue'].sum().reset_index()

train = df.loc[df.index < '09-01-2011']
test = df.loc[df.index >= '09-01-2011']

fig, ax = plt.subplots(figsize=(15, 5))
train['revenue'].rolling(window=7).mean().plot(ax=ax, label='Training Set')
test['revenue'].rolling(window=7).mean().plot(ax=ax, label='Test Set')
ax.set_title('Data Train/Test')
ax.axvline('09-01-2011', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

train1 = daily_revenue_df.loc[pd.to_datetime(daily_revenue_df['days']) < '09-01-2011']
test1 = daily_revenue_df.loc[pd.to_datetime(daily_revenue_df['days']) >= '09-01-2011']

fig, ax = plt.subplots(figsize=(15, 5))

daily_revenue_df['days'] = pd.to_datetime(daily_revenue_df['days'])

ax.plot(train1['days'], train1['revenue'], label='Training Set')
ax.plot(test1['days'], test1['revenue'], label='Test Set')

ax.axvline(pd.to_datetime('09-01-2011'), color='black', ls='--')

ax.set_title('Data Train/Test Split by Days')
ax.legend()
plt.show()



split_date = pd.to_datetime('2011-09-01')

train2 = weekly_revenue_df[weekly_revenue_df['week'] < split_date]
test2 = weekly_revenue_df[weekly_revenue_df['week'] >= split_date]

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(train2['week'], train2['revenue'], label='Training Set')
ax.plot(test2['week'], test2['revenue'], label='Test Set')

ax.axvline(split_date, color='black', linestyle='--')

ax.set_title('Data Train/Test Split by Weeks')
ax.set_xlabel('Week')
ax.set_ylabel('Total Revenue')
ax.legend()
plt.grid(True)
plt.show()

daily_revenue_df['days'] = pd.to_datetime(daily_revenue_df['days'])
daily_revenue_df = daily_revenue_df.sort_values('days').reset_index(drop=True)

total_len = len(daily_revenue_df)
test_size = int(total_len * 0.30)
buffer = int((total_len - test_size) / 2)

train1 = daily_revenue_df.iloc[:buffer]
test1 = daily_revenue_df.iloc[buffer:buffer + test_size]
train2 = daily_revenue_df.iloc[buffer + test_size:]

train = pd.concat([train1, train2])

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(train1['days'], train1['revenue'], label='Training Set', color='blue')
ax.plot(train2['days'], train2['revenue'], color='blue')  # No label to avoid duplicate

ax.plot(test1['days'], test1['revenue'], label='Test Set', color='orange')

ax.axvline(test1['days'].min(), color='black', ls='--', label='Test Start')
ax.axvline(test1['days'].max(), color='gray', ls='--', label='Test End')

ax.set_title('Middle-Based 30% Test Split (Time Series)')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

daily_revenue_df['days'] = pd.to_datetime(daily_revenue_df['days'])

week_df = daily_revenue_df[
    (daily_revenue_df['days'] >= '2011-01-3') & 
    (daily_revenue_df['days'] <= '2011-01-10')
]

week2_df = daily_revenue_df[
    (daily_revenue_df['days'] >= '2011-03-14') & 
    (daily_revenue_df['days'] <= '2011-03-21')
]

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_totals = week_df.groupby('days')['revenue'].sum()
daily_totals2 = week2_df.groupby('days')['revenue'].sum()

daily_revenue_df['days'] = pd.to_datetime(daily_revenue_df['days'])

daily_revenue_df['month'] = daily_revenue_df['days'].dt.strftime('%b')  # 'Jan', 'Feb', etc.
daily_revenue_df['month_num'] = daily_revenue_df['days'].dt.month       # numeric to help order

daily_totals = daily_revenue_df.groupby(['days', 'month', 'month_num'])['revenue'].sum().reset_index()

daily_totals = daily_totals.sort_values(by='month_num')

