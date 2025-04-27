# Modecraft Ecommerce Sales Analysis

**Accelerating Modecraft’s Online Retail Sales through Data-Driven Insights & Forecasting**
#The-Forecast-Foundry
---

## 🛠 Tech Stack

- **Language & Environment**  
  - Python 3.9+  
  - Jupyter Notebook (optional)
- **Data Manipulation**  
  - pandas, NumPy
- **Visualization**  
  - matplotlib, seaborn
- **Modeling**  
  - XGBoost, scikit-learn
- **Utilities**  
  - Python’s `collections.defaultdict`

---

## 📁 Visual Graphs of Data 
![image](https://github.com/user-attachments/assets/276de6ba-3cb5-4630-9b1e-77a9225db7e2)
![image](https://github.com/user-attachments/assets/e58cb0a6-b2d8-4b6b-af00-c39d5e248a4f)
![image](https://github.com/user-attachments/assets/a77b3690-30ac-4fe9-973c-023dd01d7bb3)

## 📁 Model Test/Train Split
![image](https://github.com/user-attachments/assets/cde51973-f311-403a-b5ef-0970867dcbe8)

## 📁 Prediction Model Results
![image](https://github.com/user-attachments/assets/e09aabb0-a9b9-4b51-aa34-477abfae61ca)
![image](https://github.com/user-attachments/assets/3a43f9b1-4c61-461e-8539-e8f15dbf7f41)

## 📁 Feature Importance
![image](https://github.com/user-attachments/assets/3d6f88e4-ff7f-413f-8320-dbb51c04faa4)

## 📄 File Descriptions

### 1️⃣ `eda_and_preprocessing.py`

**Purpose:**  
Load the raw order data, engineer time-series features, clean cancellations/outliers, and produce exploratory plots.

**What we did:**
- **Imports & Style**  
  - Loaded pandas, NumPy, matplotlib & seaborn; set `fivethirtyeight` theme.  
- **Data Loading**  
  - Read CSV with `dtype={'InvoiceNo': str}` and parsed `InvoiceDate` as index.  
- **Feature Engineering**  
  - Computed `revenue = Quantity × UnitPrice`.  
  - Extracted `dayofyear`, `quarter`, `is_weekend`.  
- **Cleaning**  
  - Removed zero-price items.  
  - Identified cancellations (`InvoiceNo` starting with ‘C’) and dropped both cancel and original pairs.  
  - Filtered extreme revenue outliers (0.001%–99.999% quantiles).  
- **Exploratory Analysis**  
  - Scatter plots of revenue over time (invoice-level, daily, weekly).  
  - Boxplots showing daily-revenue distribution by month.

---

### 2️⃣ `sales_forecast_model.py`

**Purpose:**  
Train and evaluate an XGBoost regressor to forecast daily revenue for the upcoming quarter.

**What we did:**
- **Train/Test Split**  
  - Split data at **2011-09-01** into training & testing sets.  
- **Feature Matrix** (`create_features_model`)  
  - Engineered temporal features:  
    - `dayofyear`, `dayofweek`, `is_weekend`, `month`, `year`, `quarter`, `hour`  
- **Model Training**  
  - Configured `XGBRegressor` (1 000 trees, learning_rate=0.01, early_stopping_rounds=50).  
  - Trained on training set with validation on test set.  
- **Evaluation & Visualization**  
  - **Test RMSE:** ~22 790.74  
  - Plotted feature-importance bar chart.  
  - Overlaid actual vs. predicted revenue curves (full timeline & focused week).  
- **Error Analysis**  
  - Identified top 10 dates with highest absolute forecasting error for deeper investigation.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher  
- `pip` (or `conda`) for package management  

### Installation

```bash
git clone https://github.com/yourusername/modecraft-sales-analysis.git
cd modecraft-sales-analysis
pip install -r requirements.txt
Usage
Data Exploration & Preprocessing
python eda_and_preprocessing.py \
  --input data/Mode_Craft_Ecommerce_Data.csv \
  --output processed/cleaned_data.csv
Model Training & Forecasting
python sales_forecast_model.py \
  --data processed/cleaned_data.csv \
  --model_out models/xgb_model.pkl
📊 Key Results & Insights

Overall Forecast RMSE: 22 790.74
Top Features: dayofweek, is_weekend, month, dayofyear
Seasonality Patterns:
Noticeable weekly cycles with weekends dipping slightly.
Monthly boxplots reveal higher variance in Q4 (holiday season).
Error Hotspots:
Specific spike dates (e.g., 2011-11-14) suggest promotional or inventory anomalies.
🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/AmazingInsight)
Commit your changes (git commit -m 'Add insightful analysis')
Push and open a Pull Request
📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
