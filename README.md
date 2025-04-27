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
![image]![image](https://github.com/user-attachments/assets/8f64a047-e806-4b33-869b-08578d7aa50c)

## 📁 Prediction Model Results
![image](https://github.com/user-attachments/assets/e09aabb0-a9b9-4b51-aa34-477abfae61ca)
![image](https://github.com/user-attachments/assets/3a43f9b1-4c61-461e-8539-e8f15dbf7f41)
![image](https://github.com/user-attachments/assets/b3a396af-e714-4051-bd32-4950c531655e)

## 📁 Feature Importance
![image](https://github.com/user-attachments/assets/15798ffc-babb-4829-ae8b-e42e1b0f6779)


## 📄 File Descriptions

### 1️⃣ `df.py / index.ipynb`

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

### 2️⃣ `model.ipynb`

**Purpose:**  
Train and evaluate an XGBoost regressor to forecast daily revenue for the upcoming quarter.

**What we did:**
- **Train/Test Split**  
  - Split data at **2011-09-01** into training & testing sets. We also saw that there were many trends to do with parts of the year. For example, the revenue stream in January and December is much higher than other months, therefore we concluded that it made sense to also create a different model that trained on the beginning and last months in order to get a proper grasp on the data.
- **Feature Matrix** (`create_features_model`)  
  - Engineered temporal features:  
    - `Day of Week`, `Quarter`, `Month`, `Year`, `Is Holiday` 
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
### 2️⃣ Conclusion
![image](https://github.com/user-attachments/assets/597c4cbb-f528-4bf4-904e-e0e632a61226)
With the model trained on the last and first 35% of the data, we were able to produce the predicted revenue stream above. The overall total predicted revenue comes out to $2,659,394.25. If we were to try this again, we would probably spend more time focusing on finding more unique features to make sure the model is accurate and not underfit.


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
