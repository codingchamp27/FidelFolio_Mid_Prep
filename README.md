# 📈 FidelFolio: Market Cap Growth Prediction using Deep Learning

This repository contains our solution for the **FidelFolio General Championship TECH Challenge**, focused on predicting **1-year, 2-year, and 3-year forward market capitalization growth** for Indian listed companies using deep learning techniques.

---

## 🧠 Objective

To build a deep learning model that accurately predicts future market cap growth based on historical **fundamental financial indicators**. The task involves time-series modeling, careful handling of missing data, and feature-driven predictive analysis.

---

## 📊 Dataset Overview

- **Total Rows**: 24,751  
- **Total Columns**: 33  
  - 28 Financial features  
  - 3 Target variables:  
    - `Target 1`: 1-Year Forward Market Cap Growth (%)  
    - `Target 2`: 2-Year Forward Market Cap Growth (%)  
    - `Target 3`: 3-Year Forward Market Cap Growth (%)  
- **Time Span**: 1999 to 2024  
- **Granularity**: One row per company per fiscal year (panel time-series format)

---

## 🧼 Data Preprocessing Pipeline

1. **Temporal Expansion**:  
   Ensure complete company timelines up to 2020, even with missing intermediate years.

2. **Founding Year Imputation**:  
   Use **spatial nearest neighbors** (similar companies in same year) to fill in starting values.

3. **Temporal Imputation**:  
   - Used `IterativeImputer` (sklearn)  
   - Combined with **linear interpolation** for year-wise continuity  
   - Ensures **no data leakage** from future periods

4. **Outlier Detection & Correction**:  
   - Applied `Isolation Forest` on a per-year basis  
   - Replaced outliers with **median of 3 nearest non-outlier neighbors**

---

## 🧪 Modeling Approach

- Deep Learning Architectures:
  - `LSTM`, `GRU`, `CNN`, `Dense` Layers
- Framework: TensorFlow / Keras  
- Optimization: `Adam`, `EarlyStopping`  
- Tuning: `Keras Tuner`, `RandomizedSearchCV`  
- Baseline Models: `LinearRegression`, `DecisionTreeRegressor`, `SVR`

---

## 📏 Evaluation

- **Metric**: RMSE (Root Mean Squared Error)  
- Comparison across:
  - Different model types
  - Short vs. long horizon targets (1Y vs 3Y)

---