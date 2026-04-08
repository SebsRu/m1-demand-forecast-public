# Module 1 — AI Forecast Comparator & Champion Model Selector

**Part of the Supply Chain AI Suite** | By Sebastián Rueda, Supply Chain AI Orchestrator

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://m1-demand-forecast-public-fyjvowtsbgsa6yfovy82xk.streamlit.app)

---

## 🎯 The Problem

In Pharma and CPG companies, demand planners struggle with a critical question every quarter:

- **Which forecasting method should we trust?** (Prophet, XGBoost, SARIMA, or the current manual approach?)
- **How much money are we leaving on the table with our current forecast accuracy?** (Stockouts? Excess inventory?)
- **Which model actually beats our legacy process?** (And by how much?)

Manual forecasting with MA7 (7-day moving average) in SAP takes **hours of debate** and yields only **~4.6% MAPE**. AI models can do better — but which one?

This tool answers all three in **seconds** by running a tournament between multiple forecasting models and showing the **financial impact** of choosing each one.

---

## 💡 The Solution

**Module 1** is an AI-powered forecast comparator that:

✅ Trains **4 competing models** in parallel (Prophet, XGBoost, SARIMA, MA7 Baseline)
✅ Calculates **MAPE, Bias, and Accuracy Improvement** for each
✅ Quantifies **Working Capital Impact** (stockout risk vs. excess inventory cost)
✅ Selects the **"Champion" model** based on test-set performance
✅ Provides **multi-dimensional filtering** (by Category, ABC Class, Analysis Window)
✅ Exports ranked model performance to **Module 2** (Inventory Optimization)

---

## 🚀 Live Demo

### **👉 [Launch the App](https://m1-demand-forecast-public-fyjvowtsbgsa6yfovy82xk.streamlit.app)**

Try it now with **demo supply chain data** (synthetic demand history) or **upload your own CSV**.

### What you'll see:

**📊 Tab 1 — Model Comparison Leaderboard**
- MAPE (forecast error %) for each model
- Bias (systematic over/under-forecasting)
- Accuracy Improvement vs. Baseline (MA7)
- Status badge: ✅ Champion selected

**💰 Tab 2 — Financial Impact Analysis**
- Working Capital Impact by model (stockout cost + excess inventory cost)
- Projected savings if you switch from MA7 to Champion model
- Risk scoring: which models minimize inventory holding cost?

**📈 Tab 3 — Forecast Decomposition**
- Prophet trend, weekly, and yearly seasonality components
- XGBoost feature importance (which variables drive demand?)
- Visual comparison: Actual vs. Forecast by model

**🎯 Tab 4 — Model Tournament**
- Head-to-head comparison of all models
- Filter by Category, ABC Class, forecast horizon
- Champion model recommendation with confidence score

---

## 📊 Key Metrics

| Metric | What It Means |
|--------|---------------|
| **MAPE** | Mean Absolute Percentage Error — lower is better (industry baseline: ~4-5%) |
| **Bias** | Systematic over/under-forecasting (negative = underforecasting, positive = overforecasting) |
| **Accuracy Improvement** | % improvement vs. traditional MA7 baseline |
| **Working Capital Impact** | Financial cost of forecast errors (stockout losses + excess inventory holding cost) |
| **Champion Model** | Best-performing model selected based on test-set MAPE |

---

## 🏗 Architecture

```
Raw Data (CSV/Excel)
    ↓
ETL Pipeline (Pandas)
    ├─ Prophet Model Training (Time Series Decomposition)
    ├─ XGBoost Model Training (ML with Feature Engineering)
    ├─ SARIMA Model Training (Statistical ARIMA)
    └─ MA7 Baseline (Traditional SAP method)
    ↓
Evaluation Metrics (MAPE, Bias, Improvement, Working Capital Impact)
    ↓
Streamlit Dashboard
    ├─ 4 Interactive Tabs
    ├─ Financial KPI Cards
    ├─ 5 Plotly Visualizations
    └─ Champion Model Selection
    ↓
Outputs
    ├─ Champion Forecast → Module 2 (Inventory Optimization)
    ├─ Model Rankings → Control Tower 360
    └─ Financial Impact → Executive QBR
```

---

## 📥 Input Format

Upload a CSV or Excel file (.csv, .xlsx) with these columns:

| Column | Type | Example |
|--------|------|---------|
| `Date` | DateTime | 2024-01-15 |
| `Units` | Integer | 1250 |
| `Category` | String | "Pain Relief" |
| `SKU` | String | "IBUPROFEN-500" |

**Optional columns** (auto-generated if missing):
- `Is_Promo` (1/0 for promotion days)
- `ABC_Classification` (A, B, or C)

If no file is uploaded, the app runs with **synthetic supply chain data** (historical demand from CPG example).

---

## 💻 Tech Stack

- **Frontend:** Streamlit (Python web framework)
- **ML/Forecasting:** Prophet (Facebook), XGBoost, SARIMA (StatsModels)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly (interactive charts), Matplotlib (decomposition)
- **Database:** SQLite (model performance tracking)
- **Styling:** Custom CSS (Glassmorphism UI)
- **Deployment:** Streamlit Community Cloud

---

## 🔄 Part of the Supply Chain AI Suite

| Module | Tool | Status | Link |
|--------|------|--------|------|
| **M1** | **Demand Planning — AI Forecast Comparator** | ✅ **LIVE** | [Demo](https://m1-demand-forecast-public-fyjvowtsbgsa6yfovy82xk.streamlit.app) |
| **M2** | Inventory Diagnosis & Coverage Analyzer | ✅ Live | [Demo](https://m2-inventory-diagnosis-public-emthenygqlck7srnw4dejt.streamlit.app) |
| **M3** | Replenishment Coach — Safety Stock Calculator | 🔄 In Dev | — |
| **M4** | Procurement — OTIF Risk Tracker | 🔄 In Dev | — |
| **M5** | Control Tower 360 — Executive Dashboard | 🔄 In Dev | — |

Module flow: **M1 → M2 → M3 → M4 → M5**

---

## 👤 About

**Sebastián Rueda** — Supply Chain AI Orchestrator

- 7+ years in **Pharma** (Novartis Colombia) and **CPG** (Kellanov)
- Pioneered **Kinaxis Maestro** implementation in Colombia
- Expert in: Demand Planning, Inventory Optimization, Supply Chain Analytics

**Connect:**
- [LinkedIn](https://www.linkedin.com/in/sebastiaan-rueda)
- [GitHub](https://github.com/SebsRu)

---

## 📜 License

Code available under **NDA** for commercial use. Contact for licensing.

---

## 🎓 Learn More

**Want to understand the math?**

**MAPE (Mean Absolute Percentage Error)**
- Formula: `Σ |Actual - Forecast| / Actual ÷ n × 100`
- Lower is better (industry baseline: 4-5%)
- Example: 3.2% MAPE is excellent for supply chain

**Prophet Model**
- Additive time series decomposition (Trend + Seasonal + Holiday)
- Handles missing data and outliers automatically
- Best for: Weekly, monthly, seasonal patterns

**XGBoost Model**
- Gradient boosting machine learning
- Feature engineering: lags (yesterday, last week), moving averages, promotions
- Best for: Complex patterns and external variables (promotions, events)

**SARIMA Model**
- Seasonal ARIMA: ARIMA(p,d,q) × ARIMA(P,D,Q)
- Handles trend and seasonal components
- Best for: Highly regular seasonal patterns

**Working Capital Impact**
- Formula: `(Stockout Days × Daily Demand × Unit Cost) + (Excess Units × Unit Cost × Holding %)`
- Quantifies the financial cost of forecast error
- Example: 1% MAPE improvement = $50K working capital freed

---

**Ready to optimize your demand forecasting?** 👉 [Launch the App](https://m1-demand-forecast-public-fyjvowtsbgsa6yfovy82xk.streamlit.app)
