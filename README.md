# 🚚 Delivery Delay Prediction & Logistics Risk Analytics

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=30&duration=3500&pause=1000&color=00BFFF&center=true&vCenter=true&width=950&lines=Delivery+Delay+Prediction+%26+Analytics;Supply+Chain+%7C+Route+%26+Traffic+Diagnostics;Leakage-Free+Classification+Pipeline;Python+%7C+Pandas+%7C+Scikit-Learn+%7C+Seaborn" alt="Typing SVG" />
</p>

<p align="center">
  <b>Developed by Neeraj Singh Danu</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas"/>
  <img src="https://img.shields.io/badge/NumPy-Numerical-blue?style=for-the-badge&logo=numpy"/>
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Seaborn-Statistical-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Classification-F7931E?style=for-the-badge&logo=scikitlearn"/>
</p>

---

## 🌍 Project Banner

<p align="center">
  <img src="images/banner.png" width="100%" alt="Delivery Delay Prediction Banner"/>
</p>

---

# 👨‍💻 About the Project

**Delivery Delay Prediction & Logistics Risk Analytics** is an end-to-end Exploratory Data Analysis (EDA) and Supervised Machine Learning classification project developed by **Neeraj Singh Danu**.

Last-mile and regional delivery delays directly impact customer satisfaction, SLA compliance, and operational costs. This project analyzes shipment-level logistics telemetry (`DATASET.csv`)—covering route distance, package weight, real-time traffic congestion, warehouse load, driver experience, multi-stop complexity, weather conditions, and vehicle/zone categories—to predict whether a shipment will be **On-Time (`0`)** or **Delayed (`1`)** before dispatch.

---

# 📖 Project Overview

An end-to-end **Data Quality Sanitization, Exploratory Data Analysis (EDA), and Binary Classification** workflow built on `DATASET.csv`.

This project covers:

- 🧹 **Data Quality & Anomaly Sanitization** (deduplication, negative-value coercion, string whitespace/case normalization)
- 🔒 **Leakage Prevention** (removing post-delivery `late_delivery_penalty`, ID column `record_id`, and synthetic `random_noise_feature`)
- 📊 **Univariate & Bivariate EDA** across operational, route, and environmental drivers
- 🔥 **Correlation & Class-Separation Analysis** (`delayed = 0` vs. `delayed = 1`)
- ⚙️ **Leakage-Free Train / Validation / Test Splitting & Imputation**
- 🤖 **Multi-Classifier Benchmarking** (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- 📈 **Threshold & Diagnostic Evaluation** (Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix)

---

# 📂 Dataset Architecture & Schema

| Property | Value |
| :--- | :--- |
| **Source File** | `DATASET.csv` |
| **Raw Dataset Shape** | `748 rows × 12 columns` |
| **After Deduplication** | `740 unique shipment records × 12 columns` (`8 exact duplicate rows removed`) |
| **Modeling Features Used** | `8 predictive features` (`6 numerical` + `2 categorical`) |
| **Target Variable** | `delayed` (`0` = On-Time: **462 [62.43%]**, `1` = Delayed: **278 [37.57%]**) |

### Feature Breakdown & Data Quality Audit

| Feature | Type | Raw Missing | Raw Anomalies / Formatting Issues | Preprocessing Strategy |
| :--- | :---: | :---: | :--- | :--- |
| `distance_km` | Numerical | `25` | `3` negative values (`min = -38.19`, `median = 39.71`) | Coerce `< 0` to `NaN` → Train Median Imputed |
| `package_weight` | Numerical | `25` | `4` negative values (`min = -160.21`, `median = 98.25`) | Coerce `< 0` to `NaN` → Train Median Imputed |
| `traffic_index` | Numerical | `25` | `3` negative values (`min = -32.09`, `median = 50.50`) | Coerce `< 0` to `NaN` → Train Median Imputed |
| `warehouse_load` | Numerical | `25` | `3` negative values (`min = -27.78`, `median = 24.79`) | Coerce `< 0` to `NaN` → Train Median Imputed |
| `driver_experience` | Numerical | `26` | `4` negative values (`min = -18.18`, `median = 10.07`) | Coerce `< 0` to `NaN` → Train Median Imputed |
| `route_stops` | Numerical | `25` | `3` negative values (`min = -43.96`, `median = 30.00`) | Coerce `< 0` to `NaN` → Train Median Imputed |
| `weather` | Categorical | `25` | Dirty casing/spaces (`'A'`, `'a'`, `' A'`, `'B'`, `'b'`, `'C'`, `'c'`) | Strip whitespace & uppercase (`A`, `B`, `C`) → Train Mode Imputed → One-Hot Encoded |
| `vehicle_type` | Categorical | `25` | Dirty casing/spaces (`'East'`, `'east'`, `'EAST'`, `' West'`, etc.) | Strip whitespace & title-case (`North`, `South`, `East`, `West`) → Train Mode Imputed → One-Hot Encoded |

### 🚫 Excluded Columns (Target Leakage & Noise Prevention)

| Column | Reason for Exclusion |
| :--- | :--- |
| `record_id` | Unique shipment identifier (`C1200001`–`C1200740`) carrying zero predictive signal. |
| `late_delivery_penalty` | **Direct Target Leakage** — post-delivery penalty flag (`Yes`/`No` with casing variants) directly mirrors `delayed`. |
| `random_noise_feature` | Standard normal random noise ($\mu \approx 0.03, \sigma \approx 1.03$) dropped to prevent spurious splits and overfitting. |

---

# 🎯 Objectives

The primary objectives of this project are:

- 🔍 Audit and clean corrupted logistics records (duplicates, negative physical measurements, inconsistent text labels)
- 🚫 Identify and eliminate post-outcome leakage features (`late_delivery_penalty`) before model training
- 📊 Analyze how `traffic_index`, `distance_km`, `warehouse_load`, `route_stops`, and `driver_experience` differ between on-time and delayed shipments
- 🌦️ Evaluate delay rates across `weather` tiers (`A`, `B`, `C`) and `vehicle_type` / dispatch zones (`North`, `South`, `East`, `West`)
- 🔒 Build a leakage-free stratified preprocessing and classification pipeline
- 🤖 Benchmark linear and tree-based ensemble classifiers using **F1-Score, ROC-AUC, Precision, and Recall**

---

# 🛠 Tech Stack

| Technology | Purpose |
| :--- | :--- |
| 🐍 **Python 3.11** | Core Programming & Modeling |
| 🐼 **Pandas** | Data Cleaning, Deduplication & Feature Engineering |
| 🔢 **NumPy** | Numerical Coercion & Array Operations |
| 📊 **Matplotlib** | Custom Diagnostic & Distribution Plots |
| 📈 **Seaborn** | Statistical Boxplots, Countplots & Correlation Heatmaps |
| 🤖 **Scikit-Learn** | Stratified Splitting, Imputation, Scaling & Classification Models |
| 📓 **Jupyter Notebook** | Interactive Analysis & Reproducible Workflow |

---

# 🧹 Data Preparation & Leakage-Free Pipeline

```text
Raw Dataset (DATASET.csv: 748 × 12)
   │
   ├──► 1. Exact Deduplication ──► 740 unique rows (8 duplicate records removed)
   ├──► 2. Drop Non-Predictive & Leakage Columns (`record_id`, `late_delivery_penalty`, `random_noise_feature`)
   ├──► 3. Domain Sanity Rules (Coerce negative numerical values to NaN; normalize string casing & whitespace)
   ├──► 4. Stratified 3-Way Split (Train / Validation / Test preserving ~62.4% vs 37.6% class balance)
   ├──► 5. Fit Imputers & Encoders Strictly on Train (Median for numeric, Mode for categorical, One-Hot Encoding)
   └──► 6. Model Selection on Validation ──► Retrain Winner on (Train + Val) ──► Final Test Evaluation
```

### Key Engineering Highlights
- **Physical Range Enforcement:** Negative values in physical metrics (`distance_km`, `package_weight`, `traffic_index`, `warehouse_load`, `driver_experience`, `route_stops`) represent corrupted entries and were safely coerced to `NaN` prior to median imputation.
- **Categorical Canonicalization:** String normalization collapsed `9` dirty `weather` variants into `3` clean categories (`A`, `B`, `C`) and `16` dirty `vehicle_type` variants into `4` clean categories (`East`, `West`, `North`, `South`).
- **Zero Statistical Leakage:** All medians, modes, feature scalers, and dummy column schemas were fit exclusively on the training partition.

---

# 📊 Exploratory Data Analysis

The notebook investigates key operational dimensions:

### 🚦 Traffic, Distance & Route Complexity
- Distribution of `distance_km` (`median = 39.71 km`), `traffic_index` (`median = 50.50`), and `route_stops` (`median = 30 stops`) across `delayed = 0` vs. `delayed = 1`
- Identification of high-congestion and multi-stop delay thresholds

### 🏭 Warehouse Load & Driver Experience
- Impact of `warehouse_load` bottlenecks (`median = 24.79`) on dispatch delays
- Whether higher `driver_experience` (`median = 10.07 years`) mitigates delay risk on complex routes

### 🌦️ Weather & Vehicle / Zone Breakdown
- Delay rate comparison across normalized `weather` categories (`A`, `B`, `C`)
- Operational delay distribution across `vehicle_type` / regional corridors (`North`, `South`, `East`, `West`)

### 🔗 Correlation & Multivariate Diagnostics
- Correlation heatmap across cleaned numerical features and target `delayed`

---

# 📈 Visualizations

## 📦 Target Class Distribution (`On-Time` vs. `Delayed`)

<p align="center">
  <img src="images/target_class_distribution.png" width="700" alt="Target Class Distribution">
</p>

---

## 🚦 Traffic Index & Distance vs. Delivery Delay

<p align="center">
  <img src="images/traffic_distance_delay.png" width="700" alt="Traffic and Distance vs Delay">
</p>

---

## 🏭 Warehouse Load & Driver Experience Diagnostics

<p align="center">
  <img src="images/warehouse_driver_analysis.png" width="700" alt="Warehouse Load and Driver Experience">
</p>

---

## 🌦️ Delay Rates by Weather & Vehicle Type

<p align="center">
  <img src="images/categorical_delay_rates.png" width="700" alt="Delay Rates by Weather and Vehicle Type">
</p>

---

## 🔥 Feature Correlation Heatmap

<p align="center">
  <img src="images/correlation_heatmap.png" width="850" alt="Correlation Heatmap">
</p>

---

# 🤖 Model Benchmarking & Evaluation

Four classification algorithms were trained and compared on validation performance:

| Model | Paradigm | Key Strengths |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear Classifier (with `StandardScaler`) | Interpretable log-odds coefficients; calibrated probability baseline |
| **Decision Tree Classifier** | Non-Linear Rule-Based Tree | Captures threshold splits across traffic, stops, and warehouse load |
| **Random Forest Classifier** | Bagged Tree Ensemble | Reduces variance and captures feature interactions robustly |
| **Gradient Boosting Classifier** | Sequential Boosted Ensemble | Iteratively corrects hard-to-classify borderline delayed shipments |

### Evaluation Framework
- **Primary Selection Metrics:** **Validation F1-Score & ROC-AUC** (balancing Precision and Recall on the `37.57%` delayed minority class)
- **Diagnostic Checks:** Confusion Matrix, Precision-Recall trade-off, ROC Curve, and Feature Importance ranking

---

# 💡 Key Takeaways

- 🧹 **Data Hygiene is Critical:** Raw logistics data contained `8` exact duplicate shipments, `~25` missing entries per feature, negative physical values, and `25+` inconsistent text casing variants—cleaning these prevented silent category fragmentation and biased imputations.
- 🔒 **Eliminating `late_delivery_penalty` Prevents False Accuracy:** Including post-event penalty columns would artificially inflate accuracy to ~100% while rendering the model useless before delivery; dropping it ensures genuine pre-dispatch predictive power.
- 🎯 **Balanced Metric Focus:** With a `62.4% / 37.6%` class split, optimizing **F1-Score** and **Recall** ensures high-risk shipments are flagged early without excessive false alarms.

---

# 🚀 Future Work

This project can be extended with:

- ⚡ Gradient Boosted frameworks (`XGBoost`, `LightGBM`, `CatBoost`) with `StratifiedKFold` hyperparameter tuning
- 🎚️ Probability threshold optimization aligned to business SLA penalty costs
- 🧠 **SHAP** (SHapley Additive exPlanations) for shipment-level delay driver explanations
- 🌐 Interactive **Streamlit** dispatch risk-scoring application for operations teams

---

# 📁 Repository Structure

```text
DELIVERY_DELAY_PREDCITION-/
│
├── DATASET.csv                          # Raw shipment & logistics dataset (748 rows × 12 columns)
├── Delivery_Delay_Prediction.ipynb      # End-to-end EDA, data cleaning, modeling & evaluation notebook
├── README.md                            # Comprehensive project documentation
│
└── images/                              # Saved EDA & model evaluation visualizations
    ├── banner.png
    ├── target_class_distribution.png
    ├── traffic_distance_delay.png
    ├── warehouse_driver_analysis.png
    ├── categorical_delay_rates.png
    └── correlation_heatmap.png
```

---

# 📦 Requirements & Quickstart

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

### Install & Run Locally

```bash
# 1. Clone the repository
git clone [https://github.com/NEERAJSINGHDANU07/DELIVERY_DELAY_PREDCITION-.git](https://github.com/NEERAJSINGHDANU07/DELIVERY_DELAY_PREDCITION-.git)
cd DELIVERY_DELAY_PREDCITION-

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch Jupyter Notebook
jupyter notebook
```

---

# 👨‍💻 Author

## **Neeraj Singh Danu**

<p align="center">
  <b>Data Science • Machine Learning • Python • SQL • EDA • Data Visualization • MLOps</b>
</p>

Passionate about **Data Science, Machine Learning, and Analytics**, with a focus on transforming raw operational data into meaningful insights and building practical data-driven solutions.

---

<p align="center">
  ⭐ <b>If you found this project useful, consider giving the repository a Star!</b> ⭐
</p>

<p align="center">
  <b>Built with 🐍 Python & 📊 Data Science</b>
</p>
