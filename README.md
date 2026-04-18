# 📈 Financial Market Intelligence System

A two-phase machine learning pipeline for **stock price trend prediction** and **fraud/anomaly detection**, combining classical ML with deep learning techniques on real-world financial datasets.

---

## 🗂️ Project Overview

This project tackles two core challenges in financial AI:

1. **Stock Return Regression** — Predict next-day returns for equities and cryptocurrencies using technical indicators and ensemble/deep learning models.
2. **Fraud & Anomaly Detection** — Identify fraudulent credit card transactions using both supervised and unsupervised approaches.

---

## 📦 Datasets

| Dataset | Source | Task |
|---|---|---|
| Yahoo Finance All-Stocks | Kaggle | Stock return regression |
| Credit Card Fraud (ULB) | Kaggle | Fraud classification |
| Crypto Market Data | Kaggle | Crypto return regression |

---

## 🧱 Pipeline Architecture

### Phase 1 — Classical ML

```
Raw Data → EDA → Feature Engineering → Classical Models → Evaluation
```

- **Exploratory Data Analysis** — distributions, rolling statistics, correlation heatmaps, outlier detection
- **Feature Engineering** — RSI (14), MACD (12/26/9), ATR (14), moving averages, lagged prices, volatility
- **KNN Regressor** — applied to crypto return prediction (full features vs PCA-reduced)
- **Naïve Bayes Classifier** — applied to fraud detection (baseline vs PCA-reduced)
- **Ensemble Regressors** — Random Forest + XGBoost on Yahoo Finance stock returns
- **Hyperparameter Tuning** — via [Optuna](https://optuna.org/)
- **Dimensionality Reduction** — PCA for feature compression and comparison

### Phase 2 — Deep Learning

```
Scaled Features → Sliding Window → 1D-CNN → Evaluation
Normal Transactions → AutoEncoder → Threshold → Anomaly Detection
AAPL CNN → Freeze Conv Layers → Fine-Tune on TSLA (Transfer Learning)
```

- **1D-CNN** — multiple architectures (linear, multi-output, windowed) for time-series regression
- **AutoEncoder** — unsupervised fraud detection trained only on legitimate transactions; anomaly threshold set at the 99th percentile of reconstruction error
- **Transfer Learning** — convolutional layers trained on AAPL are frozen and fine-tuned on TSLA to transfer market pattern knowledge across assets

---

## 🔧 Technical Indicators

| Function | Indicator | Parameters |
|---|---|---|
| `calc_rsi` | Relative Strength Index | 14-period EWM |
| `calc_macd` | MACD Line, Signal, Histogram | 12/26/9 EWM |
| `calc_atr` | Average True Range | 14-period rolling |

---

## 🧠 Models Summary

| Model | Task | Dataset |
|---|---|---|
| KNN Regressor | Return prediction | Crypto |
| Gaussian Naïve Bayes | Fraud classification | Credit Card |
| Random Forest | Return prediction | Yahoo Finance |
| XGBoost | Return prediction | Yahoo Finance |
| 1D-CNN (Linear) | Return regression | Yahoo Finance |
| 1D-CNN (Windowed, 30-step) | Return regression | Yahoo Finance |
| Transfer Learning CNN | Return regression | AAPL → TSLA |
| AutoEncoder | Anomaly detection | Credit Card Fraud |

---

## 📊 Evaluation Metrics

- **Regression:** R², MAE, MSE, RMSE
- **Classification:** Accuracy, Precision, Recall, F1, AUC-ROC
- **Anomaly Detection:** Reconstruction Error, ROC-AUC, TPR/FPR at chosen threshold

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost optuna tensorflow
```

### Running the Notebook

1. Download the required datasets from Kaggle and place them at the paths referenced in **Step 2** of the notebook.
2. Open the notebook in **Kaggle**, **Jupyter**, or any compatible environment.
3. Run cells sequentially from **Step 1 (Imports)** through **Step 8 (Results Summary)**.

> **GPU recommended** — the deep learning steps (1D-CNN, AutoEncoder) benefit significantly from GPU acceleration.

### Dataset Paths (default)

```
/kaggle/input/datasets/tanavbajaj/yahoo-finance-all-stocks-dataset-daily-update/*.csv
/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv
/kaggle/input/datasets/anasshahid88/crypto-market-data/**/*.csv
```

---

## 📁 Notebook Structure

| Step | Description |
|---|---|
| Step 1 | Imports & Configuration |
| Step 2 | Data Collection & Loading |
| Step 3 | Exploratory Data Analysis (EDA) — Stocks, Fraud, Crypto |
| Step 4 | Feature Engineering — Technical Indicators |
| Step 5 | Classical ML — KNN, Naïve Bayes, PCA |
| Step 6 | Ensemble Models — Random Forest, XGBoost + Optuna Tuning |
| Step 7 | Deep Learning — 1D-CNN, Transfer Learning, AutoEncoder |
| Step 8 | Final Results Summary |

---

## 🔬 Key Design Decisions

- **Time-series integrity** — `TimeSeriesSplit` is used for cross-validation to avoid data leakage.
- **Robust scaling** — `RobustScaler` is used for the fraud dataset to handle its heavy-tailed distributions.
- **AutoEncoder training discipline** — the model is trained exclusively on legitimate transactions so that fraud signals emerge purely from elevated reconstruction error.
- **Transfer Learning rationale** — momentum and volatility regime patterns are assumed to be shared across equities in the same market, making AAPL a suitable source for TSLA fine-tuning.

---

## 📄 License

This project is intended for educational and research purposes.
