# Cloud Agent Setup

## Project Overview

PhD research project on volatility clustering and time-series prediction in financial markets. The codebase consists of Jupyter notebooks implementing ARIMA, LSTM, SVM, Random Forest, CNN, N-BEATS, MMAR, and Hurst exponent analyses on financial data (VIX, VVIX, etc.).

## Environment Setup

```bash
pip install -r requirements.txt
```

This installs all dependencies needed to run the notebooks: TensorFlow/Keras for deep learning, scikit-learn for classical ML, statsmodels and arch for econometrics, yfinance for market data, and matplotlib/seaborn for visualization.

## Key Notebooks

| Notebook | Purpose |
|----------|---------|
| `FullPipeline_ARIMA_LSTM_SVM_RF_CNN_07Sep25.ipynb` | Full forecasting pipeline (latest) |
| `MAIN_CODE_MMAR.ipynb` | Multifractal Model of Asset Returns |
| `Hurst.ipynb` | Hurst exponent estimation |
| `Normalised_Log_difference_NBeats_Forecasting_VVIX_VIX.ipynb` | N-BEATS forecasting on VIX/VVIX |
| `ARIMA_LSTM_SVM_RF_CNN_*.ipynb` | Iterative model development snapshots |

## Notes for Agents

- Notebooks may download live market data via `yfinance` at runtime — network access is required.
- Some notebooks were originally run in Google Colab; `google.colab` imports can be ignored in local/cloud-agent environments.
- TensorFlow can be slow to install; the VM snapshot should already have it cached.
