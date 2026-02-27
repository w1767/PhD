# AGENTS.md

## Cursor Cloud specific instructions

This is a PhD research project for financial volatility time-series prediction. The codebase consists of 9 Jupyter notebooks (`.ipynb`) originally developed on Google Colab, plus PDF research papers.

### Key services and how to run them

- **Jupyter**: `jupyter lab --no-browser --port=8888` or `jupyter notebook --no-browser --port=8888`
- **Execute a notebook headlessly**: `jupyter execute <notebook>.ipynb --timeout=300`

### Linting

- Run `nbqa flake8 <notebook>.ipynb --max-line-length=150` to lint notebooks.
- Expect many E402 (import order) warnings — this is normal for Jupyter notebooks.

### Testing

- There are no automated test suites. Validation is done by executing notebooks.
- `Hurst.ipynb` is the best notebook for quick verification — it downloads live S&P 500/VIX data via `yfinance` and computes Hurst exponents. It has no Google Drive dependency.
- Most other notebooks (`ARIMA_LSTM_*`, `Normalised_Log_difference_*`, `FullPipeline_*`) require Google Drive-mounted data files (e.g., `VIX_daily.xlsx`, `VVIX.xlsx`) that are not in the repo.
- `MAIN_CODE_MMAR.ipynb` partially works without Google Drive (the `yfinance` sections), but also mounts Drive for some data.

### Known issues

- `pandas_datareader` has a compatibility issue with pandas 3.x (`deprecate_kwarg()` error). This only affects `MAIN_CODE_MMAR.ipynb` and is non-critical since `yfinance` is the primary data source.
- `tensorflow-addons==0.9.1` (referenced in the N-BEATS notebooks) is no longer available for modern Python/TF. Those specific cells will error.
- TensorFlow runs in CPU mode (no GPU/CUDA). Notebooks will work but training steps will be slower.
- `~/.local/bin` must be on PATH for `jupyter`, `nbqa`, and `flake8` commands.
