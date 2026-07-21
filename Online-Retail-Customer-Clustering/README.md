
# Online Retail Customer Clustering

Modular, production-style version of the original `online-retail-data-clustering.ipynb`
notebook. The notebook logic has been split into reusable Python components, an
orchestrated training pipeline, and a prediction pipeline for scoring new customers.

## Project structure

```
retail_clustering/
├── data/
│   └── online_retail_II.xlsx        # place the source dataset here
├── artifacts/
│   ├── data/                        # intermediate + final CSV outputs
│   ├── models/                      # saved scaler.pkl, kmeans_model.pkl, outlier_bounds.pkl
│   └── plots/                       # saved PNG charts (optional)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # reads raw Excel -> CSV artifact
│   │   ├── data_cleaning.py         # invoice/stock code/customer id/price rules
│   │   ├── feature_engineering.py   # builds RFM (Recency, Frequency, Monetary) features
│   │   ├── data_transformation.py   # IQR outlier split + StandardScaler
│   │   ├── model_trainer.py         # elbow/silhouette search + final KMeans + labeling
│   │   └── visualization.py         # optional plots (saved as PNG, headless)
│   ├── pipeline/
│   │   ├── training_pipeline.py     # orchestrates the full training workflow
│   │   └── predict_pipeline.py      # scores new customers using saved artifacts
│   ├── exception.py                 # CustomException with file/line context
│   ├── logger.py                    # logging to logs/ and stdout
│   └── utils.py                     # save_object / load_object (pickle helpers)
├── main.py                          # entry point: runs the training pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Download the "Online Retail II" dataset (UCI ML repository) and place it at
`data/online_retail_II.xlsx`.

## Usage

### 1. Train

```bash
python main.py
```

This runs, in order:

1. **Data ingestion** – reads the Excel file, saves `artifacts/data/raw_data.csv`.
2. **Data cleaning** – drops invalid invoices/stock codes, missing customer IDs,
   and non-positive prices.
3. **Feature engineering** – aggregates invoice lines into per-customer RFM
   features (MonetaryValue, Frequency, Recency).
4. **Data transformation** – splits out IQR outliers (monetary/frequency/both)
   and standard-scales the remaining "core" customers. Saves `scaler.pkl` and
   `outlier_bounds.pkl`.
5. **Model training** – searches k=2..12 for inertia/silhouette scores, fits
   the final KMeans (k=4 by default), assigns cluster ids to every customer
   (including outlier segments), and maps ids to business labels:
   `RETAIN`, `RE-ENGAGE`, `NURTURE`, `REWARD`, `PAMPER`, `UPSELL`, `DELIGHT`.
   Saves `kmeans_model.pkl` and the final labeled dataframe.

### 2. Visualize (optional)

```bash
python -m src.components.visualization
```

Saves distribution/boxplot/elbow/violin/summary charts to `artifacts/plots/`.

### 3. Predict segments for new customers

```python
from src.pipeline.predict_pipeline import PredictPipeline
import pandas as pd

pipeline = PredictPipeline()

# Option A: you already have RFM values per customer
new_customers = pd.DataFrame({
    "MonetaryValue": [5000.0, 120.0],
    "Frequency": [15, 1],
    "Recency": [5, 200],
})
result = pipeline.predict_from_rfm(new_customers)
print(result)

# Option B: you have raw invoice-line transactions (same shape as the source data)
# result = pipeline.predict_from_transactions(raw_transactions_df)
```

## Notes

* All configuration (file paths, k range, IQR multiplier, etc.) lives in small
  `@dataclass` config objects at the top of each component, so paths/params can
  be overridden without touching the core logic.
* `src/logger.py` writes timestamped logs under `logs/` for every run.
* The original notebook (`notebooks/online-retail-data-clustering.ipynb`) is kept
  for reference/EDA but is no longer required to run the project.
