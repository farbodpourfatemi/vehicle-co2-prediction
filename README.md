# Vehicle CO2 Prediction (Linear Regression)

An educational machine-learning project exploring baseline CO2 prediction using vehicle characteristics. This refactors several regression exercises from my Aiolearn coursework into one reproducible script.

## Method

- Inputs: engine size (`engine`), number of cylinders (`cylandr`, matching the source dataset spelling), and combined fuel use (`fuelcomb`).
- Target: CO2 value (`out1`). Units should be verified against the dataset documentation before any real-world interpretation.
- Split: 80% train / 20% held-out test, with `random_state=42`.
- Evaluation: MAE, RMSE, R², and an actual-versus-predicted plot.
- Limitation: linear regression is a baseline, not a validated emissions estimator.

## Run

```bash
python -m pip install -r requirements.txt
python make_demo_data.py --out demo_co2.csv
python predict_co2.py --csv demo_co2.csv
```

**Important:** `demo_co2.csv` is randomly generated **synthetic** data for checking the pipeline; do not report metrics from it as real vehicle-performance results.

To run against the original course dataset, use `python predict_co2.py --csv path/to/co2.csv` on your computer. The original CSV has not been included here because its license/redistribution status was not provided. Check permissions before uploading that dataset.

## Improvements from the lesson scripts

- Removed repeated blocks and interactive plotting.
- Added a deterministic split, metrics and a saved visualization.
- Validated the CSV schema and added input error handling.
- Separated model training from synthetic demo-data generation.

**Credits:** Based on personal coursework; refactored and documented with AI assistance.
