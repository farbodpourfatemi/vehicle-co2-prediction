"""Train and evaluate a baseline vehicle CO2 regression model.

Expects columns engine, cylandr, fuelcomb, out1 in the CSV. The original
course dataset is not bundled because its redistribution rights are unknown.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Works without a desktop display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["engine", "cylandr", "fuelcomb"]
TARGET = "out1"


def run(csv_file: Path, output_dir: Path) -> dict[str, float]:
    df = pd.read_csv(csv_file)
    required = set(FEATURES + [TARGET])
    if not required.issubset(df.columns):
        raise ValueError(f"CSV is missing columns: {sorted(required - set(df.columns))}")
    data = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 20:
        raise ValueError("At least 20 valid rows are required.")
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "r2": float(r2_score(y_test, predicted)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, predicted, alpha=0.7)
    lower = min(y_test.min(), predicted.min())
    upper = max(y_test.max(), predicted.max())
    plt.plot([lower, upper], [lower, upper], linestyle="--", color="black")
    plt.xlabel("Actual CO2")
    plt.ylabel("Predicted CO2")
    plt.title("Linear regression: test-set predictions")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()
    example = pd.DataFrame([{"engine": 2.0, "cylandr": 4, "fuelcomb": 8.5}])
    print(f"Example estimate (illustrative): {model.predict(example)[0]:.2f}")
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.3f}")
    print(f"Plot saved to {output_dir / 'actual_vs_predicted.png'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Path to the vehicle dataset")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    try:
        run(args.csv, args.out)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
