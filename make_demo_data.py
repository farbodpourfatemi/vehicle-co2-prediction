"""Create clearly synthetic data to test the pipeline; NOT real-world evidence."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_demo_data(path: Path, rows: int = 120) -> None:
    rng = np.random.default_rng(42)
    engine = np.round(rng.uniform(1.0, 4.0, rows), 1)
    cylandr = rng.choice([3, 4, 6, 8], rows)
    fuelcomb = np.round(3.5 + engine * 1.9 + rng.normal(0, 0.8, rows), 1)
    co2 = np.round(35 + fuelcomb * 21 + 3 * engine + rng.normal(0, 11, rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"engine": engine, "cylandr": cylandr,
                  "fuelcomb": fuelcomb, "out1": co2}).to_csv(path, index=False)
    print(f"Wrote {rows} SYNTHETIC example rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demo_co2.csv"))
    args = parser.parse_args()
    make_demo_data(args.out)
