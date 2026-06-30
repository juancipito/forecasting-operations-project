from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sample_synthetic_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["date"] = pd.to_datetime(featured["date"])
    featured["day_index"] = (featured["date"] - featured["date"].min()).dt.days
    featured["is_weekend"] = featured["day_of_week"].isin([5, 6]).astype(int)
    featured["sin_13"] = np.sin(2 * np.pi * featured["day_index"] / 13)
    featured["cos_13"] = np.cos(2 * np.pi * featured["day_index"] / 13)
    return featured


def main() -> None:
    df = add_features(pd.read_csv(DATA_PATH))
    train = df.iloc[:-28]
    test = df.iloc[-28:]
    features = ["day_index", "day_of_week", "is_weekend", "special_event", "baseline_forecast", "sin_13", "cos_13"]
    model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=3)
    model.fit(train[features], train["actual_contacts"])
    test = test.copy()
    test["model_forecast"] = model.predict(test[features]).round().astype(int)
    mae = mean_absolute_error(test["actual_contacts"], test["model_forecast"])
    mape = mean_absolute_percentage_error(test["actual_contacts"], test["model_forecast"])
    baseline_mae = mean_absolute_error(test["actual_contacts"], test["baseline_forecast"])
    OUTPUT_DIR.mkdir(exist_ok=True)
    test[["date", "actual_contacts", "baseline_forecast", "model_forecast"]].to_csv(OUTPUT_DIR / "forecast_holdout.csv", index=False)
    ax = test.set_index("date")[["actual_contacts", "baseline_forecast", "model_forecast"]].plot(figsize=(10, 5))
    ax.set_title("Synthetic operations forecast holdout")
    ax.set_ylabel("Contacts")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast_holdout.png", dpi=140)
    print(f"Model MAE: {mae:.2f}")
    print(f"Model MAPE: {mape:.3f}")
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"MAE improvement vs baseline: {baseline_mae - mae:.2f}")
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
