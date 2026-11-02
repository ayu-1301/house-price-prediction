"""Train California housing models from the notebook pipeline and save them here."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"
MODELS_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
]
OCEAN_CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_and_prepare() -> pd.DataFrame:
    housing_pd = pd.read_csv(DATA_PATH)
    housing_pd_shuffled = housing_pd.sample(n=len(housing_pd), random_state=1)
    dummies = pd.get_dummies(housing_pd_shuffled["ocean_proximity"]).astype(int)
    for category in OCEAN_CATEGORIES:
        if category not in dummies.columns:
            dummies[category] = 0
    housing_pd_final = pd.concat(
        [housing_pd_shuffled.drop("ocean_proximity", axis=1), dummies],
        axis=1,
    )
    housing_pd_final = housing_pd_final[FEATURE_COLUMNS + ["median_house_value"]]
    return housing_pd_final.dropna()


def split_features(housing_pd_final: pd.DataFrame):
    train_pd = housing_pd_final[:18000]
    test_pd = housing_pd_final[18000:19217]
    val_pd = housing_pd_final[19215:]

    X_train, y_train = train_pd.to_numpy()[:, :-1], train_pd.to_numpy()[:, -1]
    X_val, y_val = val_pd.to_numpy()[:, :-1], val_pd.to_numpy()[:, -1]
    X_test, y_test = test_pd.to_numpy()[:, :-1], test_pd.to_numpy()[:, -1]
    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_numeric(X_train, X_val, X_test):
    scaler = StandardScaler().fit(X_train[:, :8])

    def preprocessor(X):
        A = np.copy(X)
        A[:, :8] = scaler.transform(A[:, :8])
        return A

    return (
        scaler,
        preprocessor(X_train),
        preprocessor(X_val),
        preprocessor(X_test),
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Place housing.csv in the data folder first."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    housing_pd_final = load_and_prepare()
    X_train, y_train, X_val, y_val, X_test, y_test = split_features(housing_pd_final)
    scaler, X_train, X_val, X_test = scale_numeric(X_train, X_val, X_test)

    models = {
        "linear_regression": LinearRegression(),
        "knn": KNeighborsRegressor(n_neighbors=10),
        "random_forest": RandomForestRegressor(max_depth=10),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=250),
    }

    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        row = {
            "model": name,
            "train_rmse": rmse(y_train, model.predict(X_train)),
            "val_rmse": rmse(y_val, model.predict(X_val)),
            "test_rmse": rmse(y_test, model.predict(X_test)),
        }
        metrics.append(row)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(
            f"{name}: train={row['train_rmse']:.2f} "
            f"val={row['val_rmse']:.2f} test={row['test_rmse']:.2f}"
        )

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_columns.joblib")
    pd.DataFrame(metrics).to_csv(MODELS_DIR / "metrics.csv", index=False)
    print(f"Saved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
