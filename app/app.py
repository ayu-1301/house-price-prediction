from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "housing.csv"

MODEL_FILES = {
    "Gradient Boosting": "gradient_boosting.joblib",
    "Random Forest": "random_forest.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Linear Regression": "linear_regression.joblib",
}

OCEAN_OPTIONS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")
    models = {
        name: joblib.load(MODELS_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    metrics = pd.read_csv(MODELS_DIR / "metrics.csv")
    return scaler, feature_columns, models, metrics


@st.cache_data
def load_sample_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def encode_features(values: dict, scaler) -> np.ndarray:
    row = [values[name] for name in NUMERIC_FEATURES]
    for category in OCEAN_OPTIONS:
        row.append(1.0 if values["ocean_proximity"] == category else 0.0)
    features = np.array(row, dtype=float).reshape(1, -1)
    features[:, :8] = scaler.transform(features[:, :8])
    return features


def format_usd(amount: float) -> str:
    return f"${amount:,.0f}"


st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero-card {
        background: linear-gradient(135deg, #0e4d5c 0%, #147d7b 55%, #c9a227 100%);
        color: white;
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
    }
    .metric-box {
        background: #f4f7f7;
        border: 1px solid #d7e4e3;
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
      <h1 style="margin:0;">California House Price Predictor</h1>
      <p style="margin:0.4rem 0 0 0; opacity:0.95;">
        Estimate median house value from census-block features using the models
        trained in this project.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    scaler, feature_columns, models, metrics = load_artifacts()
except FileNotFoundError:
    st.error(
        "Trained models were not found. Run `python models/train.py` from the project root first."
    )
    st.stop()

housing = load_sample_data()
metrics = metrics.copy()
metrics["label"] = metrics["model"].map(
    {
        "gradient_boosting": "Gradient Boosting",
        "random_forest": "Random Forest",
        "knn": "K-Nearest Neighbors",
        "linear_regression": "Linear Regression",
    }
)

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    selected_metrics = metrics.loc[metrics["label"] == model_name].iloc[0]
    st.caption("Lower RMSE is better. Gradient Boosting had the best test score.")
    st.metric("Validation RMSE", f"${selected_metrics['val_rmse']:,.0f}")
    st.metric("Test RMSE", f"${selected_metrics['test_rmse']:,.0f}")
    st.divider()
    st.markdown(
        "Median income is entered in **tens of thousands of USD**, matching the original dataset. "
        "Example: `8.32` means about $83,200."
    )

predict_tab, compare_tab = st.tabs(["Predict price", "Compare models"])

with predict_tab:
    defaults = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }
    if housing is not None:
        sample = housing.dropna().iloc[0]
        defaults.update(
            {
                "longitude": float(sample["longitude"]),
                "latitude": float(sample["latitude"]),
                "housing_median_age": float(sample["housing_median_age"]),
                "total_rooms": float(sample["total_rooms"]),
                "total_bedrooms": float(sample["total_bedrooms"]),
                "population": float(sample["population"]),
                "households": float(sample["households"]),
                "median_income": float(sample["median_income"]),
                "ocean_proximity": str(sample["ocean_proximity"]),
            }
        )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("District details")
        col1, col2 = st.columns(2)
        with col1:
            longitude = st.number_input(
                "Longitude",
                min_value=-124.5,
                max_value=-114.0,
                value=defaults["longitude"],
                step=0.01,
                format="%.2f",
            )
            housing_median_age = st.slider(
                "Housing median age",
                min_value=1,
                max_value=52,
                value=int(defaults["housing_median_age"]),
            )
            total_rooms = st.number_input(
                "Total rooms",
                min_value=1.0,
                value=defaults["total_rooms"],
                step=10.0,
            )
            population = st.number_input(
                "Population",
                min_value=1.0,
                value=defaults["population"],
                step=10.0,
            )
        with col2:
            latitude = st.number_input(
                "Latitude",
                min_value=32.5,
                max_value=42.0,
                value=defaults["latitude"],
                step=0.01,
                format="%.2f",
            )
            median_income = st.number_input(
                "Median income (tens of thousands USD)",
                min_value=0.5,
                max_value=15.0,
                value=float(defaults["median_income"]),
                step=0.1,
                format="%.4f",
            )
            total_bedrooms = st.number_input(
                "Total bedrooms",
                min_value=1.0,
                value=defaults["total_bedrooms"],
                step=5.0,
            )
            households = st.number_input(
                "Households",
                min_value=1.0,
                value=defaults["households"],
                step=5.0,
            )

        ocean_proximity = st.selectbox(
            "Ocean proximity",
            OCEAN_OPTIONS,
            index=OCEAN_OPTIONS.index(defaults["ocean_proximity"])
            if defaults["ocean_proximity"] in OCEAN_OPTIONS
            else 0,
        )
        predict_clicked = st.button("Predict house value", type="primary", use_container_width=True)

    values = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }

    with right:
        st.subheader("Estimate")
        if predict_clicked:
            features = encode_features(values, scaler)
            prediction = float(models[model_name].predict(features)[0])
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Predicted median house value", format_usd(prediction))
            st.caption(f"Using {model_name}")
            st.markdown("</div>", unsafe_allow_html=True)

            all_preds = {
                name: float(model.predict(features)[0])
                for name, model in models.items()
            }
            st.dataframe(
                pd.DataFrame(
                    {"Model": all_preds.keys(), "Predicted value": [format_usd(v) for v in all_preds.values()]}
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Fill in the district details, then click **Predict house value**.")

        st.map(
            pd.DataFrame({"lat": [latitude], "lon": [longitude]}),
            zoom=6,
            height=280,
        )

with compare_tab:
    st.subheader("Held-out model performance")
    display = metrics[["label", "train_rmse", "val_rmse", "test_rmse"]].rename(
        columns={
            "label": "Model",
            "train_rmse": "Train RMSE",
            "val_rmse": "Validation RMSE",
            "test_rmse": "Test RMSE",
        }
    )
    st.dataframe(
        display.style.format(
            {
                "Train RMSE": "${:,.0f}",
                "Validation RMSE": "${:,.0f}",
                "Test RMSE": "${:,.0f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    chart_data = display.set_index("Model")[["Validation RMSE", "Test RMSE"]]
    st.bar_chart(chart_data)

    if housing is not None:
        st.subheader("Sample of the training data")
        st.dataframe(housing.head(10), use_container_width=True)
