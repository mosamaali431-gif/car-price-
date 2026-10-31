from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "car_price_model.pkl"

# Your model's predictions are around 10-13, which means it was trained on a
# LOG-transformed price (np.log1p). Set this to False if you trained on raw price.
LOG_TARGET = True

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


# ----------------------------------------------------------------------------
# Load model once (cached so it is not reloaded on every click)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
FEATURES = list(model.feature_names_in_)

# Build the dropdown options directly from the model's own column names,
# so the app can never drift out of sync with the model.
MAKES = sorted(c.removeprefix("make_") for c in FEATURES if c.startswith("make_"))
SIZES = sorted(c.removeprefix("vehicle_size_") for c in FEATURES if c.startswith("vehicle_size_"))
TRANSMISSIONS = sorted(
    c.removeprefix("transmission_type_") for c in FEATURES if c.startswith("transmission_type_")
)


def build_input_row(year, engine_hp, make, size, transmission) -> pd.DataFrame:
    """Turn the form values into the exact 58-column one-hot row the model expects."""
    row = dict.fromkeys(FEATURES, 0)
    row["year"] = year
    row["engine_hp"] = engine_hp
    for col in (f"make_{make}", f"vehicle_size_{size}", f"transmission_type_{transmission}"):
        if col in row:
            row[col] = 1
    return pd.DataFrame([row])[FEATURES]  # keep the training column order


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🚗 Car Price Predictor")
st.write("Enter the car's details and the model will estimate its price.")

with st.form("car_form"):
    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox("Make", MAKES, index=MAKES.index("Toyota") if "Toyota" in MAKES else 0)
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2015, step=1)
        engine_hp = st.number_input("Engine horsepower (HP)", min_value=50, max_value=1200, value=200, step=5)
    with col2:
        size = st.selectbox("Vehicle size", SIZES)
        transmission = st.selectbox(
            "Transmission type",
            TRANSMISSIONS,
            format_func=lambda t: t.replace("_", " ").title(),
        )
    submitted = st.form_submit_button("Predict price", use_container_width=True)

if submitted:
    X = build_input_row(year, engine_hp, make, size, transmission)
    raw_pred = float(model.predict(X)[0])
    price = float(np.expm1(raw_pred)) if LOG_TARGET else raw_pred

    st.success("Prediction complete")
    st.metric("Estimated price", f"${price:,.0f}")

    with st.expander("See the input the model received"):
        st.dataframe(X.loc[:, (X != 0).any()])  # show only the non-zero columns

st.caption(
    "Estimates come from a Decision Tree model. It cannot extrapolate beyond the data "
    "it was trained on, so very new years or unusual specs may be less reliable."
)
