
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_REPO_ID = "Mariaaaah2/crop-yield-model"
MODEL_FILENAME = "crop_yield_model_compressed.joblib"

FEATURE_COLUMNS = [
    "Region",
    "Soil_Type",
    "Crop",
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
    "Weather_Condition",
    "Days_to_Harvest",
]


# =========================================================
# 2. LOAD MODEL
# =========================================================

@st.cache_resource(show_spinner="Downloading and loading the crop yield model...")
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    return joblib.load(model_path)


# =========================================================
# 3. LOAD CROP OPTIONS
# =========================================================

@st.cache_data
def load_options():
    options_path = BASE_DIR / "crop_options.json"

    with options_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required_keys = [
        "Region",
        "Soil_Type",
        "Crop",
        "Weather_Condition",
        "Fertilizer_Used",
        "Irrigation_Used",
    ]

    for key in required_keys:
        if key not in data or not isinstance(data[key], list) or not data[key]:
            raise ValueError(
                f"crop_options.json must contain a non-empty list for '{key}'."
            )

    return data


# =========================================================
# 4. INITIALIZE APP
# =========================================================

try:
    options = load_options()
    model = load_model()

except Exception as error:
    st.error("The app could not initialize successfully.")
    st.markdown(
        "Check that `crop_options.json` exists and that the "
        "Hugging Face model is accessible."
    )
    st.exception(error)
    st.stop()


# =========================================================
# 5. APP HEADER
# =========================================================

st.title("🌾 Crop Yield Prediction System")

st.write(
    "Estimate crop yield using region, soil conditions, "
    "weather, farming practices, and harvest duration."
)

st.divider()


# =========================================================
# 6. INPUT FORM
# =========================================================

st.subheader("Enter farming details")

with st.form("crop_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox(
            "Region",
            options=options["Region"]
        )

        soil_type = st.selectbox(
            "Soil Type",
            options=options["Soil_Type"]
        )

        crop = st.selectbox(
            "Crop",
            options=options["Crop"]
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-50.0,
            max_value=70.0,
            value=25.0,
            step=1.0
        )

    with col2:
        fertilizer = st.selectbox(
            "Fertilizer Used",
            options=options["Fertilizer_Used"]
        )

        irrigation = st.selectbox(
            "Irrigation Used",
            options=options["Irrigation_Used"]
        )

        weather = st.selectbox(
            "Weather Condition",
            options=options["Weather_Condition"]
        )

        days_to_harvest = st.number_input(
            "Days to Harvest",
            min_value=1,
            max_value=1000,
            value=90,
            step=1
        )

    submitted = st.form_submit_button(
        "Predict Crop Yield",
        type="primary",
        use_container_width=True
    )


# =========================================================
# 7. PREDICTION
# =========================================================

if submitted:

    input_data = pd.DataFrame(
        [{
            "Region": region,
            "Soil_Type": soil_type,
            "Crop": crop,
            "Rainfall_mm": rainfall,
            "Temperature_Celsius": temperature,
            "Fertilizer_Used": fertilizer,
            "Irrigation_Used": irrigation,
            "Weather_Condition": weather,
            "Days_to_Harvest": days_to_harvest,
        }],
        columns=FEATURE_COLUMNS
    )

    try:
        with st.spinner("Predicting crop yield..."):
            prediction = model.predict(input_data)

        if len(prediction) == 0:
            raise ValueError("The model returned no prediction.")

        predicted_yield = float(prediction[0])

        if not pd.notna(predicted_yield):
            raise ValueError("The model returned an invalid numeric result.")

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Crop Yield",
            value=f"{predicted_yield:,.2f}"
        )

        st.caption(
            "The displayed value uses the model's output units. "
            "Confirm the training dataset's target units before "
            "interpreting this as tonnes, kilograms, or another unit."
        )

        st.success("Prediction completed successfully!")

        # Show the values used for this prediction.
        with st.expander("View prediction inputs"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)


# =========================================================
# 8. FARMING ADVICE
# =========================================================

st.divider()
st.subheader("🌱 Farming Assistant")

st.write(
    "Use these general suggestions as a starting point. "
    "Actual recommendations depend on the crop, soil test, "
    "local climate, and agricultural guidance."
)

if submitted:
    advice = []

    if rainfall < 50:
        advice.append(
            "Rainfall input is low. Check soil moisture and "
            "consider whether supplemental irrigation is needed."
        )
    elif rainfall > 300:
        advice.append(
            "Rainfall input is high. Check drainage and monitor "
            "for waterlogging, erosion, and crop disease."
        )
    else:
        advice.append(
            "Compare the rainfall input with the water requirements "
            "of your selected crop and its growth stage."
        )

    if temperature < 10:
        advice.append(
            "The entered temperature is low for many crops. "
            "Check the selected crop's temperature tolerance."
        )
    elif temperature > 35:
        advice.append(
            "High temperatures can stress some crops. Monitor "
            "plant condition and soil moisture."
        )
    else:
        advice.append(
            "Check whether the entered temperature is suitable "
            "for your crop's current growth stage."
        )

    if str(irrigation).strip().lower() in ("no", "false", "0"):
        advice.append(
            "Irrigation is marked as unused. Monitor soil moisture "
            "and water availability, especially during dry periods."
        )

    advice.append(
        "Base fertilizer decisions on soil testing and crop-specific "
        "nutrient requirements rather than applying extra fertilizer "
        "solely to increase predicted yield."
    )

    for item in advice:
        st.write(f"• {item}")

else:
    st.info(
        "Enter the farming details above and click "
        "'Predict Crop Yield' to receive suggestions."
    )


# =========================================================
# 9. FOOTER
# =========================================================

st.divider()

st.caption(
    "Crop Yield Prediction System | Machine-learning estimate "
    "for informational and educational use."
)
