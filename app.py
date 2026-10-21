
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

st.title("Crop Yield Model Test")

st.write("Step 1: Streamlit started successfully.")

try:
    with st.spinner("Downloading model from Hugging Face..."):
        model_path = hf_hub_download(
            repo_id="Mariaaaah2/crop-yield-model",
            filename="crop_yield_model_compressed.joblib",
            repo_type="model"
        )

    st.success("Step 2: Model downloaded successfully.")

    with st.spinner("Loading model..."):
        model = joblib.load(model_path)

    st.success("Step 3: Model loaded successfully!")
    st.write("Model type:", type(model).__name__)

except Exception as error:
    st.error("The model download or loading failed.")
    st.exception(error)
