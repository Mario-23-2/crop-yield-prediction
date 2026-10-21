# Crop Yield Prediction

A machine learning-based application designed to predict crop yield using a trained Python model.

## Features

- Interactive crop-yield prediction interface
- Model-powered predictions
- Streamlit-based application
- Large trained model hosted separately on Hugging Face

## Tech Stack

- Python
- Streamlit
- NumPy
- Pandas
- SciPy
- Statsmodels
- Joblib
- Hugging Face Hub

## Project Structure

- `app.py` — application interface and prediction logic
- `requirements.txt` — Python dependencies
- `.gitignore` — files excluded from version control

## Model

The trained model is hosted on Hugging Face Hub and downloaded by the application when required. The model file is intentionally excluded from this repository because of its size.

## Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Deployment

The application is intended to be deployed through Streamlit Community Cloud, with the trained model stored separately on Hugging Face Hub.
