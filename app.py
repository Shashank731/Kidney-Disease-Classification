import streamlit as st
from PIL import Image
from src.kidney_disease_prediction.pipeline.prediction import PredictionPipeline

st.title("Kidney Disease Classification")

uploaded_file = st.file_uploader(
    "Upload CT Scan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    predictor = PredictionPipeline(
        "artifacts/training/trained_model.pth"
    )

    prediction = predictor.predict(image)

    st.success(f"Prediction: {prediction}")