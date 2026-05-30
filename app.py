import streamlit as st
from PIL import Image
from src.kidney_disease_prediction.pipeline.prediction import PredictionPipeline

st.set_page_config(
    page_title="Kidney Disease Classifier",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_predictor():
    return PredictionPipeline(
        "models/trained_model.pth"
    )
predictor = load_predictor()

st.title("🩺 Kidney Disease Classification")
st.markdown(
    "Upload a Kidney CT scan image and let the AI model classify it."
)

with st.sidebar:
    st.header("About")
    st.write(
        """
        This AI system classifies Kidney CT scans into:
        - Cyst
        - Normal
        - Stone
        - Tumor
        """
    )

    st.info(
        "Upload only Kidney CT scan images."
    )

uploaded_file = st.file_uploader(
    "Upload CT Scan",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.write("### Ready for Prediction")

        if st.button(
            "🔍 Predict",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing CT Scan..."
            ):

                try:

                    prediction, confidence = predictor.predict(image)

                    st.success(
                        f"Prediction: {prediction}"
                   )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                 )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {e}"
                    )

st.markdown("---")
st.caption(
    "DenseNet121 • PyTorch • Streamlit"
)