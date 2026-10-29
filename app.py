import time
import joblib
import streamlit as st
from prometheus_client import Counter, Histogram, start_http_server
from sklearn.datasets import load_iris

NAMES = load_iris().target_names


@st.cache_resource
def setup():
    """Load model and start the Prometheus metrics endpoint (port 8000) once per process."""
    start_http_server(8000)
    return (
        joblib.load("model.joblib"),
        Counter("iris_predictions_total", "Predictions served", ["species"]),
        Histogram("iris_prediction_latency_seconds", "Prediction latency"),
    )


model, PRED, LAT = setup()

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("🌸 Iris Species Classifier")
st.caption("MLOps Lab 8 - Aayush Nayak, E007, BTI DS Sem 11, E1")

c1, c2 = st.columns(2)
sl = c1.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
sw = c1.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
pl = c2.slider("Petal length (cm)", 1.0, 7.0, 4.3)
pw = c2.slider("Petal width (cm)", 0.1, 2.5, 1.3)

if st.button("Predict"):
    t0 = time.perf_counter()
    probs = model.predict_proba([[sl, sw, pl, pw]])[0]
    LAT.observe(time.perf_counter() - t0)
    species = NAMES[probs.argmax()]
    PRED.labels(species).inc()
    st.success(f"Predicted species: **{species}** ({probs.max():.0%} confidence)")
    st.bar_chart(dict(zip(NAMES, probs)))
