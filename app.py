import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet50 import preprocess_input

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MODEL_PATH = "cifar10_resnet50.keras"

st.set_page_config(page_title="CIFAR-10 Object Recognition", page_icon="🖼️")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict(img: Image.Image) -> np.ndarray:
    # Same pipeline as training: 32x32 RGB -> ResNet50 preprocess_input.
    # (Resizing to 96x96 happens inside the saved model.)
    img = ImageOps.fit(img.convert("RGB"), (32, 32), Image.LANCZOS)  # centre-crop + resize
    arr = preprocess_input(np.asarray(img, dtype="float32")[None, ...])
    return model.predict(arr, verbose=0)[0]


st.title("🖼️ CIFAR-10 Object Recognition")
st.write("ResNet50 (transfer learning) trained on CIFAR-10, about 94% test accuracy. "
         "Upload a photo of one of: " + ", ".join(CLASSES) + ".")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`. Make sure it is in the same folder as app.py.\n\n{e}")
    st.stop()

SAMPLES_DIR = "samples"
sample_files = sorted(os.listdir(SAMPLES_DIR)) if os.path.isdir(SAMPLES_DIR) else []

options = ["Upload", "Camera"] + (["Example photos"] if sample_files else [])
source = st.radio("Image source", options, horizontal=True)
if source == "Upload":
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
elif source == "Camera":
    file = st.camera_input("Take a photo")
else:
    choice = st.selectbox("Pick an example", sample_files,
                          format_func=lambda f: os.path.splitext(f)[0].replace("_", " #"))
    file = os.path.join(SAMPLES_DIR, choice)

if file is not None:
    image = Image.open(file)
    probs = predict(image)
    top = int(np.argmax(probs))

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Your image", width="stretch")
    with col2:
        st.subheader(f"Prediction: {CLASSES[top].title()}")
        st.metric("Confidence", f"{probs[top]:.1%}")
        order = np.argsort(-probs)[:3]
        st.write("Top 3")
        st.bar_chart(pd.DataFrame({"probability": probs[order]},
                                  index=[CLASSES[i] for i in order]))
    if probs[top] < 0.6:
        st.info("Low confidence. The model only knows these 10 classes, "
                "so photos of anything else will be forced into one of them.")

with st.sidebar:
    st.header("About")
    st.write("Model: ResNet50 pretrained on ImageNet, fine-tuned on 40,000 CIFAR-10 images.")
    st.write("Test accuracy: **93.76%** on 5,000 held-out images.")
    st.caption("Images are shrunk to 32x32 (like CIFAR-10), so small, centred objects work best.")