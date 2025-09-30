import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from model import unet

model = unet()
model.load_weights("models/pothole_model.h5")

st.title("Pothole Detection App")
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    mask = (prediction > 0.5).astype(np.uint8) * 255

    st.image([img[0], mask[0]], caption=["Input", "Pothole Mask"], use_column_width=True)
