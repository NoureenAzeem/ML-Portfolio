import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load both models
@st.cache_resource
def load_models():
    vgg16_model = tf.keras.models.load_model("vgg16_flowers.h5")
    mobilenet_model = tf.keras.models.load_model("mobilenet_flowers.h5")
    return vgg16_model, mobilenet_model

vgg16_model, mobilenet_model = load_models()

# Class labels (make sure these match your dataset)
class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

st.title("🌸 Flower Classification App")
st.write("Upload an image of a flower and choose a model to classify it.")

# Dropdown for model selection
model_choice = st.selectbox("Choose a model:", ["VGG16", "MobileNetV2"])

# Upload image
uploaded_file = st.file_uploader("Upload a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Choose model
    if model_choice == "VGG16":
        predictions = vgg16_model.predict(img_array)
    else:
        predictions = mobilenet_model.predict(img_array)

    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.write(f"### 🌼 Prediction: {predicted_class} ({confidence:.2f}% confidence)")