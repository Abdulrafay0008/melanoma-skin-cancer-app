import streamlit as tf_streamlit
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Configuration
tf_streamlit.set_page_config(
    page_title="Melanoma Skin Cancer Classification",
    page_icon="🩺",
    layout="centered"
)

# Cache the model loading to prevent reloading on every prediction
@tf_streamlit.cache_resource
def load_model():
    # Load your trained model file
    model = tf.keras.models.load_model("melanoma_ncc_model.h5")
    return model

tf_streamlit.title("🩺 Melanoma Skin Cancer Detection")
tf_streamlit.markdown(
    "Upload a skin lesion image to classify it using your trained Convolutional Neural Network (NCC) model."
)

# Load model
with tf_streamlit.spinner("Loading model weights..."):
    model = load_model()

# Class names (Update these to match your actual dataset classes, e.g., ['benign', 'malignant'])
class_names = ['Benign', 'Malignant'] 

# File Uploader
uploaded_file = tf_streamlit.file_uploader(
    "Choose a skin lesion image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    tf_streamlit.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image for the model
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # Rescale to [0, 1] matching training pipeline
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension -> shape (1, 224, 224, 3)
    
    if tf_streamlit.button("Predict"):
        with tf_streamlit.spinner("Analyzing image..."):
            predictions = model.predict(img_array)
            
            # Handle binary vs categorical output mapping
            if predictions.shape[1] == 1:
                score = float(predictions[0][0])
                # Assuming sigmoid output: > 0.5 is malignant, else benign (adjust based on your training labels)
                predicted_class = class_names[1] if score > 0.5 else class_names[0]
                confidence = score if score > 0.5 else (1 - score)
            else:
                score_idx = np.argmax(predictions[0])
                predicted_class = class_names[score_idx]
                confidence = float(predictions[0][score_idx])
            
            # Display results
            tf_streamlit.success(f"**Prediction:** {predicted_class}")
            tf_streamlit.info(f"**Confidence Score:** {confidence * 100:.2f}%")
            
            if predicted_class == 'Malignant':
                tf_streamlit.warning("⚠️ **Alert:** This scan indicates potential malignancy. Please consult a professional dermatologist immediately.")
            else:
                tf_streamlit.info("ℹ️ **Notice:** This scan appears benign, but always consult a certified medical practitioner for a definitive diagnosis.")