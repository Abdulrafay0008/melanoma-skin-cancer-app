import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Skin Cancer Detection App",
    page_icon="🩺",
    layout="centered"
)

# Model Load karne ka function (Cache use kiya hai taake bar bar load na ho)
@st.cache_resource
def load_model():
    # Yahan apni model file ka exact naam likhein jo aapne download kiya hai
    model = tf.keras.models.load_model("skin_cancer_model.h5")
    return model

# App loading state
with st.spinner('Model load ho raha hai, bara-e-karam intezaar karein...'):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model load karne mein masla aa raha hai: {e}")
        st.stop()

# UI Layout
st.title("🩺 Skin Cancer Detection App")
st.write("Apni skin lesion ki image upload karein taake trained CNN/NCC model prediction kar sake.")

# Image Upload Widget
uploaded_file = st.file_uploader("Image select karein (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image show karein
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Skin Image", use_column_width=True)
    
    st.write("---")
    
    if st.button("Predict Karo"):
        with st.spinner('Analysis ki ja rahi hai...'):
            # Image preprocessing (Model ki zaroorat ke mutabiq resize karein, misal ke tor par 224x224)
            image = image.resize((224, 224))
            
            # Image ko numpy array mein convert karein aur normalize karein
            img_array = np.array(image) / 255.0
            
            # Agar image RGBA (4 channels) hai to RGB (3 channels) mein convert karein
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
                
            # Batch dimension add karein (Model expects shape: (1, 224, 224, 3))
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction run karein
            prediction = model.predict(img_array)
            
            # Results display karein
            st.subheader("📊 Prediction Results:")
            
            # Agar aapka binary classification model hai (sigmoid output)
            if prediction.shape[-1] == 1 or prediction.shape[-1] == 2:
                score = float(prediction[0][0]) if prediction.shape[-1] == 1 else float(prediction[0][1])
                st.write(f"Confidence Score: **{score:.4f}**")
                if score > 0.5:
                    st.warning("⚠️ Result: Potential risk detected. Bar-e-karam kisi doctor ya specialist se rujoo karein.")
                else:
                    st.success("✅ Result: Normal / Benign appearance.")
            else:
                # Multi-class classification ke liye raw probabilities
                st.write("Model Output Probabilities:")
                st.write(prediction)

# Disclaimer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** Yeh application sirf educational aur research maqsad ke liye hai. Isey professional medical diagnosis ke taur par istemal na kiya jaye.")
