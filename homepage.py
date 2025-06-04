import streamlit as st
import tensorflow as tf
import numpy as np
import base64

#Tensorflow Model Prediction
def model_prediction(test_image):
    model  = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence_score = prediction[0][result_index]
    return result_index, confidence_score


# Page Configuration
st.set_page_config(page_title="Plant Disease Recognition", layout="centered")

# Custom Styling (optional CSS for soft layout)
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .prediction {
        background-color: #e0f7fa;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        color: #004d40;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.image("home-page.jpg", use_column_width=True)
st.markdown("## 🌿 Plant Disease Recognition System")
st.write("""
Welcome to the **Plant Disease Recognition System!**  
Our mission is to help farmers and plant lovers identify plant diseases accurately and efficiently.

---

### 🔍 How It Works
1. **Upload an image** of a plant leaf.
2. **Model analyzes** it using machine learning.
3. **You get results** instantly with a prediction and confidence level.

---

### 🚀 Why Use This Tool?
- ✅ High Accuracy
- 🎯 Fast & Reliable
- 🤖 Powered by ML
- 🧑‍🌾 Helpful for Farmers & Researchers

---
""")

# Upload Section
st.markdown("## 📸 Upload a Leaf Image")
test_image = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"])

if test_image is not None:
    if st.button("📷 Show Image"):
        st.image(test_image, use_column_width=True)

    if st.button("🧠 Predict Disease"):
        with st.spinner("Analyzing Image... Please wait..."):
            st.markdown("### 🔬 Prediction Result")

            # Call your model prediction function
            result_index, confidence_score = model_prediction(test_image)

            # Disease class names
            class_name = [
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            confidence_percent = confidence_score * 100
            st.markdown(f"""
            <div class='prediction'>
            🌱 Prediction: <strong>{class_name[result_index]}</strong><br>
            📊 Confidence: <strong>{confidence_percent:.2f}%</strong>
            </div>
            """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


