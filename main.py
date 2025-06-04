import streamlit as st
import tensorflow as tf
import numpy as np

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

# Title and Image
st.header("TOMATO DISEASE RECOGNITION SYSTEM")
image_path = "home-page.jpg"
st.image(image_path, use_column_width=True)

# Introduction Section
st.markdown("""
Welcome to the Tomato Disease Recognition System! 🌿🔍

Our mission is to help in identifying tomato diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

### How It Works
1. **Upload Image:** Upload an image of a plant's leaf with suspected diseases.
2. **Analysis:** Our system will process the image to identify potential diseases.
3. **Results:** View the results and recommendations for further action.

""")

# Disease Prediction Section
st.subheader("🌱 Upload an Image for Disease Recognition")
test_image = st.file_uploader("Choose an Image:")

if test_image is not None:
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Please Wait..."):
            st.write("Our Prediction")

            # Call your model prediction function
            result_index, confidence_score = model_prediction(test_image)

            # Define class names
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
            st.success(f"Model is predicting it's a **{class_name[result_index]}** with **{confidence_percent:.2f}%** confidence.")
