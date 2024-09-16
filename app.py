import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Load your trained model (adjust the path as needed)
model = tf.keras.models.load_model('face_mask_detection_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  # Resize image to the input size of the model
    image = image / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return predictions

# Streamlit app
st.title("Face Mask Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    predictions = predict(image)

    # Print out raw prediction scores for debugging
    st.write("Raw Predictions:", predictions)

    # Adjust this threshold based on your model's performance
    threshold = 0.1

    if predictions[0][0] > threshold:
        st.write(f"Result: Mask (Confidence: {predictions[0][0]})")
    else:
        st.write(f"Result:  No Mask (Confidence: {predictions[0][0]})")

    if st.button("Show Confidence Scores"):
        st.write(predictions)
