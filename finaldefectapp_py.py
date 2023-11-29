# Import necessary libraries
import streamlit as st
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

# Load the trained MobileNet model
model = 'mobilenet_model(1).h5'

# Function to preprocess an image for prediction
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Streamlit app
st.title("Defect Classification App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Make a prediction
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = preprocess_image(uploaded_file)

    # Make prediction
    prediction = model.predict(img_array)

    # Get the class labels
    classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

    # Display the prediction result
    st.write(f"Prediction: {classes[np.argmax(prediction)]}")
