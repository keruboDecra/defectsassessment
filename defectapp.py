# Importing necessary libraries
import streamlit as st
import os
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

# Function to load the trained MobileNet model
def load_mobilenet_model():
    model_path = 'mobilenet_model (1).h5'

    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Function to make predictions
def predict_defect(img, model):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    return prediction

# Function to assess the highest probability predicted and print out the class of the image
def assess_defect(prediction, classes, threshold):
    max_prob_index = np.argmax(prediction)
    max_prob_class = classes[max_prob_index]

    # Check if the max probability is above the threshold
    if max(prediction[0]) >= threshold:
        return max_prob_class
    else:
        return "No relevant defect found"

# Streamlit App
def main():
    st.title("Defects Assessment App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "bmp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Load the model
        model = load_mobilenet_model()

        if model is not None:
            # Make predictions
            img = image.load_img(uploaded_file, target_size=(150, 150))
            prediction = predict_defect(img, model)

            # Display the results
            st.subheader("Prediction Results:")
            for i, class_name in enumerate(classes):
                st.write(f"{class_name}: {prediction[0][i]}")

            # Set a threshold for alerting
            threshold = st.slider("Set Threshold", 0.0, 1.0, 0.95, 0.01)
            max_prob_class = assess_defect(prediction[0], classes, threshold)
            st.success(f"This metal surface has a defect of: {max_prob_class}")

# Define your classes
classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Run the app
if __name__ == '__main__':
    main()
