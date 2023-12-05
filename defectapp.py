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
def predict_defect(image_path, model):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    return prediction

# Function to assess the highest probability predicted and print out the class of the image
def assess_defect(prediction, classes):
    max_prob_index = np.argmax(prediction)
    max_prob_class = classes[max_prob_index]
    return max_prob_class

# Streamlit App
def main():
    st.title("Defects Assessment App")

    # Add a section for sample images
    st.sidebar.subheader("Choose a Sample Image:")
    sample_images = {
        "Crazing": "Crazing.bmp",
        "Inclusion": "Inclusion.jpg",
        "Patches": "Patches.bmp",
        "Pitted": "Pitted.bmp",
        "Rolled": "Rolled.jpg",
        "Scratches": "Scratches.bmp",
    }

    sample_choice = st.sidebar.selectbox("Select a sample image:", list(sample_images.keys()))

    # Display selected sample image
    sample_path = os.path.join("samples", sample_images[sample_choice])
    st.image(sample_path, caption=f"Sample Image: {sample_choice}", use_column_width=True)

    # Add a sidebar for user inputs
    with st.sidebar:
        st.subheader("Threshold Settings")
        threshold = st.slider("Select Threshold", min_value=0.0, max_value=1.0, value=0.95)

    # Load the model
    model = load_mobilenet_model()

    if model is not None:
        # Make predictions for the selected sample image
        prediction = predict_defect(sample_path, model)

        # Assess the highest probability predicted and print out the class
        max_prob_class = assess_defect(prediction[0], classes)

        # Set the threshold for alerting
        max_prob = max(prediction[0])
        if max_prob < threshold or max_prob_class == "Non-Metal":
            st.warning(f"This is likely not a metallic surface ({sample_choice}), please check the image again.")
        else:
            # Display the detailed prediction results only if it's a defect class
            st.subheader(f"Prediction Results for {sample_choice}:")
            for i, class_name in enumerate(classes):
                st.write(f"{class_name}: {prediction[0][i]}")

            st.success(f"This metal surface ({sample_choice}) has a defect of: {max_prob_class}")

# Define your classes
classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Run the app
if __name__ == '__main__':
    main()
