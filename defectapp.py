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

    # Display sample images horizontally
    sample_images = ['Crazing.bmp', 'inclusion.jpg', 'Patches.bmp', 'Pitted.bmp', 'Rolled.jpg', 'Scratches.bmp']
    
    # Define your classes
    classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
    
    # Load the model
    model = load_mobilenet_model()

    if model is None:
        st.warning("Model failed to load. Please check the logs for details.")
        return

    for sample_image in sample_images:
        img_path = os.path.join(os.getcwd(), sample_image)
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        if st.button(f"Use {sample_image}", key=sample_image):
            # Make predictions
            prediction = model.predict(np.expand_dims(img_array, axis=0))

            # Assess the highest probability predicted and print out the class
            max_prob_class = assess_defect(prediction[0], classes)

            # Display the detailed prediction results only if it's a defect class
            st.subheader(f"Prediction Results for {sample_image}:")
            for i, class_name in enumerate(classes):
                st.write(f"{class_name}: {prediction[0][i]}")

            st.success(f"This metal surface ({sample_image}) has a defect of: {max_prob_class}")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "bmp"])  # Allow BMP files

    # Add a sidebar for user inputs
    with st.sidebar:
        st.subheader("Threshold Settings")
        threshold = st.slider("Select Threshold", min_value=0.0, max_value=1.0, value=0.95)

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Create a temporary directory if it doesn't exist
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)

        # Create a path for the temporary image
        filename = uploaded_file.name
        temp_path = os.path.join(temp_dir, filename)

        uploaded_file.seek(0)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Make predictions
        prediction = predict_defect(temp_path, model)

        # Assess the highest probability predicted and print out the class
        max_prob_class = assess_defect(prediction[0], classes)

        # Set the threshold for alerting
        max_prob = max(prediction[0])
        if max_prob < threshold or max_prob_class == "Non-Metal":
            st.warning(f"This is likely not a metallic surface ({filename}), please check the image again.")
        else:
            # Display the detailed prediction results only if it's a defect class
            st.subheader(f"Prediction Results for {filename}:")
            for i, class_name in enumerate(classes):
                st.write(f"{class_name}: {prediction[0][i]}")

            st.success(f"This metal surface ({filename}) has a defect of: {max_prob_class}")

# Run the app
if __name__ == '__main__':
    main()
