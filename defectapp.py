# Importing necessary libraries
import streamlit as st
import os
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

# Function to check if an image is grayscale
def is_grayscale(img_path, min_unique_values=0, max_unique_values=255):
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    unique_values = np.unique(img_array)
    return min_unique_values <= len(unique_values) <= max_unique_values

# Streamlit App
def main():
    st.title("Defects Assessment App")

    # Subtitle
    st.subheader("You can either use these images")

    # Display sample images horizontally
    sample_images = ['Crazing.bmp', 'inclusion.jpg', 'Patches.bmp', 'Pitted.bmp', 'Rolled.jpg', 'Scratches.bmp']
    sample_columns = st.columns(len(sample_images))
    for col, sample_image in zip(sample_columns, sample_images):
        img_path = os.path.join(os.getcwd(), sample_image)
        col.image(img_path, caption=f"Use {sample_image}", use_column_width=True)
        if col.button(f"Use {sample_image}"):
            process_sample_image(sample_image)

    # Or upload a file here
    st.subheader("Or upload a file here:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp"], key="file_uploader")  # Allow BMP files

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

        # Check if the image is grayscale
        if not is_grayscale(temp_path):
            st.warning(f"The image, {filename}, is likely not a metallic surface, please check the image again.")
        else:
            # Load the model
            model = load_mobilenet_model()

            if model is not None:
                # Make predictions
                prediction = predict_defect(temp_path, model)

                # Assess the highest probability predicted and print out the class
                max_prob_class = assess_defect(prediction[0], classes)

                # Set the threshold for alerting
                max_prob = max(prediction[0])
                if max_prob < threshold or max_prob_class == "Non-Metal":
                    st.warning(f"The image, {filename}, is likely not a metallic surface, please check the image again.")
                else:
                    # Display the detailed prediction results only if it's a defect class
                    st.subheader(f"Prediction Results for {filename}:")
                    for i, class_name in enumerate(classes):
                        st.write(f"{class_name}: {prediction[0][i]}")

                    st.success(f"This metal surface ({filename}) has a defect of: {max_prob_class}")

# Function to process a sample image
def process_sample_image(sample_image):
    # Load the model
    model = load_mobilenet_model()

    if model is not None:
        # Create a temporary directory if it doesn't exist
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)

        # Create a path for the temporary image
        temp_path = os.path.join(temp_dir, sample_image)

        # Load the sample image
        img_path = os.path.join(os.getcwd(), sample_image)
        img = image.load_img(img_path, target_size=(150, 150))
        img.save(temp_path)

        # Make predictions
        prediction = predict_defect(temp_path, model)

        # Assess the highest probability predicted and print out the class
        max_prob_class = assess_defect(prediction[0], classes)

        # Display the detailed prediction results only if it's a defect class
        st.subheader(f"Prediction Results for {sample_image}:")
        for i, class_name in enumerate(classes):
            st.write(f"{class_name}: {prediction[0][i]}")

        st.success(f"This metal surface ({sample_image}) has a defect of: {max_prob_class}")

# Define your classes
classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Run the app
if __name__ == '__main__':
    main()
