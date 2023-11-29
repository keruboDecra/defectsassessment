# Import necessary libraries
import streamlit as st
import numpy as np

from keras.models import load_model
from keras.preprocessing import image

# Load the trained model
model = 'mobilenet_model(1).h5'  # Update with the actual path

# Define the defect classes
classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Function to preprocess the input image
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict_defect(image_path):
    # Preprocess the input image
    processed_img = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(processed_img)

    # Get the predicted class
    predicted_class = classes[np.argmax(prediction)]

    return predicted_class

# Streamlit app
def main():
    st.title("Defects Assessment App")
    st.write("Upload an image to assess defects.")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make prediction
        predicted_class = predict_defect(uploaded_file)

        st.write(f"Prediction: {predicted_class}")

# Run the app
if __name__ == '__main__':
    main()
