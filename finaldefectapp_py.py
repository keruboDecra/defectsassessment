# Importing necessary libraries
import streamlit as st
import os
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

# Function to load the trained MobileNet model
def load_mobilenet_model():
    model_path = 'mobilenet_model (1).h5'
    model = load_model(model_path)
    return model

# Function to make predictions
def predict_defect(image_path, model):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction

# Function to assess the highest probability predicted
def assess_highest_probability(prediction, threshold):
    max_prob_index = np.argmax(prediction)
    max_prob = prediction[0][max_prob_index]

    if max_prob < threshold:
        st.warning("No relevant defect found. Please check the image again.")
    else:
        defect_class = classes[max_prob_index]
        st.write(f"This metal surface has a defect of {defect_class} with probability {max_prob}")

# Streamlit App
def main():
    st.title("Defects Assessment App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Create a temporary directory if it doesn't exist
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)

        # Create a path for the temporary image
        temp_path = os.path.join(temp_dir, 'temp_image.jpg')
        uploaded_file.seek(0)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Load the model
        model = load_mobilenet_model()

        # Make predictions
        prediction = predict_defect(temp_path, model)

        # Display the results
        st.subheader("Prediction Results:")
        assess_highest_probability(prediction, threshold=0.5)

# Run the app
if __name__ == '__main__':
    # Define the classes based on your training data
    classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

    main()
