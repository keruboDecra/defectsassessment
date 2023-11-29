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
    return prediction[0]

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

        # Display the predicted class and probability
        st.subheader("Prediction Results:")
        classes = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]
        max_prob_class = classes[np.argmax(prediction)]
        max_prob = np.max(prediction)

        st.write(f"Predicted Class: {max_prob_class}")
        st.write(f"Predicted Probability: {max_prob}")

        # Set a threshold for alerting
        threshold = 0.5

        if max_prob > threshold:
            st.success("Metal surface detected. Image is relevant for defect assessment.")
        else:
            st.warning("No relevant metal defect found. Please check the image again.")

# Run the app
if __name__ == '__main__':
    main()
