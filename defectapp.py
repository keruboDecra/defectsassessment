# Importing necessary libraries
import streamlit as st
import os
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import pandas as pd


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
# Function to assess the highest probability predicted and print out the class of the image
def assess_defect(prediction, classes):
    if len(prediction) == 0:
        # Handle the case where the prediction array is empty
        return None, None

    max_prob_index = np.argmax(prediction)
    
    if max_prob_index >= len(classes):
        # Handle the case where the max_prob_index is out of bounds
        return None, None

    max_prob_class = classes[max_prob_index]
    return max_prob_class, prediction[max_prob_index]


# Function to check if the uploaded image is relevant to the task
import numpy as np

def is_relevant_image(image_path, gray_threshold=0.8):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the percentage of pixels in shades of gray
    gray_percentage = np.count_nonzero(gray_img < 200) / gray_img.size

    # Check if the image is predominantly in shades of gray
    return gray_percentage > gray_threshold


# Streamlit App with Dashboard
def main():
    st.title("Defects Assessment App")

    # Create a simple dashboard
    st.sidebar.title("Dashboard")

    # Slider for selecting probability threshold
    selected_threshold = st.sidebar.slider("Select Probability Threshold", 0.0, 1.0, 0.95)

    # Slider for selecting result display percentage
    result_percentage = st.sidebar.slider("Select Result Display Percentage", 0, 100, 50)

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

        # Check if the uploaded image is relevant to the task
        if is_relevant_image(temp_path):
            # Load the model
            model = load_mobilenet_model()

            if model is not None:
                # Display the "Classify" button
                if st.button("Classify"):
                    # Make predictions
                    prediction = predict_defect(temp_path, model)

                    # Display the results
                    st.subheader("Prediction Results:")
                    for i, class_name in enumerate(classes):
                        st.write(f"{class_name}: {prediction[0][i]:.4f}")

                    # Display the probability dashboard
                    st.subheader("Probability Dashboard:")
                    probabilities = {class_name: prediction[0][i] for i, class_name in enumerate(classes)}

                    # Convert the dictionary to a DataFrame
                    df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
                    st.bar_chart(df.set_index('Class'))

                    # Assess the highest probability predicted and print out the class
                    max_prob_class, max_prob = assess_defect(prediction[0], classes)

                    # Check if any predicted probability exceeds the threshold
                    relevant_defect_found = any(prob >= selected_threshold for prob in prediction[0])

                    # Display result based on the selected result percentage and reject non-relevant images
                    if relevant_defect_found:
                        st.success(f"This metal surface has a defect of: {max_prob_class}")
                        # st.write(f"Result: {max_prob * result_percentage:.4f}")
                    else:
                        st.warning("No relevant defect found. Please check the image again.")
                        st.write(f"Highest predicted class: {max_prob_class}")
                        st.write(f"Probability: {max_prob * result_percentage:.4f}")
        else:
            st.warning("The uploaded image is not relevant to the task.")

# Define your classes
classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Run the app
if __name__ == '__main__':
    main()
