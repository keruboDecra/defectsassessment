# Import necessary libraries
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image

# Load the trained MobileNet model
model = 'mobilenet_model(1).h5'

# Define the image size for model input
img_width, img_height = 150, 150

# Function for image preprocessing
def preprocess_image(img):
    # Resize the image to the required dimensions
    img = img.resize((img_width, img_height))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand the dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values to the range [0, 1]
    img_array /= 255.0
    return img_array

# Function for model prediction
def predict_defect(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img_array = preprocess_image(img)
    
    # Make predictions
    predictions = model.predict(img_array)
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)
    
    return predicted_class

# Streamlit user interface
st.title("Defect Assessment App")

# Upload image through the Streamlit interface
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Make predictions when the "Predict" button is clicked
    if st.button("Predict"):
        # Get the predictions
        prediction = predict_defect(uploaded_file)

        # Display the prediction result
        defect_classes = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]
        result = f"The predicted defect class is: {defect_classes[prediction]}"
        st.success(result)
