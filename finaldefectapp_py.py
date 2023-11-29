# Import necessary libraries
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained MobileNet model
model = 'mobilenet_model(1).h5'

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize image to match the input size of the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Prediction function
def predict_defect(img_array):
    prediction = model.predict(img_array)
    return prediction

# Streamlit UI
st.title('Defects Assessment')

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make a prediction
    prediction = predict_defect(img_array)

    # Display the prediction result
    st.write("Prediction:")
    st.write(prediction)

    # You can further process the prediction result to display a meaningful output based on your classes.
    # For example, if you have classes like ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches'],
    # you can display the predicted class with the highest probability.
    predicted_class = np.argmax(prediction)
    st.write("Predicted Class:", predicted_class)
