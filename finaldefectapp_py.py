# Import necessary libraries
from keras.models import load_model
from keras.preprocessing import image

# Load the trained MobileNet model
model = 'mobilenet_model(1).h5'

# Define the input size for the model
img_width, img_height = 150, 150

# Function to preprocess the image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array

# Streamlit app
st.title("Defects Assessment App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Make a prediction when an image is uploaded
if uploaded_file is not None:
    # Preprocess the image
    img = preprocess_image(uploaded_file)

    # Make prediction
    prediction = model.predict(img)

    # Get the predicted class
    predicted_class = np.argmax(prediction)
    classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
    predicted_class_name = classes[predicted_class]

    # Display the image and prediction
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Class Prediction: ", predicted_class_name)
    st.write("Confidence: {:.2f}%".format(prediction[0][predicted_class] * 100))

    # Bar chart for class probabilities
    fig, ax = plt.subplots()
    ax.bar(classes, prediction[0] * 100)
    ax.set_ylabel('Probability (%)')
    ax.set_title('Class Probabilities')
    st.pyplot(fig)
