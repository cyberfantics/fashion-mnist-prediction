import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image

# Load Fashion MNIST dataset for reference
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Load the pre-trained model for Fashion MNIST
model_path = 'model/fashion_mnist_model.keras'  # Ensure this is a model trained on Fashion MNIST

model = load_model(
    model_path, 
    custom_objects={'softmax_v2': tf.nn.softmax}
)

# Set the page configuration
st.set_page_config(page_title="Fashion MNIST Classification", layout="wide")

# Sidebar configuration
st.sidebar.title("Fashion MNIST Classification")
num_images = st.sidebar.slider('Number of images to display from test set:', min_value=1, max_value=20, value=5)
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Define the class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display app title and description
st.title("👗 Fashion MNIST Clothing Classification")
st.markdown("""
    This app classifies clothing items from the Fashion MNIST dataset. You can either view random images from the test set or upload your own image for prediction!
""")

# Show uploaded image if available
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert the image to grayscale
    image = image.resize((28, 28))  # Resize the image to 28x28

    # Display the uploaded image
    st.subheader("Uploaded Image")
    st.image(image, caption='Uploaded Image', use_column_width=False)

    # Preprocess the image for prediction
    image_array = np.array(image)
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape to match input shape (1, 28, 28, 1)
    image_array = image_array.astype('float32') / 255  # Normalize pixel values to range [0,1]

    # Make prediction
    pred_prob = model.predict(image_array)
    pred_label = np.argmax(pred_prob, axis=1)[0]

    st.subheader(f"Prediction for Uploaded Image: **{class_names[pred_label]}**")

# Select random images from the test set
sample_indices = np.random.choice(x_test.shape[0], num_images, replace=False)
x_valid = x_test[sample_indices]
y_valid = y_test[sample_indices]

# Reshape the data to fit the model's input expectations
x_valid_reshaped = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
x_valid_reshaped = x_valid_reshaped.astype('float32') / 255  # Normalize the data

# Predict the probabilities for the selected samples
y_pred_prob = model.predict(x_valid_reshaped)
y_pred = np.argmax(y_pred_prob, axis=1)

# Display the selected images
st.subheader('Selected Fashion MNIST Images and Predictions')
fig, axs = plt.subplots(2, num_images, figsize=(15, 4))
for i in range(num_images):
    # Display the image
    axs[0, i].imshow(x_valid[i], cmap='gray')
    axs[0, i].axis('off')
    
    # Display the prediction and true label
    axs[1, i].text(0.5, 0.7, f'Pred: {class_names[y_pred[i]]}', fontsize=12, ha='center', wrap=True)
    axs[1, i].text(0.5, 0.3, f'True: {class_names[y_valid[i]]}', fontsize=12, ha='center', wrap=True)
    axs[1, i].axis('off')

st.pyplot(fig)

# Add a footer with developer information
st.markdown("""
    <hr>
    <p style='font-size:16px; color:#4CAF50;'>Developed by <b>Syed Mansoor ul Hassan Bukhari</b></p>
    <p style='font-size:16px;'>For more details, visit the <a href=|https://github.com/cyberfantics/fashion-mnist-prediction" style="color:#4CAF50;">GitHub Repository</a>.</p>
""", unsafe_allow_html=True)
