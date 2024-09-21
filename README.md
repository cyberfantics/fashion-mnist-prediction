# Fashion MNIST Clothing Classification App

![App Screenshot](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

This Streamlit web app allows users to upload images of clothing items or explore random test images from the **Fashion MNIST** dataset. The app uses a pre-trained model to predict the clothing category, making it an educational and interactive tool for exploring image classification in machine learning.

## 🎯 **Features**

1. **Home Page:**
   - Provides an overview of the Fashion MNIST dataset and explains how to navigate through the app.
   - Displays a visual of the Fashion MNIST dataset to give a better understanding of the clothing items.

2. **Upload & Predict:**
   - Allows users to upload their own clothing images in PNG, JPG, or JPEG format.
   - The model predicts the clothing category and displays it along with the uploaded image.

3. **Random Test Images & Predictions:**
   - Displays random images from the Fashion MNIST test set in **colorful format**.
   - Shows both the predicted and true labels of these images, allowing you to see how well the model performs.

## 🌐 **Live Demo**

Try the app live on Streamlit:  
👉 [Fashion MNIST Prediction App](https://cyberfantics-fashion-mnist-prediction-app-r2pvy5.streamlit.app/)

## 🚀 **Retrain the Model**

You can retrain the model by adjusting the network architecture and hyperparameters in the `mnist_fashion_dataset.ipynb` notebook. The notebook provides an interactive environment where you can experiment with various configurations to improve the model's performance.

Explore and modify the notebook here:  
👉 [Fashion MNIST Retrain Notebook](https://github.com/cyberfantics/fashion-mnist-prediction)

## 📁 **Project Structure**

- **`app.py`**: The main Streamlit application file.
- **`model/fashion_mnist_model.keras`**: Pre-trained model used for making predictions.
- **`mnist_fashion_dataset.ipynb`**: Jupyter notebook for retraining and experimenting with the model.
- **`README.md`**: Project documentation (this file).
  
## 📚 **About the Dataset**

The **Fashion MNIST** dataset consists of 60,000 training images and 10,000 testing images of various clothing items. The goal is to classify these images into one of the following categories:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Each image is 28x28 pixels, and the dataset serves as a more challenging alternative to the classic MNIST digits dataset.

## 🔧 **How to Run Locally**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cyberfantics/fashion-mnist-prediction.git
   cd fashion-mnist-prediction
   ```

2. **Install dependencies:**
   `pip install -r requirements.txt`

3. **Run the App**
   `streamlit run app.py`

**Open your browser to `http://localhost:8501/` to interact with the app.**

## 🔨 Customization
You can retrain the model by playing with the network architecture and hyperparameters in the provided `mnist_fashion_dataset.ipynb` notebook. Feel free to experiment and improve the model’s accuracy!

## Steps to Contribute:
- Fork the repository.
- Create a new feature branch.
- Make your changes.
- Submit a pull request.

## Developed by `Syed Mansoor ul Hassan Bukhari`
