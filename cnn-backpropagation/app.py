from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/veggies.keras')

def preprocess_new_image(image_path):
    img = image.load_img(image_path, target_size=(100, 100))
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, height, width, 3)
    img_array = preprocess_input(img_array)  # Preprocess with ResNet50's preprocessing function
    return img_array

def predict_image_class(image_path):
    img_array = preprocess_new_image(image_path)
    prediction = model.predict(img_array)  # Make prediction
    predicted_class = np.argmax(prediction, axis=1)  # Get the index of the class with highest probability
    return predicted_class

# Example usage
image_path = 'data/veggies/validation/Tomato/1201.jpg'
predicted_class = predict_image_class(image_path)
print(f'Predicted class: {predicted_class}')