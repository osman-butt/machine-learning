from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
model_numbers = load_model('models/number1to2.keras')

test_datagen_numbers = ImageDataGenerator(rescale = 1./255.)

test_set_letters = test_datagen_numbers.flow_from_directory(
    'data/two-digitsData/test',
    target_size = (28,28),
    batch_size = 32,
    class_mode = 'categorical', # for multi-class classification
    color_mode = 'grayscale')

# Predict a single image
# singlePred_numbers = "data/two-digitsData/test/2/1.jpg"
# test_image_numbers = image.load_img(singlePred_numbers, target_size = (28, 28), color_mode = "grayscale") # load image
# test_image_numbers = image.img_to_array(test_image_numbers) # convert image to array

# test_image_numbers = np.expand_dims(test_image_numbers, axis = 0) # add a dimension to the image

# result_numbers = model_numbers.predict(test_image_numbers/255.0) # predict the image
# predictedClass_numbers = np.argmax(result_numbers) # get the class index

# print("predictedClass: ")
# print(predictedClass_numbers)
print("---------------------------------")
print("Evaluate the numbers model")
# Evaluate the model
model_numbers.evaluate(test_set_letters)
print("---------------------------------")


# Load the saved model
model_letters = load_model('models/AB-classification.keras')

test_datagen_letters = ImageDataGenerator(rescale = 1./255.)

test_set_letters = test_datagen_letters.flow_from_directory(
    'data/letters/AB/test',
    target_size = (28,28),
    batch_size = 32,
    class_mode = 'categorical', # for multi-class classification
    color_mode = 'grayscale')

# Evaluate the model
print("---------------------------------")
print("Evaluate the letters model")
model_letters.evaluate(test_set_letters)
print("---------------------------------")