from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model('models/number1to2.keras')

# Predict a single image
singlePred = "data/two-digitsData/test/2/1.jpg"
test_image = image.load_img(singlePred, target_size = (28, 28), color_mode = "grayscale") # load image
test_image = image.img_to_array(test_image) # convert image to array

test_image = np.expand_dims(test_image, axis = 0) # add a dimension to the image

result = model.predict(test_image/255.0) # predict the image
predictedClass = np.argmax(result) # get the class index

print("predictedClass: ")
print(predictedClass)