import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from matplotlib import pyplot

print("Classifying images of letters A and B")

# Part 1 -  ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.)
train_set = train_datagen.flow_from_directory(
    'data/letters/AB/train',
    target_size = (28,28),
    batch_size = 32,
    class_mode = 'categorical', # for multi-class classification
    color_mode = 'grayscale')

# Part 2 - Convolution
model = Sequential()
model.add(Conv2D(
    filters=2, 
    kernel_size=3, 
    activation='relu',
    input_shape=[28, 28, 1] # 1 for grayscale, 3 for RGB
    ))

# Part 3 - Max Pooling
# Max pooling is used to reduce the spatial dimensions of the output volume.
model.add(MaxPool2D(pool_size=2, strides=2))

# Part 4 - CNN Deep Neural Network

# Flattening = From matrix to vector
model.add(Flatten()) # Flattening is converting the data into a 1-dimensional array for inputting it to the next layer

# Full Connection
model.add(Dense(
    units=10, # number of neurons
    activation='relu')) # Hidden layer

model.add(Dense(units=2, activation='softmax')) # Output layer

adam = Adam(learning_rate=0.001) # Adam optimizer

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=train_set, epochs=10)

model.save('models/AB-classification.keras')