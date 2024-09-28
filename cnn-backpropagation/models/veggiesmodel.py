from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import itertools

# CHECK IF IMPORTS ARE WORKING
print("Imports are working")

# GLOBALS
IMAGE_SIZE = [100,100]
epochs = 10
batch_size = 128

trainPath = 'data/veggies/trainRed'
testPath = 'data/veggies/testRed'

imageFiles = glob(trainPath + '/*/*.jp*g')
folders = glob(trainPath + '/*')

resNet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in resNet.layers:
    layer.trainable = False

# Layers
layers = Flatten()(resNet.output) # Flatten the output of the ResNet
layers = Dense(30, activation='relu')(layers)
layers = Dropout(0.2)(layers) # Dropout layer to prevent overfitting
layers = Dense(30, activation='relu')(layers)
layers = Dense(len(folders), activation='softmax')(layers)

model = Model(inputs=resNet.input, outputs=layers)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Training set
train_datagen = ImageDataGenerator(
    rotation_range=20, # Rotate the image
    width_shift_range=0.1, # Shift the width of the image
    height_shift_range=0.1, # Shift the height of the image
    shear_range=0.1, # Shear the image (skew)
    zoom_range=0.2, # Zoom in on the image
    horizontal_flip=True, # Flip the image horizontally
    vertical_flip=True, # Flip the image vertically
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

training_set = train_datagen.flow_from_directory(
    trainPath,
    target_size=IMAGE_SIZE,
    shuffle=True,
    batch_size=batch_size,
    class_mode='sparse' # sparse memory efficient
)

test_set = train_datagen.flow_from_directory(
    testPath,
    target_size=IMAGE_SIZE,
    shuffle=False,
    batch_size=batch_size,
    class_mode='sparse' # sparse memory efficient
)

# Train model
model.fit(
    training_set,
    validation_data=test_set,
    epochs=epochs
    )

model.save('models/veggies.keras')
