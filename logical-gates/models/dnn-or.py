import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

model = Sequential()
model.add(Dense(1, input_dim=2, activation="sigmoid", use_bias=True))
adam = optimizers.Adam(learning_rate=0.1)
model.compile(loss='mse',optimizer=adam)

model.summary()

## Input / output data
x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[0],[1],[1],[1]])

## Train the model
print("Fitting the model")
model.fit(x, y, epochs=1000, batch_size=2, verbose=0)

## Evaluate the model
print("Predictions:")
print(model.predict(x))

## Save the model
model.save("models/pretrained-models/or-model.keras")