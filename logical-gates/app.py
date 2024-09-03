import numpy as np
from tensorflow.keras.models import load_model

## Load the models
model_and = load_model("models/pretrained-models/and-model.keras")
model_or = load_model("models/pretrained-models/or-model.keras")
model_xor = load_model("models/pretrained-models/xor-model.keras")


## PRETRAINED MODEL PREDICTIONS
print("AND MODEL PREDICTIONS:")
print(model_and.predict(np.array([[0,0],[1,0],[0,1],[1,1]])))
print("OR MODEL PREDICTIONS:")
print(model_or.predict(np.array([[0,0],[1,0],[0,1],[1,1]])))
print("XOR MODEL PREDICTIONS:")
print(model_xor.predict(np.array([[0,0],[1,0],[0,1],[1,1]])))

