import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score

## Import data
dataframe = pd.read_csv('data/customer_staying_or_not.csv')

# print(dataframe.head())
# print(dataframe.isnull().sum())

## Remove NA values
dataframe.dropna(inplace=True)
print(dataframe.head())

## Extract X and y
X = dataframe.iloc[:, 3:13]
print(X.head())

y = dataframe.iloc[:, -1]
print(y.head())

## Convert categorical to numerical
X= pd.get_dummies(X)

## Save column names
columnNames = list(X.columns)

## Convert Panda to Numpy
X = X.values
y = y.values

## Normalizing/scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ## How scaling works
# data = np.array([[2.0,30.0],[5.0, 60.0], [8.0, 90.0]])
# scaler = StandardScaler()
# standardized_data = scaler.fit_transform(data)
# print (standardized_data)

## Split the data (80% train, 20% test) (train_test_split returns a tuple)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ## Tuple example
# myList = [1,2,3]
# a,b,c = myList


## Create the model
model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Sigmoid for binary classification

adam = Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

## Train the model
model.fit(x=X_train, y=y_train, epochs=100, verbose=0)

## Accuracy

# Loss
loss = model.history.history['loss']
lossPlot = sns.lineplot(x=range(len(loss)), y=loss)
fig = lossPlot.get_figure()
fig.savefig('loss.png')

# Evaluate the model
model.evaluate(X_test, y_test, verbose=1)

# Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
confussionMatrix = confusion_matrix(y_test, y_pred)
# TN FP
# FN TP
# bias towards false negatives
# Accuracy
# accuracy = model.history.history['accuracy']

print(columnNames)

# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 
# Tenure: 3 years
# Balance: $ 60000
# Number of Products: 2
# Does this customer have a credit card? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $ 50000

# ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France', 'Geography_Germany', 'Geography_Spain', 'Gender_Female', 'Gender_Male']
newCustomer = [[600, 40, 3, 60000, 2, 1, 1, 50000, 1, 0 , 0, 0, 1]]
newCustomer = scaler.transform(newCustomer)

print(model.predict(newCustomer))

model.save('leaveStayModel.keras')